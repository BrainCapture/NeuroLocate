# Copyright 2026 NeuroLocate contributors
# SPDX-License-Identifier: Apache-2.0
r"""Gate 3: fine-tuning the proposal network *through* the physics solver.

The supervised loss teaches the network where sources are. It cannot teach it
what the sensors say, because it never evaluates a forward model. This adds a
sensor-space term whose gradient reaches the network's parameters by way of
OpenMEEG:

.. math::

    L = \lambda_{\text{source}}\, L_{\text{source}}
      + \lambda_{\text{sensor}}\, L_{\text{physics}}

``L_source`` is the heatmap/offset/moment/count loss of
:mod:`neurolayout.hybrid.losses`, in PyTorch. ``L_physics`` is the projector
residual of :mod:`neurolayout.hybrid.physics`, in JAX, through two Tesseracts.

Why the two gradients are added rather than fused
-------------------------------------------------
Both are gradients of a scalar with respect to **the same flat parameter
vector** — the proposal component's differentiable ``weights`` input — so they
add, exactly. Computing them in their own frameworks and summing is not an
approximation of a joint backward pass; it *is* the joint backward pass, written
in the only place both frameworks agree on, which is the parameter vector itself.
It also keeps the supervised half from having to be reimplemented in JAX, where
it would be a second copy of a loss to keep in step with the first.

The control this exists to be measured against
----------------------------------------------
:func:`finetune` takes ``use_physics_gradient``. With it off, the physics term is
still computed and still logged — so the two runs can be compared on the same
quantity — and its gradient is discarded. Everything else is identical: the same
architecture, the same starting checkpoint, the same batches in the same order,
the same number of steps, the same optimizer, the same schedule. The only
difference is whether ``dL_physics/dweights`` reaches the optimizer.

That is the control the question needs. If the physics-trained network wins, it
won against a network that saw exactly as much data for exactly as long.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from neurolayout.hybrid.losses import LossWeights, proposal_loss
from neurolayout.hybrid.model import (
    ProposalNet,
    load_flat_parameters,
)
from neurolayout.hybrid.synth import Synthesizer, SynthSpec
from neurolayout.hybrid.train import batch_to_torch

__all__ = ["FinetuneConfig", "finetune", "flat_gradient"]


@dataclass(frozen=True)
class FinetuneConfig:
    """The fine-tuning run.

    Attributes:
        steps: Optimizer steps.
        batch_size: Samples per step. Small, because every step is a real BEM
            assembly over ``batch * K`` dipoles and ``6 * batch * K`` more for the
            position sensitivity.
        learning_rate: AdamW step size. An order of magnitude below the supervised
            run's, because this starts from a trained network.
        weight_decay: AdamW decay.
        source_weight: ``lambda_source``.
        sensor_weight: ``lambda_sensor``.
        grad_clip: Global gradient-norm clip on the summed gradient.
        k_cycle: The source counts to cycle through. One ``K`` per step, because
            the component's schema fixes ``K`` for a batch — and cycling rather
            than sampling makes the control see the identical sequence.
        correlation_cycle: The correlation regimes to cycle through. Weighted to
            the hard end: this stage exists for the correlated and shared cases.
        containment_weight: Multiplier on the out-of-brain penalty inside
            ``L_physics``.
        nms_radius_m: Suppression radius for the decode.
        log_every: Logging interval.
        seed: Deterministic seed for the batches.
        loss: The supervised term's weights, carried over unchanged from
            pre-training so the fine-tune does not quietly re-weight it.
        spec: The training distribution.
    """

    steps: int = 1500
    batch_size: int = 16
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    source_weight: float = 1.0
    sensor_weight: float = 1.0
    grad_clip: float = 1.0
    k_cycle: tuple[int, ...] = (2, 4, 2, 4, 1)
    correlation_cycle: tuple[str, ...] = (
        "shared",
        "correlated",
        "shared",
        "correlated",
        "distinct",
    )
    containment_weight: float = 10.0
    nms_radius_m: float = 0.010
    log_every: int = 25
    seed: int = 424242
    loss: LossWeights = field(default_factory=LossWeights)
    spec: SynthSpec = field(default_factory=SynthSpec)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable record. This is what goes in the checkpoint."""
        return {
            "steps": self.steps,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "source_weight": self.source_weight,
            "sensor_weight": self.sensor_weight,
            "grad_clip": self.grad_clip,
            "k_cycle": list(self.k_cycle),
            "correlation_cycle": list(self.correlation_cycle),
            "containment_weight": self.containment_weight,
            "nms_radius_m": self.nms_radius_m,
            "seed": self.seed,
            "loss": self.loss.to_dict(),
            "spec": self.spec.to_dict(),
        }


def flat_gradient(model: ProposalNet) -> torch.Tensor:
    """Every parameter's gradient as one 1-D tensor, in the wire order.

    The order is ``sorted(named_parameters())``, the same order
    :func:`neurolayout.hybrid.model.flat_parameters` uses, because this vector is
    added to one that came back from a Tesseract.
    """
    return torch.cat(
        [
            (
                parameter.grad
                if parameter.grad is not None
                else torch.zeros_like(parameter)
            ).reshape(-1)
            for _, parameter in sorted(model.named_parameters())
        ]
    )


def finetune(
    model: ProposalNet,
    bank: Any,
    config: FinetuneConfig,
    *,
    headfield: Any,
    proposal: Any,
    localize_config: Any,
    containment: Any,
    checkpoint_name: str | None,
    use_physics_gradient: bool,
    indices: np.ndarray | None = None,
    variants: tuple[str, ...] | None = None,
    on_log: Any = None,
    checkpoint_path: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the composed fine-tune, or its stop-gradient control.

    Args:
        model: The pre-trained network, which is updated in place.
        bank: The gain bank.
        config: The run.
        headfield: An opened ``headfield`` Tesseract.
        proposal: An opened ``proposal`` Tesseract.
        localize_config: A :class:`~neurolayout.localize.LocalizeConfig`.
        containment: The containment ellipsoid.
        checkpoint_name: The checkpoint the ``proposal`` component should load its
            *architecture* from. Its stored weights are overridden every step by
            the vector this loop carries, so the two runs cannot diverge through
            the component's own state.
        use_physics_gradient: ``False`` runs the control — the physics term is
            computed and logged, and its gradient is discarded.
        indices: Bank positions to draw from.
        variants: Head models to draw from.
        on_log: Optional ``(record) -> None``.
        checkpoint_path: Where to write the checkpoint, rewritten at every log
            point.
        metadata: Extra provenance for the checkpoint.

    Returns:
        A dict with the history and the timing.
    """
    import jax
    import jax.numpy as jnp
    import optax

    from neurolayout.hybrid.model import save_checkpoint
    from neurolayout.hybrid.physics import make_physics_loss, physics_terms

    synthesizer = Synthesizer(bank, config.spec, indices=indices, variants=variants)
    rng = np.random.default_rng(config.seed)
    model.double().train()
    centres = model.voxel_centres.double()

    weights = torch.cat(
        [
            parameter.detach().reshape(-1)
            for _, parameter in sorted(model.named_parameters())
        ]
    ).numpy()
    optimizer = optax.adamw(config.learning_rate, weight_decay=config.weight_decay)
    state = optimizer.init(jnp.asarray(weights))

    # One closure per K: the component's schema fixes the source count for a
    # batch, and rebuilding the closure per step would rebuild the trace.
    physics = {
        count: jax.value_and_grad(
            make_physics_loss(
                headfield,
                proposal,
                localize_config,
                containment,
                n_sources=count,
                nms_radius_m=config.nms_radius_m,
                checkpoint=checkpoint_name,
                containment_weight=config.containment_weight,
            )
        )
        for count in sorted(set(config.k_cycle))
    }

    history: list[dict[str, Any]] = []
    running: dict[str, float] = {}
    accumulated = 0
    started = time.perf_counter()

    for step in range(config.steps):
        count = config.k_cycle[step % len(config.k_cycle)]
        mode = config.correlation_cycle[step % len(config.correlation_cycle)]
        raw = synthesizer.batch(
            rng,
            config.batch_size,
            n_sources=count,
            correlation_mode="distinct" if count == 1 else mode,
        )
        batch = batch_to_torch(raw, "cpu")

        load_flat_parameters(model, torch.as_tensor(weights, dtype=torch.float64))
        outputs = model(batch["eeg"], batch["channel_mask"])
        source_loss, parts = proposal_loss(
            outputs, batch, centres, config.loss, pitch_m=model.config.grid_pitch_m
        )
        model.zero_grad(set_to_none=True)
        source_loss.backward()
        source_gradient = flat_gradient(model).numpy()

        physics_value, physics_gradient = physics[count](
            jnp.asarray(weights),
            jnp.asarray(np.asarray(raw["eeg"])),
            jnp.asarray(np.asarray(raw["channel_mask"], dtype=np.float64)),
        )
        physics_gradient = np.asarray(physics_gradient)

        total = config.source_weight * source_gradient
        if use_physics_gradient:
            total = total + config.sensor_weight * np.asarray(physics_gradient)
        norm = float(np.linalg.norm(total))
        if norm > config.grad_clip:
            total = total * (config.grad_clip / norm)

        updates, state = optimizer.update(
            jnp.asarray(total), state, jnp.asarray(weights)
        )
        # A copy, not a view of the JAX buffer: `torch.as_tensor` on a
        # read-only array warns, and a parameter vector really should be
        # writable by the thing that owns it.
        weights = np.array(
            optax.apply_updates(jnp.asarray(weights), updates), dtype=np.float64
        )

        for key, value in (
            ("source_loss", float(source_loss.detach())),
            ("physics_loss", float(physics_value)),
            ("source_grad_norm", float(np.linalg.norm(source_gradient))),
            ("physics_grad_norm", float(np.linalg.norm(physics_gradient))),
            ("total_grad_norm", norm),
            ("offset_mm", parts["offset_mm"]),
        ):
            running[key] = running.get(key, 0.0) + value
        accumulated += 1

        if (step + 1) % config.log_every == 0 or step + 1 == config.steps:
            # The number of steps actually accumulated, not the interval: the
            # last log point of a short run covers a partial interval, and
            # dividing it by the interval would report a sum as a mean.
            divisor = max(accumulated, 1)
            terms = physics_terms(
                headfield,
                proposal,
                jnp.asarray(weights),
                jnp.asarray(np.asarray(raw["eeg"])),
                jnp.asarray(np.asarray(raw["channel_mask"], dtype=np.float64)),
                localize_config,
                containment,
                n_sources=count,
                nms_radius_m=config.nms_radius_m,
                checkpoint=checkpoint_name,
            )
            record = {
                "step": step + 1,
                "seconds": time.perf_counter() - started,
                "k": count,
                "correlation": mode,
                "sensor_residual": float(np.mean(terms["data"])),
                "outside_fraction": terms["outside_fraction"],
                **{key: value / divisor for key, value in running.items()},
            }
            running = {}
            accumulated = 0
            history.append(record)
            if on_log is not None:
                on_log(record)
            if checkpoint_path is not None:
                load_flat_parameters(
                    model, torch.as_tensor(weights, dtype=torch.float64)
                )
                save_checkpoint(
                    checkpoint_path,
                    model,
                    {
                        **(metadata or {}),
                        "finetune": config.to_dict(),
                        "use_physics_gradient": use_physics_gradient,
                        "step": step + 1,
                    },
                )

    load_flat_parameters(model, torch.as_tensor(weights, dtype=torch.float64))
    return {
        "history": history,
        "seconds": time.perf_counter() - started,
        "use_physics_gradient": use_physics_gradient,
        "steps": config.steps,
        "n_parameters": model.n_parameters(),
    }
