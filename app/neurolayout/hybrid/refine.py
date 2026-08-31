# Copyright 2026 NeuroLocate contributors
# SPDX-License-Identifier: Apache-2.0
r"""Continuous refinement of a source set through OpenMEEG, in JAX.

Given ``K`` starting positions and moments — from the proposal network, or from
the uninformed draw the gradient-only estimator has always used — this runs Optax
on the projector residual of :mod:`neurolayout.hybrid.physics`, whose gradient
with respect to position is central differences through OpenMEEG's C++ source
assembly.

This is the *only* difference between two of the benchmark's methods. "Gradient
only" is this refinement from an uninformed start; "proposal plus refinement" is
this refinement from the network's proposal. Same objective, same optimizer, same
number of steps, same regularizers, same physics — so the comparison measures the
initialization and nothing else. Getting that right is worth more than any of the
tuning that was deliberately not done.

Why not the frozen benchmark's own loop
---------------------------------------
:func:`neurolayout.localize.run_optimization` optimizes position, moment and — in
the ``free`` temporal model — ``K x T`` waveform numbers. This optimizes position
and moment only, because the waveforms enter linearly and are profiled out in
closed form. That is the same estimator, not a different one: at any fixed
``(p, m)`` the profiled residual *is* the minimum over the waveforms, so the two
objectives have the same minimizers.

That equivalence was measured, not assumed, before this estimator was used in
place of the full one: 0.26 mm median disagreement, and a data fit at least as
good every time.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from neurolayout.hybrid.physics import (
    best_moments,
    columns,
    containment_penalty,
    projector_residual,
    source_gains,
)

__all__ = [
    "RefineConfig",
    "warm_start_moments",
    "DEFAULT_REFINE",
    "refine",
    "separation_penalty",
]


@dataclass(frozen=True)
class RefineConfig:
    """The refinement loop.

    Attributes:
        steps: Optimizer iterations.
        position_lr: Adam step on the position, in centimetres, so 0.05 is half a
            millimetre per step at full gradient.
        moment_lr: Adam step on the moment direction. The moment enters the
            projector only through the column's *direction*, so its scale is
            gauge and only its direction has to move.
        containment_weight: Multiplier on the out-of-brain penalty.
        separation_weight: Multiplier on the minimum-separation penalty.
        min_separation_mm: Distance below which the separation penalty engages.
            The same 15 mm the frozen benchmark uses, and for the same reason: it
            forbids a collapse, and is exactly zero at every configuration the
            benchmark reports.
        record_every: Trajectory sampling interval.
    """

    steps: int = 300
    position_lr: float = 0.05
    moment_lr: float = 0.05
    containment_weight: float = 10.0
    separation_weight: float = 1.0
    min_separation_mm: float = 15.0
    record_every: int = 10

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable record."""
        return {
            "steps": self.steps,
            "position_lr": self.position_lr,
            "moment_lr": self.moment_lr,
            "containment_weight": self.containment_weight,
            "separation_weight": self.separation_weight,
            "min_separation_mm": self.min_separation_mm,
            "record_every": self.record_every,
        }


#: The default refinement loop, as a singleton so it can be a default argument.
DEFAULT_REFINE = RefineConfig()


def separation_penalty(position_cm: jnp.ndarray, min_distance_cm: float) -> jnp.ndarray:
    """``sum_{i<j} max(0, d_min - d_ij)^2``, per batch entry; zero when separated."""
    positions = jnp.atleast_3d(position_cm)
    n_sources = positions.shape[1]
    if n_sources < 2:
        return jnp.zeros(positions.shape[0])
    delta = positions[:, :, None, :] - positions[:, None, :, :]
    distance = jnp.sqrt(jnp.sum(delta**2, axis=-1) + 1e-12)
    violation = jnp.maximum(min_distance_cm - distance, 0.0) ** 2
    upper = jnp.triu(jnp.ones((n_sources, n_sources)), k=1)
    return jnp.sum(violation * upper[None], axis=(1, 2))


def warm_start_moments(
    headfield: Any, observed: np.ndarray, positions_m: np.ndarray, config: Any
) -> np.ndarray:
    """``[B, K, 3]`` closed-form starting orientations at the given positions.

    One batched solver call for the gains, then a 3x3 generalized eigenproblem per
    source. See :func:`neurolayout.hybrid.physics.best_moments` for why this is
    here: a refinement started from a random orientation is a weaker baseline than
    the frozen benchmark's own, and every method in the matrix gets this so it
    changes no comparison.
    """
    positions = jnp.asarray(np.asarray(positions_m, dtype=np.float64))
    target = jnp.asarray(np.asarray(observed, dtype=np.float64))
    gains = source_gains(headfield, positions, config)
    return np.asarray(best_moments(gains, target))


def refine(
    headfield: Any,
    observed: np.ndarray,
    positions_m: np.ndarray,
    moments: np.ndarray | None,
    config: Any,
    containment: Any,
    settings: RefineConfig = DEFAULT_REFINE,
    *,
    truth_m: np.ndarray | None = None,
) -> dict[str, Any]:
    """Refine a batch of source sets against their own observations.

    Args:
        headfield: An opened ``headfield`` Tesseract.
        observed: ``[B, C, T]`` sensor signals, volts.
        positions_m: ``[B, K, 3]`` starting positions, metres.
        moments: ``[B, K, 3]`` starting dipole directions (any scale), or ``None``
            to warm-start them in closed form with :func:`warm_start_moments`.
        config: A :class:`~neurolayout.localize.LocalizeConfig` — supplies the
            backend, the head model and the channel subset.
        containment: The containment ellipsoid.
        settings: The refinement loop.
        truth_m: ``[B, K, 3]`` true positions, for the recorded error curve only.
            The optimizer never sees it.

    Returns:
        Dict with the refined positions and moments, the loss and error curves,
        the trajectory, and the timing.
    """
    from neurolayout.matching import match_sources

    target = jnp.asarray(np.asarray(observed, dtype=np.float64))
    start_position_cm = jnp.asarray(np.asarray(positions_m, dtype=np.float64) * 1e2)
    if moments is None:
        moments = warm_start_moments(headfield, observed, positions_m, config)
    start_moment = jnp.asarray(np.asarray(moments, dtype=np.float64))
    start_moment = start_moment / jnp.maximum(
        jnp.linalg.norm(start_moment, axis=-1, keepdims=True), 1e-30
    )
    min_separation_cm = settings.min_separation_mm * 1e-1

    def parts(params: dict[str, jnp.ndarray]) -> dict[str, jnp.ndarray]:
        positions = params["position_cm"] * 1e-2
        column_matrix = columns(headfield, positions, params["moment"] * 1e-8, config)
        return {
            "data": projector_residual(column_matrix, target),
            "containment": containment_penalty(positions, containment),
            "separation": separation_penalty(params["position_cm"], min_separation_cm),
        }

    def loss(params: dict[str, jnp.ndarray]) -> jnp.ndarray:
        terms = parts(params)
        return jnp.sum(
            terms["data"]
            + settings.containment_weight * terms["containment"]
            + settings.separation_weight * terms["separation"]
        )

    optimizer = optax.multi_transform(
        {
            "position_cm": optax.adam(settings.position_lr),
            "moment": optax.adam(settings.moment_lr),
        },
        param_labels={"position_cm": "position_cm", "moment": "moment"},
    )
    params = {"position_cm": start_position_cm, "moment": start_moment}
    state = optimizer.init(params)
    value_and_grad = jax.value_and_grad(loss)

    history: dict[str, list] = {
        "step": [], "loss": [], "data": [], "error_mm": [], "position_cm": []
    }

    def record(step: int, params: dict[str, jnp.ndarray]) -> None:
        terms = parts(params)
        history["step"].append(step)
        history["loss"].append(float(loss(params)))
        history["data"].append([float(x) for x in terms["data"]])
        # `[B, K, 3]` per record. Nothing in the benchmark reads this — the shards
        # store answers, not paths — but the figures need the path itself, and
        # recording it changes no computation.
        history["position_cm"].append(np.asarray(params["position_cm"]).tolist())
        history["error_mm"].append(
            None if truth_m is None else _errors(params, truth_m, match_sources)
        )

    started = time.perf_counter()
    record(0, params)
    for step in range(1, settings.steps + 1):
        _, grads = value_and_grad(params)
        updates, state = optimizer.update(grads, state, params)
        params = optax.apply_updates(params, updates)
        if step % settings.record_every == 0 or step == settings.steps:
            record(step, params)
    seconds = time.perf_counter() - started

    final_parts = parts(params)
    return {
        "converged": _converged(history),
        "positions_m": np.asarray(params["position_cm"]) * 1e-2,
        "moments": np.asarray(params["moment"]),
        "data_loss": [float(x) for x in final_parts["data"]],
        "containment": [float(x) for x in final_parts["containment"]],
        "separation": [float(x) for x in final_parts["separation"]],
        "history": history,
        "seconds": seconds,
        "steps": settings.steps,
        "settings": settings.to_dict(),
    }


def _converged(history: dict[str, list], tail: float = 0.2, tolerance: float = 0.01):
    """Per-entry flag: did the data term stop moving over the last of the run?

    Convergence and accuracy are different questions and are reported separately,
    because at ``K = 4`` on shared dynamics many runs converge cleanly onto the
    *wrong* sources. A run that has not converged is a statement about the budget;
    a run that converged to the wrong answer is a statement about the objective,
    and conflating them would blame the optimizer for the physics.

    Args:
        history: The recorded trajectory.
        tail: Fraction of the run to judge over.
        tolerance: Largest relative change in the data term that still counts as
            converged.

    Returns:
        One bool per batch entry.
    """
    curve = np.asarray(history["data"], dtype=np.float64)  # [n_records, B]
    if curve.shape[0] < 3:
        return [False] * curve.shape[1]
    start = max(1, int(curve.shape[0] * (1.0 - tail)))
    early, late = curve[start - 1], curve[-1]
    change = np.abs(late - early) / np.maximum(np.abs(early), 1e-30)
    return [bool(value) for value in change <= tolerance]


def _errors(
    params: dict[str, jnp.ndarray], truth_m: np.ndarray, match: Any
) -> list[list[float]]:
    """Per-entry, per-source error under the assignment, for the recorded curve."""
    positions = np.asarray(params["position_cm"]) * 1e-2
    return [
        [float(value) for value in match(positions[index], truth_m[index]).errors_mm]
        for index in range(len(positions))
    ]
