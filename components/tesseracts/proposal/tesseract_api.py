# Copyright 2026 NeuroLocate contributors
# SPDX-License-Identifier: Apache-2.0
r"""``proposal`` — global source-set proposal Tesseract (PyTorch).

Maps an epoch of scalp EEG to a sparse set of continuous source positions::

    eeg          [B, C, T]        volts, average-referenced, frozen channel order
    channel_mask [B, C]           1 for a retained channel
    weights      [P]              the network's parameters
        ->
    positions_m  [B, K, 3]        metres, MNE head frame
    moments      [B, K, 3]        unit dipole directions

and exposes the ``torch.autograd`` derivative of that map with respect to **both**
``eeg`` and ``weights``.

Why the weights are an input
----------------------------
A Tesseract differentiates with respect to its inputs. Putting the network's
parameters *inside* the component would make them state, and ``jax.grad`` of a
loss defined downstream could then reach the proposal's coordinates but never its
weights. Putting them in the schema makes ``dL/dweights`` an ordinary cotangent,
so the same single ``jax.grad`` that runs

    loss  ->  headfield VJP (central differences through OpenMEEG's C++ BEM)
          ->  source positions and moments
          ->  proposal VJP (torch.autograd)
          ->  network parameters

is what *trains* the network. The physics solver is not attached after training
for display; it is inside the training gradient.

A component whose network is frozen at inference would make the opposite choice
and keep its weights inside, where the VJP can never touch them. This network is
not frozen, so they are inputs.

What carries a gradient, and what does not
------------------------------------------
The network predicts a logit and a continuous offset per voxel of a coarse
lattice. Which voxels are selected is a hard ``argmax`` with non-maximum
suppression: discrete, and detached. The returned coordinates are the selected
voxels' centres plus their own continuous offsets, and *those* carry gradient.
Stated plainly because it is the honest description of a heatmap read-out — the
choice of basin is discrete, the position within it is not.

The component carries PyTorch and nothing else. It has never heard of OpenMEEG,
JAX, or MNE.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float64

# Loaded by path in both transports, so the component's own directory is not
# importable by default. The module name is deliberately component-unique: every
# opened component's directory lands on the same `sys.path`, so a shared
# top-level name means whichever loads second gets the other's module.
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from proposal_model import (  # noqa: E402
    ProposalNet,
    decode,
    load_checkpoint,
)

#: Environment variable naming a directory of proposal checkpoints.
#:
#: The packaged checkpoint ships inside the image. This is how an alternative one
#: — a stop-gradient control, an ablation, a retrain — is selected without
#: rebuilding: the name travels over the boundary as an input field, the bytes do
#: not.
CHECKPOINT_DIR_ENV = "NEUROLOCATE_PROPOSAL_DIR"

#: Name of the checkpoint packaged with the image.
PACKAGED_CHECKPOINT = "proposal.pt"


#
# Schemas
#


class InputSchema(BaseModel):
    """An epoch, a channel mask, and the parameters to run it through."""

    eeg: Differentiable[Array[(None, None, None), Float64]] = Field(
        description=(
            "Sensor signals [batch, n_channels, n_times], volts, average- "
            "referenced, in the frozen 64-channel order. Only the epoch's spatial "
            "covariance is read, so the absolute scale and the sign of any "
            "source's time course make no difference to the output."
        )
    )
    channel_mask: Array[(None, None), Float64] | None = Field(
        default=None,
        description=(
            "1.0 for a retained channel, 0.0 for a dropped one, [batch, "
            "n_channels]. null (the default) keeps every channel."
        ),
    )
    weights: Differentiable[Array[(None,), Float64]] | None = Field(
        default=None,
        description=(
            "Flat network parameters [n_parameters], in the checkpoint's own "
            "order. null uses the checkpoint's trained values. Supplying them is "
            "what makes dL/dweights reachable from a loss defined downstream of "
            "the BEM solver."
        ),
    )
    checkpoint: str | None = Field(
        default=None,
        description=(
            "Name of a checkpoint to use, resolved inside the component against "
            f"{CHECKPOINT_DIR_ENV}. null is the packaged one."
        ),
    )
    n_sources: int = Field(
        default=0,
        ge=0,
        description=(
            "How many sources to return. 0 asks the count head, which is what an "
            "estimator with no prior on K would do; the benchmark passes K, "
            "because every method it compares against is given K."
        ),
    )
    nms_radius_m: float = Field(
        default=0.010,
        gt=0.0,
        description=(
            "Non-maximum suppression radius in metres. A little over one lattice "
            "pitch, so one broad blob is not read as several sources."
        ),
    )


class OutputSchema(BaseModel):
    """The proposed source set."""

    positions_m: Differentiable[Array[(None, None, 3), Float64]] = Field(
        description="Proposed source positions [batch, K, 3], metres, head frame."
    )
    moments: Differentiable[Array[(None, None, 3), Float64]] = Field(
        description="Proposed unit dipole directions [batch, K, 3]."
    )
    scores: Differentiable[Array[(None, None), Float64]] = Field(
        description="Heatmap logit at each selected voxel [batch, K]."
    )
    count_logits: Differentiable[Array[(None, None), Float64]] = Field(
        description="Logits over K = 1..max_sources [batch, max_sources]."
    )


#
# Model plumbing
#


def _resolve(name: str | None) -> Path:
    """Locate a checkpoint by name, or the packaged one."""
    if name is None:
        return Path(__file__).parent / PACKAGED_CHECKPOINT
    if "/" in name or "\\" in name or ".." in name:
        raise ValueError(f"checkpoint name {name!r} must be a bare name, not a path")
    directory = os.environ.get(CHECKPOINT_DIR_ENV)
    if not directory:
        raise FileNotFoundError(
            f"checkpoint {name!r} was requested but {CHECKPOINT_DIR_ENV} is not set"
        )
    path = Path(directory).expanduser() / (name if name.endswith(".pt") else f"{name}.pt")
    if not path.exists():
        raise FileNotFoundError(f"no proposal checkpoint {path}")
    return path


@lru_cache(maxsize=4)
def _model(name: str | None) -> ProposalNet:
    """The requested checkpoint's network, built once per process."""
    path = _resolve(name)
    if not path.exists():
        raise FileNotFoundError(
            f"no proposal checkpoint at {path}. Train one with "
            "`python scripts/train_proposal.py`."
        )
    model, _ = load_checkpoint(path)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.double()


def _to_tensor(array: Any) -> torch.Tensor:
    """A writable float64 tensor from whatever the runtime handed over."""
    data = np.asarray(array, dtype=np.float64)
    if not data.flags.writeable:
        data = data.copy()
    return torch.as_tensor(data, dtype=torch.float64)


def _mask_for(inputs: InputSchema, eeg: torch.Tensor) -> torch.Tensor:
    if inputs.channel_mask is None:
        return torch.ones(eeg.shape[:2], dtype=torch.float64)
    return _to_tensor(inputs.channel_mask)


def _default_weights(model: ProposalNet) -> torch.Tensor:
    return torch.cat(
        [
            value.detach().reshape(-1)
            for _, value in sorted(model.named_parameters())
        ]
    )


def _run(
    inputs: InputSchema, eeg: torch.Tensor, weights: torch.Tensor
) -> dict[str, torch.Tensor]:
    """The whole map, as a pure function of ``(eeg, weights)``.

    ``torch.func.functional_call`` is what makes it pure: the parameters are
    supplied per call rather than read off the module, so the same module object
    can serve the primal and every VJP without carrying state between them.
    """
    model = _model(inputs.checkpoint)
    names = [name for name, _ in sorted(model.named_parameters())]
    shapes = [dict(model.named_parameters())[name].shape for name in names]
    parameters, offset = {}, 0
    for name, shape in zip(names, shapes, strict=True):
        size = int(np.prod(shape)) if len(shape) else 1
        parameters[name] = weights[offset : offset + size].reshape(shape)
        offset += size
    outputs = torch.func.functional_call(
        model, parameters, (eeg, _mask_for(inputs, eeg))
    )
    return decode(
        outputs,
        None if inputs.n_sources == 0 else inputs.n_sources,
        nms_radius_m=inputs.nms_radius_m,
        max_sources=model.config.max_sources,
    ) | {"count_logits": outputs["count_logits"]}


#
# Required endpoints
#


def apply(inputs: InputSchema) -> OutputSchema:
    """Propose a source set."""
    eeg = _to_tensor(inputs.eeg)
    model = _model(inputs.checkpoint)
    weights = (
        _default_weights(model) if inputs.weights is None else _to_tensor(inputs.weights)
    )
    with torch.no_grad():
        result = _run(inputs, eeg, weights)
    return OutputSchema(
        positions_m=result["positions_m"].numpy(),
        moments=result["moments"].numpy(),
        scores=result["scores"].numpy(),
        count_logits=result["count_logits"].numpy(),
    )


#
# Optional endpoints
#


def abstract_eval(abstract_inputs: Any) -> dict[str, Any]:
    """Output shapes/dtypes from input shapes/dtypes, without evaluating."""
    inputs = abstract_inputs.model_dump()
    batch, n_channels, _ = inputs["eeg"]["shape"]
    model = _model(inputs.get("checkpoint"))
    if n_channels != model.config.n_channels:
        raise ValueError(
            f"eeg has {n_channels} channels but the checkpoint expects "
            f"{model.config.n_channels}"
        )
    requested = int(inputs.get("n_sources") or 0)
    # With n_sources = 0 the count head decides, and that is data-dependent, so
    # the only shape that can be promised without evaluating is the maximum.
    n_sources = requested or model.config.max_sources
    return {
        "positions_m": {"shape": (batch, n_sources, 3), "dtype": "float64"},
        "moments": {"shape": (batch, n_sources, 3), "dtype": "float64"},
        "scores": {"shape": (batch, n_sources), "dtype": "float64"},
        "count_logits": {
            "shape": (batch, model.config.max_sources),
            "dtype": "float64",
        },
    }


_VJP_INPUTS = {"eeg", "weights"}
_VJP_OUTPUTS = {"positions_m", "moments", "scores", "count_logits"}


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
) -> dict[str, Any]:
    """Reverse-mode product ``w^T J`` via ``torch.func.vjp``.

    Differentiates with respect to the epoch and the parameters together, so a
    single call serves both the sensitivity analysis and the training step.
    """
    unknown_in = set(vjp_inputs) - _VJP_INPUTS
    unknown_out = set(vjp_outputs) - _VJP_OUTPUTS
    if unknown_in or unknown_out:
        raise ValueError(
            f"unsupported vjp request: inputs={sorted(unknown_in)}, "
            f"outputs={sorted(unknown_out)}"
        )
    if not vjp_inputs:
        raise ValueError("vjp_inputs is empty")

    model = _model(inputs.checkpoint)  # warm the cache outside the transform
    eeg = _to_tensor(inputs.eeg)
    weights = (
        _default_weights(model) if inputs.weights is None else _to_tensor(inputs.weights)
    )
    ordered = sorted(_VJP_OUTPUTS)

    def evaluate(
        epoch: torch.Tensor, parameters: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        result = _run(inputs, epoch, parameters)
        return tuple(result[name] for name in ordered)

    primal, vjp_fn = torch.func.vjp(evaluate, eeg, weights)
    cotangents = tuple(
        _to_tensor(cotangent_vector[name])
        if name in vjp_outputs
        else torch.zeros_like(value)
        for name, value in zip(ordered, primal, strict=True)
    )
    grad_eeg, grad_weights = vjp_fn(cotangents)
    return {
        name: (grad_eeg if name == "eeg" else grad_weights).numpy()
        for name in vjp_inputs
    }
