# Copyright 2026 NeuroLocate contributors
# SPDX-License-Identifier: Apache-2.0
"""``headfield`` — differentiable EEG volume-conduction Tesseract.

The component owns all of NeuroLocate's head physics and contains **no autodiff
framework of any kind**. Its derivative contract is written by hand, which is
precisely the situation Tesseract exists for: OpenMEEG is a compiled C++ solver
that cannot be differentiated, and the optimizer upstream is JAX.

Three modes share one component, selected by :attr:`InputSchema.mode`.

``mode="localize"`` — the NeuroLocate forward
---------------------------------------------
Source parameters in, sensor signals out::

    source_positions   [K, 3]        metres, MNE head frame
    source_timecourses [B, K, 3, T]  vector moment time courses, A m

    eeg[b, c, t] = sum_{k,j} G(p)[c, k, j] * source_timecourses[b, k, j, t]

``G`` is the average-referenced free-orientation gain, from either backend:

``backend="openmeeg"``
    OpenMEEG's symmetric BEM on a cached fsaverage template head. Per call it
    assembles a genuine ``DipSourceMat`` at the requested continuous position —
    no nearest-vertex snapping, no interpolation.

``backend="sphere"``
    The analytic homogeneous-sphere Legendre series. Self-contained (no cached
    artifact, no OpenMEEG) and kept permanently as the fallback solver and as an
    independent check on the BEM.

Derivatives: exact and analytic with respect to ``source_timecourses`` (the map
is linear in them), and central finite differences through the real solver with
respect to ``source_positions`` — three numbers per source, so six extra gain
evaluations. See :mod:`neurolayout_shared.source_model`.

``mode="localize_batch"`` — the same forward, for many source sets at once
---------------------------------------------------------------------------
Identical physics and identical derivative rules, with each batch entry carrying
its own ``K`` positions::

    source_positions_batch [B, K, 3]
    source_timecourses     [B, K, 3, T]

    eeg[b, c, t] = sum_{k,j} G(p[b])[c, k, j] * source_timecourses[b, k, j, t]

This exists because the BEM assembly cost is per *dipole*, not per call: ``B``
separate ``localize`` calls are ``B`` source-term assemblies, while this is one
over ``B*K`` dipoles — and one over ``6*B*K`` for the position sensitivity. It is
what a training step over a batch of proposals needs, and it changes no
mathematics: ``tests/test_batched_headfield.py`` checks it against both the
per-entry loop and the flattened single-set construction, in value and gradient.

``mode="montage"`` — the legacy sensor-placement forward
--------------------------------------------------------
The original NeuroLayout objective, kept green and unchanged: continuous
electrode design variables sample a cached dense sphere lead field through a
smooth softmax interpolant, with a hand-written analytic reverse-mode rule
(:mod:`neurolayout_shared.sampling`). This is the default mode so every frozen
test case and every existing gradient gate keeps passing bit-for-bit.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from neurolayout_shared.headmodel import HeadModelSpec, get_head_model
from neurolayout_shared.sampling import backward as sampling_backward
from neurolayout_shared.sampling import forward as sampling_forward
from neurolayout_shared.source_model import DEFAULT_FD_STEP, average_reference_operator
from neurolayout_shared.source_model import backward as source_backward
from neurolayout_shared.source_model import backward_batched as source_backward_batched
from neurolayout_shared.source_model import forward as source_forward
from neurolayout_shared.source_model import forward_batched as source_forward_batched
from neurolayout_shared.sphere_model import SphereHead, sphere_source_gain
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float64, Int64

#
# Schemas
#

#: Placeholder arrays for the fields the active mode does not read. Every field
#: is always present and always valid, so the schema stays a plain product type
#: rather than a union the runtime would have to discriminate.
_UNUSED_ELECTRODES = np.array([[0.0, 0.0, 1.0]])
_UNUSED_ACTIVITY = np.zeros((1, 1, 1))
_UNUSED_POSITIONS = np.zeros((1, 3))
_UNUSED_TIMECOURSES = np.zeros((1, 1, 3, 1))
_UNUSED_POSITIONS_BATCH = np.zeros((1, 1, 3))


class InputSchema(BaseModel):
    """Source parameters (``localize``) or electrode design variables (``montage``)."""

    mode: Literal["montage", "localize", "localize_batch"] = Field(
        default="montage",
        description=(
            "'localize': map source parameters to sensor signals (NeuroLocate). "
            "'localize_batch': the same map for a batch of independent source "
            "sets, each with its own K positions, in one solver call — what a "
            "training step over many proposals needs. "
            "'montage': map continuous electrode positions to virtual sensor "
            "signals (the original NeuroLayout objective, kept for regression)."
        ),
    )
    backend: Literal["sphere", "openmeeg"] = Field(
        default="sphere",
        description=(
            "Volume-conduction solver. 'openmeeg' is a cached BEM head model and "
            "requires the cached head-model artifact; 'sphere' is the "
            "self-contained analytic fallback. 'montage' mode is sphere-only."
        ),
    )
    head_model: str | None = Field(
        default=None,
        description=(
            "Name of a head-model artifact to use with backend='openmeeg', "
            "resolved inside the component against NEUROLOCATE_HEADMODEL_DIR. "
            "null (the default) is the packaged fsaverage template. This is how "
            "a subject-specific model is selected without shipping anyone's "
            "anatomy across the boundary: the name travels, the bytes do not."
        ),
    )

    # --- mode="localize" ----------------------------------------------------
    source_positions: Differentiable[Array[(None, 3), Float64]] = Field(
        default_factory=lambda: _UNUSED_POSITIONS.copy(),
        description=(
            "Source positions [K, 3] in metres, MNE head coordinates. Must lie "
            "inside the inner-skull surface for the OpenMEEG backend."
        ),
    )
    source_positions_batch: Differentiable[Array[(None, None, 3), Float64]] = Field(
        default_factory=lambda: _UNUSED_POSITIONS_BATCH.copy(),
        description=(
            "Source positions [batch, K, 3] in metres, MNE head coordinates, for "
            "mode='localize_batch'. Each batch entry is an independent source set; "
            "all batch*K dipoles are assembled in one OpenMEEG call, and so are "
            "the 6*batch*K perturbed dipoles the position sensitivity needs."
        ),
    )
    source_timecourses: Differentiable[Array[(None, None, 3, None), Float64]] = Field(
        default_factory=lambda: _UNUSED_TIMECOURSES.copy(),
        description="Vector moment time courses [batch, K, 3, n_times], in A m.",
    )
    fd_step: float = Field(
        default=DEFAULT_FD_STEP,
        gt=0.0,
        description=(
            "Central-difference step in metres for the source-position "
            "sensitivity. The default sits in the middle of a four-decade "
            "plateau; see docs/OPENMEEG_HEADMODEL.md."
        ),
    )
    fd_order: Literal[2, 4] = Field(
        default=2,
        description="Finite-difference stencil order for the position sensitivity.",
    )
    sphere_radius: float = Field(
        default=0.09, gt=0.0, description="Sphere radius (m), sphere backend only."
    )
    sphere_sigma: float = Field(
        default=0.33, gt=0.0, description="Conductivity (S/m), sphere backend only."
    )
    n_channels: int = Field(
        default=64, ge=2, description="Sensor count for the sphere backend."
    )
    channel_subset: Array[(None,), Int64] | None = Field(
        default=None,
        description=(
            "Indices of the channels to keep, into the backend's canonical order. "
            "``null`` (the default) keeps all of them. Subsetting happens *before* "
            "average referencing, so a 16-channel run carries the reference a "
            "16-channel recording would have, not a restriction of the 64-channel "
            "one."
        ),
    )

    # --- mode="montage" -----------------------------------------------------
    electrode_vectors: Differentiable[Array[(None, 3), Float64]] = Field(
        default_factory=lambda: _UNUSED_ELECTRODES.copy(),
        description=(
            "Unconstrained electrode design variables [K, 3]. Only the direction "
            "matters; the magnitude is projected out, so the optimizer never has "
            "to respect a norm constraint."
        ),
    )
    source_activity: Differentiable[Array[(None, None, None), Float64]] = Field(
        default_factory=lambda: _UNUSED_ACTIVITY.copy(),
        description="Cortical source amplitudes [batch, n_sources, n_times].",
    )
    kappa: float = Field(
        default=60.0,
        gt=0.0,
        description=(
            "Locality of the smooth scalp sampler. Larger values approach "
            "nearest-vertex lookup; smaller values blur across the scalp."
        ),
    )
    n_scalp: int = Field(
        default=HeadModelSpec().n_scalp,
        ge=4,
        description="Scalp lattice size J of the cached dense lead field.",
    )
    n_sources: int = Field(
        default=HeadModelSpec().n_sources,
        ge=1,
        description="Source-basis size S. Must match source_activity.shape[1].",
    )


class OutputSchema(BaseModel):
    """Sensor signals and the positions they were evaluated at."""

    eeg: Differentiable[Array[(None, None, None), Float64]] = Field(
        description="Sensor signals [batch, n_channels, n_times], in volts."
    )
    electrode_xyz: Differentiable[Array[(None, 3), Float64]] = Field(
        description=(
            "Positions the signals were evaluated at [n_channels, 3], metres. "
            "In 'montage' mode these are the realized electrode positions and "
            "depend on the design variables; in 'localize' mode they are the "
            "fixed canonical sensor array."
        )
    )


#
# Backends
#


def _sphere_head(inputs: InputSchema) -> SphereHead:
    return SphereHead(
        radius=inputs.sphere_radius,
        sigma=inputs.sphere_sigma,
        n_channels=inputs.n_channels,
    )


def _channel_subset(inputs: InputSchema) -> np.ndarray | None:
    """Validated channel indices to keep, or ``None`` for the whole array.

    ``None`` rather than an empty array is deliberate. A zero-length array is a
    legal NumPy value that the in-process transport handles, but it fails
    validation on the way into a *served* container — the base64 payload of a
    ``shape: [0]`` numeric array does not parse as numeric — so "keep everything"
    encoded that way worked locally and 422'd over HTTP. Optional is also the
    honest type: there is no such thing as a zero-channel montage.
    """
    if inputs.channel_subset is None:
        return None
    subset = np.asarray(inputs.channel_subset, dtype=np.int64).ravel()
    if subset.size == 0:
        raise ValueError("channel_subset is empty; pass null to keep all channels")
    available = _n_available_channels(inputs)
    if subset.min() < 0 or subset.max() >= available:
        raise ValueError(
            f"channel_subset indexes outside the {available}-channel array: "
            f"[{subset.min()}, {subset.max()}]"
        )
    if len(np.unique(subset)) != len(subset):
        raise ValueError("channel_subset contains duplicate channels")
    return subset


def _n_available_channels(inputs: InputSchema) -> int:
    if inputs.backend == "openmeeg":
        return _openmeeg_forward(inputs).geometry.n_channels
    return inputs.n_channels


def _openmeeg_forward(inputs: InputSchema, *, reference: bool = True):
    """The requested OpenMEEG head model, memoized per (name, reference)."""
    from neurolayout_shared.openmeeg_model import load_forward

    return load_forward(reference=reference, name=inputs.head_model)


def _gain_function(inputs: InputSchema):
    """Resolve ``positions -> gain [C, P, 3]`` for the requested backend.

    When a channel subset is requested the backend is asked for the *unreferenced*
    gain, the rows are selected, and the average reference is then formed over the
    retained channels — the reference a recording with that many electrodes would
    actually carry.
    """
    subset = _channel_subset(inputs)
    referenced = subset is None
    if inputs.backend == "openmeeg":
        base = _openmeeg_forward(inputs, reference=referenced).gain
    else:
        head = _sphere_head(inputs)

        def base(positions: np.ndarray) -> np.ndarray:
            return sphere_source_gain(head, positions, reference=referenced)

    if subset is None:
        return base

    reference = average_reference_operator(len(subset))

    def subset_gain(positions: np.ndarray) -> np.ndarray:
        return np.einsum("cd,dpj->cpj", reference, base(positions)[subset])

    return subset_gain


def _sensor_xyz(inputs: InputSchema) -> np.ndarray:
    if inputs.backend == "openmeeg":
        sensors = _openmeeg_forward(inputs).geometry.sensor_xyz
    else:
        sensors = _sphere_head(inputs).sensor_xyz()
    subset = _channel_subset(inputs)
    return sensors if subset is None else sensors[subset]


def _localize_forward(inputs: InputSchema) -> dict[str, np.ndarray]:
    cache = source_forward(
        _gain_function(inputs),
        np.asarray(inputs.source_positions, dtype=np.float64),
        np.asarray(inputs.source_timecourses, dtype=np.float64),
    )
    cache["electrode_xyz"] = _sensor_xyz(inputs)
    return cache


def _localize_batch_forward(inputs: InputSchema) -> dict[str, np.ndarray]:
    cache = source_forward_batched(
        _gain_function(inputs),
        np.asarray(inputs.source_positions_batch, dtype=np.float64),
        np.asarray(inputs.source_timecourses, dtype=np.float64),
    )
    cache["electrode_xyz"] = _sensor_xyz(inputs)
    return cache


#: The forward for each mode, so ``apply`` and the VJP cannot disagree about it.
_FORWARDS = {
    "localize": _localize_forward,
    "localize_batch": _localize_batch_forward,
}


#
# Legacy montage-design plumbing (unchanged)
#


def _head_model(inputs: InputSchema):
    """Resolve (and memoize) the dense lead field requested by ``inputs``."""
    spec = HeadModelSpec(n_scalp=inputs.n_scalp, n_sources=inputs.n_sources)
    model = get_head_model(spec)
    n_sources = np.asarray(inputs.source_activity).shape[1]
    if n_sources != model.n_sources:
        raise ValueError(
            f"source_activity has {n_sources} sources but the head model was "
            f"built with {model.n_sources}; set n_sources to match"
        )
    return model


def _montage_forward(inputs: InputSchema) -> dict[str, np.ndarray]:
    model = _head_model(inputs)
    return sampling_forward(
        np.asarray(inputs.electrode_vectors, dtype=np.float64),
        np.asarray(inputs.source_activity, dtype=np.float64),
        model.scalp_directions,
        model.lead_field,
        inputs.kappa,
        model.spec.radius,
    )


def _check_mode(inputs: InputSchema) -> None:
    if inputs.mode == "montage" and inputs.backend != "sphere":
        raise ValueError(
            "mode='montage' is only defined for backend='sphere'; the OpenMEEG "
            "backend answers the source-localization question, not the "
            "electrode-placement one"
        )


#
# Required endpoints
#


def apply(inputs: InputSchema) -> OutputSchema:
    """Predict sensor signals for the requested mode and backend."""
    _check_mode(inputs)
    forward = _FORWARDS.get(inputs.mode)
    cache = _montage_forward(inputs) if forward is None else forward(inputs)
    return OutputSchema(eeg=cache["eeg"], electrode_xyz=cache["electrode_xyz"])


#
# Optional endpoints
#


def abstract_eval(abstract_inputs: Any) -> dict[str, Any]:
    """Output shapes/dtypes from input shapes/dtypes, without evaluating."""
    inputs = abstract_inputs.model_dump()
    if inputs["mode"] in ("localize", "localize_batch"):
        batch, _, _, n_times = inputs["source_timecourses"]["shape"]
        # Every array field is abstract here, so the subset contributes only its
        # length — which is exactly the channel count it selects.
        subset = inputs["channel_subset"]
        n_subset = 0 if subset is None else int(subset["shape"][0])
        if n_subset:
            n_channels = int(n_subset)
        elif inputs["backend"] == "openmeeg":
            from neurolayout_shared.openmeeg_model import load_forward

            n_channels = load_forward(name=inputs.get("head_model")).geometry.n_channels
        else:
            n_channels = inputs["n_channels"]
    else:
        n_channels = inputs["electrode_vectors"]["shape"][0]
        batch, _, n_times = inputs["source_activity"]["shape"]
    return {
        "eeg": {"shape": (batch, n_channels, n_times), "dtype": "float64"},
        "electrode_xyz": {"shape": (n_channels, 3), "dtype": "float64"},
    }


_LOCALIZE_INPUTS = {"source_positions", "source_timecourses"}
_LOCALIZE_BATCH_INPUTS = {"source_positions_batch", "source_timecourses"}
_MONTAGE_INPUTS = {"electrode_vectors", "source_activity"}
_ALLOWED_VJP_INPUTS = {
    "localize": _LOCALIZE_INPUTS,
    "localize_batch": _LOCALIZE_BATCH_INPUTS,
    "montage": _MONTAGE_INPUTS,
}


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
) -> dict[str, Any]:
    """Reverse-mode product ``w^T J``, hand-written in both modes.

    In ``localize`` mode the time-course cotangent is exact algebra and the
    position cotangent is central differences through the active solver. In
    ``montage`` mode it is the original analytic chain through the einsum, the
    lead-field contraction, the softmax and the ``u -> u/||u||`` projection.
    """
    _check_mode(inputs)
    allowed = _ALLOWED_VJP_INPUTS[inputs.mode]
    unknown_in = set(vjp_inputs) - allowed
    unknown_out = set(vjp_outputs) - {"eeg", "electrode_xyz"}
    if unknown_in or unknown_out:
        raise ValueError(
            f"unsupported vjp request for mode={inputs.mode!r}: "
            f"inputs={sorted(unknown_in)}, outputs={sorted(unknown_out)}"
        )

    def cotangent(name: str) -> np.ndarray | None:
        if name not in vjp_outputs:
            return None
        return np.asarray(cotangent_vector[name], dtype=np.float64)

    if inputs.mode in ("localize", "localize_batch"):
        batched = inputs.mode == "localize_batch"
        position_name = "source_positions_batch" if batched else "source_positions"
        positions = np.asarray(getattr(inputs, position_name), dtype=np.float64)
        timecourses = np.asarray(inputs.source_timecourses, dtype=np.float64)
        grad_eeg = cotangent("eeg")
        if grad_eeg is None:
            # Only `electrode_xyz` was requested, and in these modes it is a
            # constant of the head model: the cotangent is exactly zero. Return
            # before the forward, which would be a wasted BEM assembly.
            zeros = {
                position_name: np.zeros_like(positions),
                "source_timecourses": np.zeros_like(timecourses),
            }
            return {name: zeros[name] for name in vjp_inputs}
        rule = source_backward_batched if batched else source_backward
        grads = rule(
            _FORWARDS[inputs.mode](inputs),
            _gain_function(inputs),
            positions,
            timecourses,
            grad_eeg,
            step=inputs.fd_step,
            order=inputs.fd_order,
            need_positions=position_name in vjp_inputs,
            need_timecourses="source_timecourses" in vjp_inputs,
        )
        return {name: grads[name] for name in vjp_inputs}

    model = _head_model(inputs)
    grads = sampling_backward(
        _montage_forward(inputs),
        np.asarray(inputs.source_activity, dtype=np.float64),
        model.scalp_directions,
        model.lead_field,
        inputs.kappa,
        model.spec.radius,
        grad_eeg=cotangent("eeg"),
        grad_electrode_xyz=cotangent("electrode_xyz"),
    )
    return {name: grads[name] for name in vjp_inputs}
