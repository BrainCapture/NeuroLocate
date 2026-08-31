# Copyright 2026 NeuroLocate contributors
# SPDX-License-Identifier: Apache-2.0
r"""The sparse source-localization problem: objective, optimizer, and scoring.

This is NeuroLocate's headline experiment, and the file that makes the claim
checkable. A trial is:

1. pick ``K`` true sources ``(p*_k, m*_k)`` in the fsaverage cortical source
   space;
2. generate 64-channel EEG — through a **different** forward model than the one
   used to invert, see :mod:`neurolayout.mismatch` — and optionally add sensor
   noise at a stated SNR;
3. **throw the true positions away**;
4. start the optimizer at deliberately wrong positions;
5. run Optax on the reconstruction loss, whose gradient goes back through
   OpenMEEG's C++ solver;
6. report how far the recovered sources ended up from the truth, in millimetres,
   under the optimal assignment of estimates to true sources.

Nothing in this module imports OpenMEEG, MNE, or PyTorch. It holds a JAX
function whose evaluation happens to cross a process boundary into a compiled
BEM solver, and ``jax.grad`` sees straight through it.

Parameterization
----------------
Position and moment differ by eight orders of magnitude in SI units (0.05 m
versus 1e-8 A·m), which no single learning rate can serve. Both are therefore
carried in *natural* units — centimetres and nano-ampere-metres — so an Adam
step of 0.1 means "1 mm" and "0.1 nA·m" at the same time. The conversion is a
constant rescaling, so it changes nothing about the physics or the gradient
check; it just makes the optimizer well behaved.

Orientation is left **free**: each source carries a full 3-vector moment rather
than an amplitude times the local cortical normal. That is a deliberate choice,
not a shortcut. A normal-constrained estimator would need the map from a
continuous position to a cortical normal, which is a nearest-vertex lookup — a
piecewise-constant function of position whose derivative is zero almost
everywhere and undefined on the cell boundaries. Since the forward is exactly
linear in the moment, the free-orientation derivative is exact and the estimator
is strictly more general. Ground-truth sources are still generated on cortical
normals, so the estimator is never handed the orientation it has to find.

Regularizers
------------
Two, both reported separately from the data term so they can never be confused
with the fit.

**Containment.** The optimizer is free to walk a source anywhere, including
outside the head, where the BEM's ``Brain`` domain assumption stops holding. A
smooth quadratic penalty on leaving a conservative ellipsoid fitted to the
inner-skull surface keeps it honest.

**Separation.** The characteristic degenerate solution of multi-dipole fitting is
two sources collapsing onto one another: ``K`` parameter sets explaining one
topography while a true source goes unexplained. A smooth hinge on pairwise
distances below a stated minimum prevents it. It is exactly zero for any
configuration whose sources are further apart than that minimum, so — like
containment — it is inactive at every solution the benchmark reports.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
import optax
from tesseract_jax import apply_tesseract

from neurolayout.matching import SourceMatch, match_sources
from neurolayout.noise import NoiseSpec, add_sensor_noise

__all__ = [
    "METRES_PER_CM",
    "AM_PER_NAM",
    "LocalizeConfig",
    "SourceParams",
    "Observation",
    "Containment",
    "Separation",
    "make_waveform",
    "predict_eeg",
    "sensor_positions",
    "reference_operator",
    "least_squares_moment",
    "simulate",
    "observe",
    "make_loss",
    "localization_error_mm",
    "run_optimization",
    "run_localization",
]

#: Position parameters are carried in centimetres.
METRES_PER_CM = 1e-2
#: Moment parameters are carried in nano-ampere-metres — the physiological scale
#: of an equivalent current dipole is 1–100 nA·m.
AM_PER_NAM = 1e-9


@dataclass(frozen=True)
class LocalizeConfig:
    """Everything that defines one localization problem instance.

    Attributes:
        backend: ``"openmeeg"`` (BEM) or ``"sphere"`` (analytic fallback).
        head_model: Name of a head-model artifact for the OpenMEEG backend,
            resolved inside the component against ``NEUROLOCATE_HEADMODEL_DIR``.
            ``None`` is the packaged fsaverage template.
        n_times: Samples per epoch ``T``.
        sfreq: Sampling rate, Hz — only used to shape the known waveform.
        waveform_hz: Carrier frequency of the known source waveform.
        n_channels: Sensor count. Ignored by the OpenMEEG backend, which uses the
            frozen 64-channel array unless ``channel_subset`` is set.
        channel_subset: Indices into the canonical channel order to keep, or
            ``None`` for all of them. Fixed in
            :mod:`neurolayout.channel_subsets` and never chosen from outcomes.
        fd_step: Central-difference step in metres for the position sensitivity.
        fd_order: Finite-difference stencil order.
        steps: Optimizer iterations.
        learning_rate: Adam step size, in the natural units described above.
        containment_weight: Multiplier on the out-of-head penalty.
        separation_weight: Multiplier on the minimum-separation penalty. Zero
            disables it.
        min_separation_cm: Distance below which the separation penalty engages.
        noise: Sensor-noise setting for :func:`simulate`.
        seed: Seed for random initializations.
    """

    backend: Literal["openmeeg", "sphere"] = "openmeeg"
    head_model: str | None = None
    n_times: int = 32
    sfreq: float = 160.0
    waveform_hz: float = 10.0
    n_channels: int = 64
    channel_subset: tuple[int, ...] | None = None
    fd_step: float = 1e-5
    fd_order: Literal[2, 4] = 2
    steps: int = 400
    learning_rate: float = 0.3
    containment_weight: float = 10.0
    separation_weight: float = 0.0
    min_separation_cm: float = 1.5
    noise: NoiseSpec = field(default_factory=NoiseSpec)
    seed: int = 0

    @property
    def noise_snr_db(self) -> float | None:
        """The SNR of :attr:`noise`, kept as a shorthand for result tables."""
        return self.noise.snr_db

    def static_inputs(self) -> dict[str, Any]:
        """The non-differentiable headfield inputs (static under JAX tracing).

        The two montage-mode arrays are passed explicitly, at their placeholder
        values, rather than left to the schema defaults. They are unused in this
        mode either way, but supplying them keeps every array field concrete
        during ``abstract_eval``, where a defaulted array would be serialized
        against the abstract ``ShapeDType`` schema and warn.
        """
        return {
            "mode": "localize",
            "backend": self.backend,
            "head_model": self.head_model,
            "fd_step": float(self.fd_step),
            "fd_order": int(self.fd_order),
            "n_channels": int(self.n_channels),
            # None, not an empty array: an empty numeric array fails validation
            # on the way into a served container, so "all channels" has to be the
            # absence of a subset rather than a zero-length one.
            "channel_subset": (
                None
                if self.channel_subset is None
                else np.asarray(self.channel_subset, dtype=np.int64)
            ),
            "electrode_vectors": np.array([[0.0, 0.0, 1.0]]),
            "source_activity": np.zeros((1, 1, 1)),
            "source_positions_batch": np.zeros((1, 1, 3)),
        }


@dataclass(frozen=True)
class SourceParams:
    """Source parameters in optimizer-natural units.

    Attributes:
        position_cm: ``[K, 3]`` source positions, centimetres.
        moment_nam: ``[K, 3]`` dipole moments, nano-ampere-metres.
    """

    position_cm: jnp.ndarray
    moment_nam: jnp.ndarray

    @property
    def n_sources(self) -> int:
        """``K``."""
        return int(np.asarray(self.position_cm).shape[0])

    def position_m(self) -> jnp.ndarray:
        """``[K, 3]`` positions in metres."""
        return self.position_cm * METRES_PER_CM

    def moment_am(self) -> jnp.ndarray:
        """``[K, 3]`` moments in A·m."""
        return self.moment_nam * AM_PER_NAM

    @classmethod
    def from_si(cls, position_m: np.ndarray, moment_am: np.ndarray) -> SourceParams:
        """Build from SI units; accepts ``[3]`` or ``[K, 3]``."""
        return cls(
            position_cm=jnp.asarray(np.atleast_2d(position_m) / METRES_PER_CM, jnp.float64),
            moment_nam=jnp.asarray(np.atleast_2d(moment_am) / AM_PER_NAM, jnp.float64),
        )


@dataclass(frozen=True)
class Observation:
    """A simulated measurement with the ground truth kept aside for scoring.

    Attributes:
        eeg: ``[1, C, T]`` observed sensor signals, volts.
        truth: The sources that generated it. The optimizer never sees this.
        noise: The noise setting applied.
        clean_rms: RMS of the noise-free signal, volts.
        provenance: How the data was generated — which forward model, which
            perturbations. This is what makes the inverse crime auditable.
    """

    eeg: jnp.ndarray
    truth: SourceParams
    noise: NoiseSpec
    clean_rms: float
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def noise_snr_db(self) -> float | None:
        """Shorthand for ``self.noise.snr_db``."""
        return self.noise.snr_db


@dataclass(frozen=True)
class Containment:
    """A conservative ellipsoid the sources are kept inside during optimization.

    Attributes:
        centre_cm: ``[3]`` ellipsoid centre, centimetres.
        semi_axes_cm: ``[3]`` semi-axes, centimetres.
    """

    centre_cm: np.ndarray
    semi_axes_cm: np.ndarray

    @classmethod
    def from_points(cls, points_m: np.ndarray, margin: float = 0.9) -> Containment:
        """Fit to a point cloud's bounding box, shrunk by ``margin``.

        Deliberately crude. Its only job is to stop the optimizer from walking a
        dipole out of the skull, where the forward model is meaningless; it is
        never active at the solutions we care about.
        """
        points = np.asarray(points_m, dtype=np.float64)
        low, high = points.min(axis=0), points.max(axis=0)
        return cls(
            centre_cm=(low + high) / 2.0 / METRES_PER_CM,
            semi_axes_cm=margin * (high - low) / 2.0 / METRES_PER_CM,
        )

    def penalty(self, position_cm: jnp.ndarray) -> jnp.ndarray:
        """Smooth hinge on leaving the ellipsoid; exactly zero inside it."""
        scaled = (position_cm - jnp.asarray(self.centre_cm)) / jnp.asarray(self.semi_axes_cm)
        radius = jnp.sqrt(jnp.sum(scaled**2, axis=-1) + 1e-12)
        return jnp.sum(jnp.maximum(radius - 1.0, 0.0) ** 2)

    def contains(self, position_m: np.ndarray) -> bool:
        """Whether an SI-unit point is inside the ellipsoid."""
        scaled = (np.asarray(position_m) / METRES_PER_CM - self.centre_cm) / self.semi_axes_cm
        return bool(np.sum(scaled**2) <= 1.0)


@dataclass(frozen=True)
class Separation:
    """A smooth floor on the pairwise distance between estimated sources.

    Attributes:
        min_distance_cm: Distance below which the penalty engages. The default
            1.5 cm is well below the separation any 64-channel array resolves, so
            the penalty forbids only genuinely degenerate configurations rather
            than shaping the answer.
    """

    min_distance_cm: float = 1.5

    def penalty(self, position_cm: jnp.ndarray) -> jnp.ndarray:
        """``sum_{i<j} max(0, d_min - d_ij)^2``; exactly zero when well separated."""
        positions = jnp.atleast_2d(position_cm)
        n_sources = positions.shape[0]
        if n_sources < 2:
            return jnp.asarray(0.0, positions.dtype)
        delta = positions[:, None, :] - positions[None, :, :]
        distance = jnp.sqrt(jnp.sum(delta**2, axis=-1) + 1e-12)
        violation = jnp.maximum(self.min_distance_cm - distance, 0.0) ** 2
        # Every unordered pair once; the diagonal is excluded because d_ii = 0
        # would otherwise contribute a constant K * d_min^2.
        upper = jnp.triu(jnp.ones((n_sources, n_sources)), k=1)
        return jnp.sum(violation * upper)


def reference_operator(n_channels: int) -> np.ndarray:
    """``R = I - 11ᵀ/C``, the average-reference operator in the frozen order."""
    return np.eye(n_channels) - np.full((n_channels, n_channels), 1.0 / n_channels)


def make_waveform(config: LocalizeConfig) -> np.ndarray:
    """``[T]`` known source waveform: a Hann-tapered burst at ``waveform_hz``.

    The shape is *known* to the estimator; only the moment (direction and
    amplitude) is unknown. A taper is used rather than a bare sinusoid so the
    epoch has no discontinuity at its edges.
    """
    times = np.arange(config.n_times) / config.sfreq
    taper = np.hanning(config.n_times + 2)[1:-1]
    return taper * np.sin(2.0 * np.pi * config.waveform_hz * times)


def predict_eeg(
    tesseract: Any,
    params: SourceParams,
    waveform: jnp.ndarray,
    config: LocalizeConfig,
) -> jnp.ndarray:
    """Push source parameters through the headfield Tesseract.

    Args:
        tesseract: An opened ``headfield`` Tesseract.
        params: Source parameters in natural units.
        waveform: ``[T]`` shape shared by every source, or ``[K, T]`` one per source.
        config: Problem configuration.

    Returns:
        ``[1, C, T]`` predicted sensor signals, volts.
    """
    shapes = jnp.atleast_2d(jnp.asarray(waveform, jnp.float64))  # [1 or K, T]
    timecourses = params.moment_am()[None, :, :, None] * shapes[None, :, None, :]
    outputs = apply_tesseract(
        tesseract,
        {
            "source_positions": params.position_m(),
            "source_timecourses": timecourses,
            **config.static_inputs(),
        },
    )
    return outputs["eeg"]


def sensor_positions(tesseract: Any, config: LocalizeConfig) -> np.ndarray:
    """``[C, 3]`` electrode positions the active backend evaluates at, metres.

    Needed by the spatially correlated noise model, which is a function of the
    geometry of the array rather than of the channel count.
    """
    outputs = apply_tesseract(
        tesseract,
        {
            "source_positions": np.zeros((1, 3)),
            "source_timecourses": np.zeros((1, 1, 3, 1)),
            **config.static_inputs(),
        },
    )
    return np.asarray(outputs["electrode_xyz"], dtype=np.float64)


def least_squares_moment(
    tesseract: Any,
    position_cm: np.ndarray,
    observation: Observation,
    config: LocalizeConfig,
    waveform: np.ndarray | None = None,
    *,
    floor_nam: float = 1.0,
) -> SourceParams:
    """Best moments for *fixed* positions, in closed form.

    The forward is linear in the moments of all ``K`` sources jointly, so the
    ``3K`` gain columns can be read off with ``3K`` unit-moment forward passes and
    the best-fitting moments are one least-squares solve.

    This is used only to warm-start the optimizer: from random moments the first
    tens of steps are spent rotating dipoles rather than moving them, which wastes
    solver calls and makes the error curve rise before it falls. It is *not* how
    the reported localization is obtained — positions and moments are then
    optimized jointly through the differentiable pipeline.

    Args:
        tesseract: An opened ``headfield`` Tesseract.
        position_cm: ``[K, 3]`` fixed positions.
        observation: The data to fit.
        config: Problem configuration.
        waveform: The temporal shape to assume, ``[T]`` or ``[K, T]``. Defaults to
            the benchmark's own burst. Pass whatever the *estimator* assumes — never
            the truth, unless the estimator is being given it deliberately.
        floor_nam: Smallest moment magnitude the warm start may return, in nA·m.
            This is not cosmetic: the position cotangent is proportional to the
            moment, so a warm start that solves to ``m ≈ 0`` — which happens when
            the assumed waveform is nearly orthogonal to the true one — leaves the
            position gradient exactly zero and the optimizer walks off on noise.
            Observed doing exactly that before the floor was added.

    Returns:
        Source parameters at ``position_cm`` with the fitted moments.
    """
    position_cm = np.atleast_2d(np.asarray(position_cm, dtype=np.float64))
    n_sources = position_cm.shape[0]
    if waveform is None:
        waveform = make_waveform(config)
    waveform = jnp.asarray(np.atleast_2d(np.asarray(waveform, dtype=np.float64)))
    columns = []
    for source in range(n_sources):
        for axis in np.eye(3):
            moments = np.zeros((n_sources, 3))
            moments[source] = axis  # one nA·m along this axis, this source only
            probe = SourceParams(
                position_cm=jnp.asarray(position_cm, jnp.float64),
                moment_nam=jnp.asarray(moments, jnp.float64),
            )
            columns.append(np.asarray(predict_eeg(tesseract, probe, waveform, config)).ravel())
    design = np.stack(columns, axis=1)  # [C*T, 3K], per 1 nA m per source-axis
    solution, *_ = np.linalg.lstsq(design, np.asarray(observation.eeg).ravel(), rcond=None)
    moments = solution.reshape(n_sources, 3)
    magnitudes = np.linalg.norm(moments, axis=1, keepdims=True)
    too_small = magnitudes < floor_nam
    if too_small.any():
        # Keep the fitted direction where there is one, and take a fixed direction
        # where the fit is numerically nothing at all.
        fallback = np.tile([0.0, 0.0, 1.0], (n_sources, 1))
        directions = np.where(magnitudes > 1e-12, moments / np.maximum(magnitudes, 1e-30),
                              fallback)
        moments = np.where(too_small, floor_nam * directions, moments)
    return SourceParams(
        position_cm=jnp.asarray(position_cm, jnp.float64),
        moment_nam=jnp.asarray(moments, jnp.float64),
    )


def observe(
    clean: np.ndarray,
    truth: SourceParams,
    config: LocalizeConfig,
    sensor_xyz: np.ndarray,
    *,
    provenance: dict[str, Any] | None = None,
    noise: NoiseSpec | None = None,
) -> Observation:
    """Wrap a clean forward prediction into a noisy :class:`Observation`.

    Kept separate from :func:`simulate` so that data generated by a *different*
    forward model — the whole point of :mod:`neurolayout.mismatch` — travels the
    same path into the objective, with the same noise model and the same
    provenance record.
    """
    clean = np.asarray(clean, dtype=np.float64)
    spec = config.noise if noise is None else noise
    noisy, report = add_sensor_noise(
        clean,
        sensor_xyz,
        spec,
        reference_operator=reference_operator(clean.shape[-2]),
    )
    return Observation(
        eeg=jnp.asarray(noisy, jnp.float64),
        truth=truth,
        noise=spec,
        clean_rms=float(np.sqrt(np.mean(clean**2))),
        provenance={**(provenance or {}), "noise": report},
    )


def simulate(
    tesseract: Any,
    truth: SourceParams,
    config: LocalizeConfig,
    *,
    seed: int | None = None,
) -> Observation:
    """Generate an observation with the **same** forward model used to invert.

    This is the *matched* condition, and it is an inverse crime: the estimator is
    handed data produced by its own physics, so the only errors left are
    optimization errors. It is retained deliberately, as the calibration point
    that isolates the optimizer from the model — but it is not a result about
    localization accuracy. For that see :mod:`neurolayout.mismatch`.
    """
    waveform = jnp.asarray(make_waveform(config), jnp.float64)
    clean = np.asarray(predict_eeg(tesseract, truth, waveform, config))
    noise = config.noise if seed is None else NoiseSpec(**{**vars(config.noise), "seed": seed})
    return observe(
        clean,
        truth,
        config,
        sensor_positions(tesseract, config),
        noise=noise,
        provenance={
            "generator": "matched",
            "backend": config.backend,
            "warning": "same forward model as inference (inverse crime)",
        },
    )


def make_loss(
    tesseract: Any,
    observation: Observation,
    config: LocalizeConfig,
    containment: Containment,
    separation: Separation | None = None,
):
    r"""Close the localization objective over everything but the parameters.

    .. math::

        L(\theta) = \frac{\lVert \mathrm{eeg}(\theta) - y \rVert^2}
                          {\lVert y \rVert^2}
                   + \lambda_c \, \mathrm{containment}(p)
                   + \lambda_s \, \mathrm{separation}(p)

    The data term is normalized by the observation's own energy so the number is
    comparable across trials and source depths: 1.0 means "no better than
    predicting zero", 0.0 means a perfect fit.

    Returns:
        ``(loss_fn, parts_fn)`` — a scalar callable for ``jax.value_and_grad``
        and a callable returning the terms separately for reporting.
    """
    waveform = jnp.asarray(make_waveform(config), jnp.float64)
    target = observation.eeg
    scale = jnp.sum(target**2) + 1e-30
    keep_apart = Separation(config.min_separation_cm) if separation is None else separation

    def parts(params: SourceParams) -> dict[str, jnp.ndarray]:
        residual = predict_eeg(tesseract, params, waveform, config) - target
        return {
            "data": jnp.sum(residual**2) / scale,
            "containment": containment.penalty(params.position_cm),
            "separation": keep_apart.penalty(params.position_cm),
        }

    def loss(params: SourceParams) -> jnp.ndarray:
        terms = parts(params)
        return (
            terms["data"]
            + config.containment_weight * terms["containment"]
            + config.separation_weight * terms["separation"]
        )

    return loss, parts


def localization_error_mm(estimate: SourceParams, truth: SourceParams) -> np.ndarray:
    """Per-source distance under the optimal assignment, in millimetres.

    Returned in estimate order, so entry ``k`` belongs to estimated source ``k``
    but is measured against whichever true source it was matched to.
    """
    return match_sources(
        np.asarray(estimate.position_cm) * METRES_PER_CM,
        np.asarray(truth.position_cm) * METRES_PER_CM,
    ).errors_mm


def _match(estimate: SourceParams, truth: SourceParams) -> SourceMatch:
    return match_sources(
        np.asarray(estimate.position_cm) * METRES_PER_CM,
        np.asarray(truth.position_cm) * METRES_PER_CM,
    )


def run_optimization(
    loss_fn: Callable[[Any], jnp.ndarray],
    parts_fn: Callable[[Any], dict[str, jnp.ndarray]],
    initial: Any,
    truth: SourceParams,
    config: LocalizeConfig,
    *,
    positions_of: Callable[[Any], SourceParams] = lambda params: params,
    record_every: int = 1,
    optimizer: Any = None,
) -> dict[str, Any]:
    """Run Optax on a localization objective and record the trajectory.

    The whole point of this function is the single line
    ``jax.value_and_grad(loss_fn)(params)``: evaluating it sends source
    parameters into containers running OpenMEEG (and, in the learned-prior
    configuration, PyTorch) and gets a gradient back.

    Args:
        loss_fn: Scalar objective over the parameter pytree.
        parts_fn: Returns the objective's terms separately, for reporting.
        initial: Starting parameters (any registered pytree).
        truth: Ground-truth sources, used only for scoring.
        config: Problem configuration.
        positions_of: Extracts a :class:`SourceParams` view of the parameters, so
            the latent-prior parameterization can be scored the same way.
        record_every: Trajectory sampling interval.
        optimizer: An Optax transformation to use instead of plain
            ``adam(config.learning_rate)``. A single step size serves the
            synthetic benchmark, where the conditioning of position, moment and
            waveform is fixed and known. Data whose amplitude spans several
            orders of magnitude needs a per-leaf optimizer here instead, rather
            than a rescaling of the physics.

    Returns:
        A dict with the final parameters, the loss and error curves, the
        trajectory, the multi-source diagnostics, and timing.
    """
    value_and_grad = jax.value_and_grad(loss_fn)
    if optimizer is None:
        optimizer = optax.adam(config.learning_rate)
    params = initial
    state = optimizer.init(params)

    history: dict[str, list] = {
        "step": [], "loss": [], "data": [], "containment": [], "separation": [],
        "error_mm": [], "error_mm_max": [], "error_mm_per_source": [],
        "position_cm": [], "grad_norm": [], "min_separation_mm": [],
    }

    def record(step: int, params: Any, loss: float, grad_norm: float) -> None:
        terms = parts_fn(params)
        sources = positions_of(params)
        match = _match(sources, truth)
        history["step"].append(step)
        history["loss"].append(float(loss))
        history["data"].append(float(terms["data"]))
        history["containment"].append(float(terms["containment"]))
        history["separation"].append(float(terms.get("separation", 0.0)))
        history["error_mm"].append(match.mean_error_mm)
        history["error_mm_max"].append(match.max_error_mm)
        history["error_mm_per_source"].append([float(e) for e in match.errors_mm])
        history["position_cm"].append(np.asarray(sources.position_cm).tolist())
        history["grad_norm"].append(grad_norm)
        history["min_separation_mm"].append(
            None if np.isinf(match.min_separation_mm) else match.min_separation_mm
        )

    start = time.perf_counter()
    record(0, params, float(loss_fn(params)), float("nan"))

    for step in range(1, config.steps + 1):
        loss, grads = value_and_grad(params)
        grad_norm = float(
            jnp.sqrt(sum(jnp.sum(leaf**2) for leaf in jax.tree_util.tree_leaves(grads)))
        )
        updates, state = optimizer.update(grads, state, params)
        params = optax.apply_updates(params, updates)
        # `loss` is the value *before* this step; recording the post-step loss
        # keeps the curve aligned with the position it is plotted against.
        if step % record_every == 0 or step == config.steps:
            record(step, params, float(loss_fn(params)), grad_norm)
    seconds = time.perf_counter() - start

    final_parts = parts_fn(params)
    final_sources = positions_of(params)
    final_match = _match(final_sources, truth)
    initial_match = _match(positions_of(initial), truth)
    return {
        "final": params,
        "final_sources": final_sources,
        "history": history,
        "seconds": seconds,
        "steps": config.steps,
        "n_sources": truth.n_sources,
        "initial_error_mm": initial_match.mean_error_mm,
        "initial_error_mm_max": initial_match.max_error_mm,
        "final_error_mm": final_match.mean_error_mm,
        "final_error_mm_max": final_match.max_error_mm,
        "final_error_mm_per_source": [float(e) for e in final_match.errors_mm],
        "match": final_match.to_dict(),
        "initial_loss": history["loss"][0],
        "final_loss": float(loss_fn(params)),
        "final_data_loss": float(final_parts["data"]),
        "final_containment": float(final_parts["containment"]),
        "final_separation": float(final_parts.get("separation", 0.0)),
    }


def run_localization(
    tesseract: Any,
    observation: Observation,
    initial: SourceParams,
    config: LocalizeConfig,
    containment: Containment,
    *,
    separation: Separation | None = None,
    record_every: int = 1,
) -> dict[str, Any]:
    """Localize ``K`` sources from an observation, scoring against its truth."""
    loss_fn, parts_fn = make_loss(tesseract, observation, config, containment, separation)
    return run_optimization(
        loss_fn,
        parts_fn,
        initial,
        observation.truth,
        config,
        record_every=record_every,
    )


# `SourceParams` has to be a JAX pytree for Optax to treat it as parameters.
jax.tree_util.register_pytree_node(
    SourceParams,
    lambda p: ((p.position_cm, p.moment_nam), None),
    lambda _, children: SourceParams(*children),
)
