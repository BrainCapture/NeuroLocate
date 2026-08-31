# Copyright 2026 NeuroLocate contributors
# SPDX-License-Identifier: Apache-2.0
r"""The K-source forward map and its derivative, independent of any solver.

The forward is deliberately trivial to state:

.. math::

    \mathrm{eeg}[b, c, t] = \sum_{k, j} G(p)[c, k, j] \; q[b, k, j, t]

where ``p`` are the ``K`` source positions and ``q`` their vector moment time
courses (A·m). Everything mechanistic hides inside ``G``, supplied as a callable
``positions -> [C, K, 3]`` by whichever backend is active. That is the entire
coupling between this module and OpenMEEG.

Splitting it this way is what makes the derivative tractable:

``q``
    The forward is **linear** in the moment time courses, so their cotangent is
    exact and costs one contraction. The "unknown amplitude of a known
    waveform" case is just ``q = m \otimes w(t)``, and its derivative comes out
    of the same rule.

``p``
    ``G`` is a compiled BEM solve with no derivative of any kind. It is also a
    function of only ``3K`` numbers, so central differences cost ``6K`` extra
    gain evaluations — about 10 ms for ``K = 1`` — and are, unusually for finite
    differences, an excellent choice here: OpenMEEG's source term is assembled
    by analytic integration, so ``G`` is smooth to machine precision and the
    error is pure truncation over five decades of step size rather than the
    usual V-shaped trade-off. ``tests/test_source_vjp.py`` sweeps it and pins
    both the O(h²) slope and the round-off floor.

:func:`position_jacobian` also offers a 4th-order Richardson stencil, used to
audit the 2nd-order one rather than in the hot loop.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np

__all__ = [
    "DEFAULT_FD_STEP",
    "GainFunction",
    "average_reference_operator",
    "forward",
    "backward",
    "forward_batched",
    "backward_batched",
    "position_jacobian",
    "moment_timecourse",
]

#: Central-difference step for source positions, in metres (10 µm).
#:
#: Chosen from the measured sweep in ``results/source_vjp.json``. Because
#: OpenMEEG assembles the source term by analytic integration, the gain is smooth
#: to machine precision and the error is pure O(h²) truncation all the way down
#: to h = 1e-7 m, where cancellation finally takes over. The measured relative
#: error is 5e-8 at this step and 5.5e-10 at the 1 µm optimum — both four to six
#: orders below the 1e-3 target — so the step is set two decades above the
#: round-off floor for margin rather than at the minimum.
DEFAULT_FD_STEP = 1e-5

#: ``positions [P, 3] -> gain [C, P, 3]``, in volts per A·m.
GainFunction = Callable[[np.ndarray], np.ndarray]


def average_reference_operator(n_channels: int) -> np.ndarray:
    r"""``R = I - 11ᵀ/C``, the average-reference operator for ``C`` channels.

    The epochs this project is calibrated against are average-referenced,
    so every forward operator carries the same projection. It lives here, in the
    solver-free module, because both backends and every channel subset need
    exactly the same matrix — a subset of an array gets the reference *its own*
    channel count implies, not a restriction of a larger one.
    """
    if n_channels < 2:
        raise ValueError(f"an average reference needs at least 2 channels, got {n_channels}")
    return np.eye(n_channels, dtype=np.float64) - np.full(
        (n_channels, n_channels), 1.0 / n_channels
    )


def moment_timecourse(moments: np.ndarray, waveforms: np.ndarray) -> np.ndarray:
    """Build ``q[b, k, j, t]`` from per-source moments and per-epoch waveforms.

    The "known waveform shape, unknown amplitude and orientation" model: the
    amplitude is folded into the length of the moment vector.

    Args:
        moments: ``[K, 3]`` or ``[B, K, 3]`` dipole moments in A·m.
        waveforms: ``[T]`` or ``[B, T]`` unit-scale time courses.

    Returns:
        ``[B, K, 3, T]``.
    """
    moments = np.asarray(moments, dtype=np.float64)
    waveforms = np.asarray(waveforms, dtype=np.float64)
    if moments.ndim == 2:
        moments = moments[None]
    if waveforms.ndim == 1:
        waveforms = waveforms[None]
    if moments.shape[0] == 1 and waveforms.shape[0] > 1:
        moments = np.repeat(moments, waveforms.shape[0], axis=0)
    if waveforms.shape[0] == 1 and moments.shape[0] > 1:
        waveforms = np.repeat(waveforms, moments.shape[0], axis=0)
    return np.einsum("bkj,bt->bkjt", moments, waveforms)


def forward(
    gain_fn: GainFunction, positions: np.ndarray, timecourses: np.ndarray
) -> dict[str, np.ndarray]:
    """Predict sensor signals for ``K`` sources.

    Args:
        gain_fn: Backend gain callable.
        positions: ``[K, 3]`` source positions, metres, head frame.
        timecourses: ``[B, K, 3, T]`` vector moment time courses, A·m.

    Returns:
        Dict with ``eeg`` ``[B, C, T]`` and the cached ``gain`` ``[C, K, 3]``
        needed by :func:`backward`.
    """
    positions = np.asarray(positions, dtype=np.float64)
    timecourses = np.asarray(timecourses, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must be [K, 3], got {positions.shape}")
    if timecourses.ndim != 4 or timecourses.shape[1:3] != (positions.shape[0], 3):
        raise ValueError(
            f"timecourses must be [B, {positions.shape[0]}, 3, T], got {timecourses.shape}"
        )
    gain = gain_fn(positions)
    return {"gain": gain, "eeg": np.einsum("ckj,bkjt->bct", gain, timecourses)}


def position_jacobian(
    gain_fn: GainFunction,
    positions: np.ndarray,
    step: float = DEFAULT_FD_STEP,
    order: Literal[2, 4] = 2,
) -> np.ndarray:
    """Sensitivity of the gain to each source coordinate, by finite differences.

    Every perturbed point is evaluated in a single batched backend call, so the
    cost is one solver invocation regardless of ``K``.

    Args:
        gain_fn: Backend gain callable.
        positions: ``[K, 3]`` source positions.
        step: Perturbation in metres.
        order: ``2`` for the central stencil, ``4`` for the Richardson-style
            five-point stencil used to audit it.

    Returns:
        ``[C, K, 3, 3]``: entry ``(c, k, j, d)`` is
        ``d gain[c, k, j] / d positions[k, d]``.
    """
    positions = np.asarray(positions, dtype=np.float64)
    n_sources = positions.shape[0]
    offsets = (-1.0, 1.0) if order == 2 else (-2.0, -1.0, 1.0, 2.0)
    if order == 2:
        weights = np.array([-0.5, 0.5]) / step
    elif order == 4:
        weights = np.array([1.0, -8.0, 8.0, -1.0]) / (12.0 * step)
    else:
        raise ValueError(f"order must be 2 or 4, got {order}")

    shifts = step * np.asarray(offsets)
    # [K, 3(coord), n_offsets, 3(xyz)] flattened into one batched gain call.
    probes = np.repeat(positions[:, None, None, :], 3, axis=1)
    probes = np.repeat(probes, len(shifts), axis=2)
    for axis in range(3):
        probes[:, axis, :, axis] += shifts
    flat = probes.reshape(-1, 3)

    gains = gain_fn(flat)  # [C, K*3*n_offsets, 3]
    n_channels = gains.shape[0]
    gains = gains.reshape(n_channels, n_sources, 3, len(shifts), 3)
    # (c, k, coord, offset, moment) -> (c, k, moment, coord)
    return np.einsum("o,ckdoj->ckjd", weights, gains)


def backward(
    cache: dict[str, np.ndarray],
    gain_fn: GainFunction,
    positions: np.ndarray,
    timecourses: np.ndarray,
    grad_eeg: np.ndarray,
    *,
    step: float = DEFAULT_FD_STEP,
    order: Literal[2, 4] = 2,
    need_positions: bool = True,
    need_timecourses: bool = True,
) -> dict[str, np.ndarray]:
    """Reverse-mode sensitivities of :func:`forward`.

    Args:
        cache: The dict :func:`forward` returned at this primal point.
        gain_fn: Backend gain callable (only used when ``need_positions``).
        positions: ``[K, 3]`` source positions.
        timecourses: ``[B, K, 3, T]`` moment time courses.
        grad_eeg: ``[B, C, T]`` cotangent on the predicted EEG.
        step: Finite-difference step for the position sensitivity, metres.
        order: Finite-difference stencil order.
        need_positions: Skip the solver calls when the position cotangent is
            not requested — this is the expensive half.
        need_timecourses: Skip the (cheap) time-course contraction.

    Returns:
        Dict with ``source_positions`` ``[K, 3]`` and ``source_timecourses``
        ``[B, K, 3, T]``.
    """
    positions = np.asarray(positions, dtype=np.float64)
    timecourses = np.asarray(timecourses, dtype=np.float64)
    grad_eeg = np.asarray(grad_eeg, dtype=np.float64)

    grads: dict[str, np.ndarray] = {}
    if need_timecourses:
        grads["source_timecourses"] = np.einsum("ckj,bct->bkjt", cache["gain"], grad_eeg)
    else:
        grads["source_timecourses"] = np.zeros_like(timecourses)

    if need_positions:
        # Cotangent on the gain itself, then chain through dG/dp.
        grad_gain = np.einsum("bct,bkjt->ckj", grad_eeg, timecourses)
        jacobian = position_jacobian(gain_fn, positions, step=step, order=order)
        grads["source_positions"] = np.einsum("ckj,ckjd->kd", grad_gain, jacobian)
    else:
        grads["source_positions"] = np.zeros_like(positions)
    return grads


def forward_batched(
    gain_fn: GainFunction, positions: np.ndarray, timecourses: np.ndarray
) -> dict[str, np.ndarray]:
    r"""Predict sensor signals for a **batch** of independent source sets.

    .. math::

        \mathrm{eeg}[b, c, t] = \sum_{k, j} G(p_b)[c, k, j] \; q[b, k, j, t]

    The difference from :func:`forward` is that each batch entry carries its own
    ``K`` positions instead of sharing one set. That is what a training step over
    ``B`` proposals needs, and doing it by looping is ``B`` separate BEM source
    assemblies for no reason: the assembly cost is per *dipole*, so the whole
    batch's ``B x K`` positions go into one call.

    The same result is reachable through :func:`forward` by flattening the batch
    into ``B*K`` shared sources and zeroing the off-block time courses, and
    ``tests/test_batched_headfield.py`` checks that it agrees — but that
    construction carries a ``[B, B*K, 3, T]`` array across the component boundary
    and contracts it in full, which is quadratic in the batch for an answer that
    is linear in it.

    Args:
        gain_fn: Backend gain callable.
        positions: ``[B, K, 3]`` source positions, metres, head frame.
        timecourses: ``[B, K, 3, T]`` vector moment time courses, A·m.

    Returns:
        Dict with ``eeg`` ``[B, C, T]`` and the cached ``gain`` ``[C, B, K, 3]``.
    """
    positions = np.asarray(positions, dtype=np.float64)
    timecourses = np.asarray(timecourses, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[2] != 3:
        raise ValueError(f"positions must be [B, K, 3], got {positions.shape}")
    batch, n_sources = positions.shape[0], positions.shape[1]
    if timecourses.ndim != 4 or timecourses.shape[:3] != (batch, n_sources, 3):
        raise ValueError(
            f"timecourses must be [{batch}, {n_sources}, 3, T], got {timecourses.shape}"
        )
    gain = gain_fn(positions.reshape(-1, 3))
    gain = gain.reshape(gain.shape[0], batch, n_sources, 3)
    return {"gain": gain, "eeg": np.einsum("cbkj,bkjt->bct", gain, timecourses)}


def backward_batched(
    cache: dict[str, np.ndarray],
    gain_fn: GainFunction,
    positions: np.ndarray,
    timecourses: np.ndarray,
    grad_eeg: np.ndarray,
    *,
    step: float = DEFAULT_FD_STEP,
    order: Literal[2, 4] = 2,
    need_positions: bool = True,
    need_timecourses: bool = True,
) -> dict[str, np.ndarray]:
    """Reverse-mode sensitivities of :func:`forward_batched`.

    One batched solver call for the position sensitivity of the whole batch —
    ``6 * B * K`` perturbed dipoles in a single assembly — rather than one call
    per batch entry.

    Args:
        cache: The dict :func:`forward_batched` returned at this primal point.
        gain_fn: Backend gain callable (only used when ``need_positions``).
        positions: ``[B, K, 3]`` source positions.
        timecourses: ``[B, K, 3, T]`` moment time courses.
        grad_eeg: ``[B, C, T]`` cotangent on the predicted EEG.
        step: Finite-difference step for the position sensitivity, metres.
        order: Finite-difference stencil order.
        need_positions: Skip the solver calls when not requested.
        need_timecourses: Skip the (cheap) time-course contraction.

    Returns:
        Dict with ``source_positions_batch`` ``[B, K, 3]`` and
        ``source_timecourses`` ``[B, K, 3, T]``.
    """
    positions = np.asarray(positions, dtype=np.float64)
    timecourses = np.asarray(timecourses, dtype=np.float64)
    grad_eeg = np.asarray(grad_eeg, dtype=np.float64)
    batch, n_sources = positions.shape[0], positions.shape[1]

    grads: dict[str, np.ndarray] = {}
    if need_timecourses:
        grads["source_timecourses"] = np.einsum("cbkj,bct->bkjt", cache["gain"], grad_eeg)
    else:
        grads["source_timecourses"] = np.zeros_like(timecourses)

    if need_positions:
        grad_gain = np.einsum("bct,bkjt->cbkj", grad_eeg, timecourses)
        jacobian = position_jacobian(
            gain_fn, positions.reshape(-1, 3), step=step, order=order
        )
        jacobian = jacobian.reshape(jacobian.shape[0], batch, n_sources, 3, 3)
        grads["source_positions_batch"] = np.einsum(
            "cbkj,cbkjd->bkd", grad_gain, jacobian
        )
    else:
        grads["source_positions_batch"] = np.zeros_like(positions)
    return grads
