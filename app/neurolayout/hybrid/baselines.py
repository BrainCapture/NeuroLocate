# Copyright 2026 NeuroLocate contributors
# SPDX-License-Identifier: Apache-2.0
r"""The classical multi-source baselines this matrix is measured against.

Two, and both run on **identical physics**: the same OpenMEEG symmetric-BEM gain
the differentiable estimator inverts with, substituted into MNE's own bookkeeping
by :class:`neurolayout.baselines.MneInverse`. A baseline on a different forward
model would be measuring the forward model.

``rapmusic``
    Recursively applied MUSIC. The right comparison for this matrix and the
    hardest one: RAP-MUSIC works in the *signal subspace*, which is exactly the
    structure that correlated sources destroy, and it was designed for the
    multi-source case. If the hybrid estimator cannot beat RAP-MUSIC on shared
    dynamics then the learned proposal is not buying anything a classical
    subspace method does not already have.

``scan``
    The discrete OpenMEEG dipole scan extended to ``K`` sources by orthogonal
    matching pursuit with alternating refinement, carried over unchanged from
    :class:`neurolayout.baselines.DipoleDictionary`. It needs a temporal profile
    to project onto, and in this matrix the time courses are unknown, so it is
    given the epoch's **leading right singular vector** — the best rank-one
    temporal description of the data. That is what a dipole scan does when the
    waveform is not known, and it is exactly right in the ``shared`` regime, where
    the data really is rank one.

Both are cortically constrained: their estimate can only be a location in the
8196-vertex source space, and every truth in this matrix is deliberately placed
*between* candidates. That limit is a property of the method class and is
reported rather than corrected for — the same way
``docs/BENCHMARK.md`` reports it.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from neurolayout.baselines import BaselineResult

__all__ = ["leading_temporal_component", "rapmusic", "scan"]


def leading_temporal_component(observed: np.ndarray) -> np.ndarray:
    """``[T]`` the epoch's leading right singular vector, unit norm.

    The best rank-one temporal description of the data, and what a scan method
    projects onto when the source waveform is unknown.
    """
    data = np.asarray(observed, dtype=np.float64)
    _, _, right = np.linalg.svd(data, full_matrices=False)
    component = right[0]
    return component / max(float(np.linalg.norm(component)), 1e-30)


def rapmusic(
    inverse: Any,
    observed: np.ndarray,
    n_sources: int,
    *,
    noise_rms_v: float,
    signal_rms_v: float,
    channels: tuple[int, ...] | None = None,
) -> BaselineResult:
    """Recursively applied MUSIC on the substituted OpenMEEG forward.

    Args:
        inverse: A :class:`neurolayout.baselines.MneInverse`.
        observed: ``[C, T]`` measured signals, volts.
        n_sources: ``K``.
        noise_rms_v: Noise level, for the diagonal covariance.
        signal_rms_v: Signal level, so the noise-free case still has an
            invertible covariance.
        channels: Channel subset, or ``None`` for all.

    Returns:
        A :class:`~neurolayout.baselines.BaselineResult` with ``K`` positions,
        padded with ``nan`` rows if MUSIC returned fewer.
    """
    from mne.beamformer import rap_music

    start = time.perf_counter()
    evoked = inverse._evoked(observed, channels)  # noqa: SLF001
    covariance = inverse._covariance(noise_rms_v, signal_rms_v, channels)  # noqa: SLF001
    dipoles, residual = rap_music(
        evoked,
        inverse.forward_for(channels),
        covariance,
        n_dipoles=n_sources,
        return_residual=True,
        verbose="ERROR",
    )
    positions = np.full((n_sources, 3), np.nan)
    for index, dipole in enumerate(dipoles[:n_sources]):
        # One dipole per recursion, each with a single position repeated over the
        # epoch; the first row is that position.
        positions[index] = np.asarray(dipole.pos)[0]
    observed_energy = float(np.sum(np.asarray(observed) ** 2))
    residual_fraction = (
        None
        if observed_energy <= 0.0
        else float(np.sum(np.asarray(residual.data) ** 2) / observed_energy)
    )
    return BaselineResult(
        name="rapmusic",
        positions_m=positions,
        seconds=time.perf_counter() - start,
        n_found=len(dipoles),
        residual_fraction=residual_fraction,
        detail={"n_dipoles_requested": n_sources},
    )


def scan(
    dictionary: Any,
    observed: np.ndarray,
    n_sources: int,
    *,
    channels: tuple[int, ...] | None = None,
) -> BaselineResult:
    """The discrete OpenMEEG scan, projected onto the epoch's leading component."""
    result = dictionary.scan(
        observed,
        leading_temporal_component(observed),
        n_sources,
        channels=channels,
    )
    result.detail = {
        **result.detail,
        "temporal_profile": "leading right singular vector of the epoch",
    }
    return result
