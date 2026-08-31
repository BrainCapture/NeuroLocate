# Copyright 2026 NeuroLocate contributors
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures and finite-difference helpers for the test gates.

Almost the whole suite runs against ``transport="local"``
(:meth:`Tesseract.from_tesseract_api`) rather than built container images. That
keeps the gradient gates runnable in CI and on a machine without a container
runtime, at the cost of not exercising the container boundary — which
``make build`` and ``make test-images`` cover separately, and which
``tests/test_image_transport.py`` checks against the in-process numbers.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pytest
from neurolayout.clients import open_component

#: Tolerance for every finite-difference gate. Central differences at
#: `FD_STEP` on a float64 pipeline land several orders of magnitude below the
#: 1e-3 target, so the gate is tightened accordingly.
FD_TOLERANCE = 1e-6

#: Central-difference step. Large enough that float64 cancellation stays well
#: below the truncation error, small enough that the O(h^2) truncation term is
#: negligible for these smooth functions.
FD_STEP = 1e-6


@dataclass(frozen=True)
class MontageConfig:
    """A problem instance for the headfield component's ``montage`` mode.

    That mode is the analytic homogeneous-sphere forward: continuous electrode
    design variables sample a cached dense sphere lead field through a smooth
    interpolant, with a hand-written analytic reverse-mode rule. It is not on the
    source-localization path, and it is kept because it is an independent check
    on the BEM and a solver that needs no cached anatomy — so its gradient gate
    is kept too.

    Attributes:
        n_electrodes: Montage size ``K``.
        n_times: Samples per epoch ``T``.
        n_scalp: Scalp lattice size of the cached dense lead field.
        n_sources: Source-basis size ``S``.
        kappa: Locality of the headfield's smooth scalp sampler.
        seed: Seed for the deterministic primal point.
    """

    n_electrodes: int = 5
    n_times: int = 24
    n_scalp: int = 42
    n_sources: int = 16
    kappa: float = 40.0
    seed: int = 7

    def headfield_static(self) -> dict:
        """Non-differentiable headfield inputs (static under JAX tracing).

        The localization-mode arrays are passed at their placeholder values so
        that every array field is concrete during ``abstract_eval``; they are
        unused in ``mode="montage"``.
        """
        return {
            "mode": "montage",
            "kappa": float(self.kappa),
            "n_scalp": int(self.n_scalp),
            "n_sources": int(self.n_sources),
            "source_positions": np.zeros((1, 3)),
            "source_timecourses": np.zeros((1, 1, 3, 1)),
        }


@dataclass(frozen=True)
class HeadfieldOnly:
    """Adapter so a gate can say ``tesseracts.headfield`` and mean one client."""

    headfield: object


@pytest.fixture(scope="session")
def tesseracts() -> Iterator[HeadfieldOnly]:
    """The headfield Tesseract, opened once for the whole session."""
    with open_component("headfield", "local") as headfield:
        yield HeadfieldOnly(headfield=headfield)


@pytest.fixture(scope="session")
def tiny_config() -> MontageConfig:
    """Smallest configuration that still exercises every montage-mode path."""
    return MontageConfig()


def central_difference(
    scalar_fn, x: np.ndarray, indices: list[tuple[int, ...]], step: float = FD_STEP
) -> np.ndarray:
    """Central-difference derivative of ``scalar_fn`` at the given indices.

    Args:
        scalar_fn: Callable mapping an array like ``x`` to a Python float.
        x: Point to differentiate at.
        indices: Entries of ``x`` to probe. Others are left at zero.
        step: Half-width of the difference.

    Returns:
        Array shaped like ``x``, populated only at ``indices``.
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    for index in indices:
        plus, minus = x.copy(), x.copy()
        plus[index] += step
        minus[index] -= step
        out[index] = (scalar_fn(plus) - scalar_fn(minus)) / (2.0 * step)
    return out


def sample_indices(
    shape: tuple[int, ...], limit: int, seed: int = 0
) -> list[tuple[int, ...]]:
    """Up to ``limit`` deterministically chosen multi-indices into ``shape``.

    Finite differences cost two forward passes per entry, so large arrays are
    subsampled rather than probed exhaustively.
    """
    all_indices = list(np.ndindex(*shape))
    if len(all_indices) <= limit:
        return all_indices
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(all_indices), size=limit, replace=False)
    return [all_indices[i] for i in sorted(chosen)]


def relative_error(analytic: np.ndarray, numeric: np.ndarray, mask: np.ndarray) -> float:
    """Max absolute deviation over ``mask``, normalized by the numeric scale."""
    scale = max(float(np.abs(numeric[mask]).max()), 1e-30)
    return float(np.abs(analytic[mask] - numeric[mask]).max() / scale)


def mask_from_indices(shape: tuple[int, ...], indices: list[tuple[int, ...]]) -> np.ndarray:
    """Boolean mask of ``shape`` that is True exactly at ``indices``."""
    mask = np.zeros(shape, dtype=bool)
    for index in indices:
        mask[index] = True
    return mask
