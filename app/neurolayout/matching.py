# Copyright 2026 NeuroLocate contributors
# SPDX-License-Identifier: Apache-2.0
r"""Scoring ``K`` recovered sources against ``K`` true ones.

With more than one source the estimate is a *set*, not a list: nothing in the
objective distinguishes "source 1" from "source 2", and an optimizer that finds
both true locations but in the opposite order has succeeded, not failed.
Comparing them index by index would report that success as a large error.

So the localization error is defined through the assignment that minimizes the
total distance,

.. math::

    E = \min_{\pi \in S_K} \sum_k \lVert \hat p_k - p^*_{\pi(k)} \rVert,

which is a linear assignment problem. For ``K <= 7`` this module solves it by
enumerating all ``K!`` permutations — exact, dependency-free, and microseconds at
these sizes — and falls back to SciPy's Hungarian solver above that.

Two multi-source failure modes are reported alongside the error, because a mean
distance hides both of them:

**Swap.** The optimal assignment is not the identity. Harmless for the error, but
worth counting: it means the trial's per-source curves cannot be read in
parameter order.

**Collapse.** Two estimated sources sit on top of each other, so ``K`` parameters
are explaining one topography and at least one true source is unaccounted for.
This is the characteristic degenerate solution of multi-dipole fitting and the
reason :mod:`neurolayout.localize` offers a minimum-separation penalty.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

import numpy as np

__all__ = [
    "BRUTE_FORCE_MAX_K",
    "COLLAPSE_THRESHOLD_MM",
    "SourceMatch",
    "match_sources",
    "min_separation_mm",
]

#: Above this ``K`` the assignment goes to SciPy instead of enumeration.
BRUTE_FORCE_MAX_K = 7

#: Two estimated sources closer than this are counted as collapsed, in mm. Chosen
#: as roughly five times the ico5 source-space spacing (2.1 mm): closer than this
#: and the two dipoles are not separable by any 64-channel EEG estimator, so they
#: are one source wearing two hats.
COLLAPSE_THRESHOLD_MM = 10.0


@dataclass(frozen=True)
class SourceMatch:
    """The optimal pairing of estimated to true sources.

    Attributes:
        assignment: ``[K]`` — ``assignment[k]`` is the index of the true source
            paired with estimate ``k``.
        errors_mm: ``[K]`` distance of each estimate from its matched truth, in
            millimetres, in estimate order.
        swapped: Whether the optimal assignment differs from the identity.
        collapsed: Whether any two estimates are within
            :data:`COLLAPSE_THRESHOLD_MM` of each other.
        min_separation_mm: Smallest pairwise distance among the estimates
            (``inf`` for ``K = 1``).
    """

    assignment: np.ndarray
    errors_mm: np.ndarray
    swapped: bool
    collapsed: bool
    min_separation_mm: float

    @property
    def mean_error_mm(self) -> float:
        """Mean matched localization error, mm."""
        return float(np.mean(self.errors_mm))

    @property
    def median_error_mm(self) -> float:
        """Median matched localization error, mm."""
        return float(np.median(self.errors_mm))

    @property
    def max_error_mm(self) -> float:
        """Worst matched localization error, mm — the number that hides least."""
        return float(np.max(self.errors_mm))

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable summary."""
        return {
            "assignment": [int(i) for i in self.assignment],
            "errors_mm": [float(e) for e in self.errors_mm],
            "mean_error_mm": self.mean_error_mm,
            "median_error_mm": self.median_error_mm,
            "max_error_mm": self.max_error_mm,
            "swapped": bool(self.swapped),
            "collapsed": bool(self.collapsed),
            "min_separation_mm": (
                None if np.isinf(self.min_separation_mm) else float(self.min_separation_mm)
            ),
        }


def min_separation_mm(positions_m: np.ndarray) -> float:
    """Smallest pairwise distance in a set of positions, mm (``inf`` if ``K < 2``)."""
    points = np.atleast_2d(np.asarray(positions_m, dtype=np.float64))
    if points.shape[0] < 2:
        return float("inf")
    distance = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    np.fill_diagonal(distance, np.inf)
    return float(distance.min() * 1e3)


def _solve_assignment(cost: np.ndarray) -> np.ndarray:
    """Minimum-cost assignment of rows to columns of a square cost matrix."""
    n = cost.shape[0]
    if n <= BRUTE_FORCE_MAX_K:
        rows = np.arange(n)
        best = min(
            permutations(range(n)), key=lambda perm: float(cost[rows, list(perm)].sum())
        )
        return np.asarray(best, dtype=int)
    from scipy.optimize import linear_sum_assignment

    _, columns = linear_sum_assignment(cost)
    return np.asarray(columns, dtype=int)


def match_sources(
    estimate_m: np.ndarray,
    truth_m: np.ndarray,
    *,
    collapse_threshold_mm: float = COLLAPSE_THRESHOLD_MM,
) -> SourceMatch:
    """Pair estimated with true source positions by minimum total distance.

    Args:
        estimate_m: ``[K, 3]`` recovered positions, metres.
        truth_m: ``[K, 3]`` true positions, metres.
        collapse_threshold_mm: Distance below which two estimates count as
            collapsed onto each other.

    Returns:
        A :class:`SourceMatch`.

    Raises:
        ValueError: If the two sets have different sizes. Estimating a different
            number of sources than were simulated is a different experiment and
            this function deliberately refuses to score it.
    """
    estimate = np.atleast_2d(np.asarray(estimate_m, dtype=np.float64))
    truth = np.atleast_2d(np.asarray(truth_m, dtype=np.float64))
    if estimate.shape != truth.shape:
        raise ValueError(
            f"estimate {estimate.shape} and truth {truth.shape} must have the same shape; "
            "matching K estimates against a different number of true sources is undefined"
        )

    distance_mm = np.linalg.norm(estimate[:, None, :] - truth[None, :, :], axis=-1) * 1e3
    assignment = _solve_assignment(distance_mm)
    errors = distance_mm[np.arange(len(assignment)), assignment]
    separation = min_separation_mm(estimate)
    return SourceMatch(
        assignment=assignment,
        errors_mm=errors,
        swapped=bool(np.any(assignment != np.arange(len(assignment)))),
        collapsed=bool(separation < collapse_threshold_mm),
        min_separation_mm=separation,
    )
