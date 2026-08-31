# Copyright 2026 NeuroLocate contributors
# SPDX-License-Identifier: Apache-2.0
"""Multi-source scoring: the assignment has to be optimal, not positional."""

from __future__ import annotations

import numpy as np
import pytest
from neurolayout.matching import (
    COLLAPSE_THRESHOLD_MM,
    match_sources,
    min_separation_mm,
)


def test_k1_is_just_a_distance() -> None:
    match = match_sources(np.array([[0.003, 0.004, 0.0]]), np.zeros((1, 3)))
    np.testing.assert_allclose(match.errors_mm, [5.0])
    assert not match.swapped
    assert not match.collapsed
    assert np.isinf(match.min_separation_mm)


def test_reversed_order_is_not_an_error() -> None:
    """The estimate is a set: finding both sources in the other order is success."""
    truth = np.array([[0.04, 0.0, 0.0], [-0.04, 0.0, 0.0]])
    estimate = truth[::-1] + np.array([[0.001, 0.0, 0.0], [0.0, 0.002, 0.0]])
    match = match_sources(estimate, truth)
    assert match.swapped
    np.testing.assert_allclose(np.sort(match.errors_mm), [1.0, 2.0], atol=1e-9)
    assert match.assignment.tolist() == [1, 0]


def test_assignment_minimizes_total_distance_not_greedy_choice() -> None:
    """A greedy nearest-truth pairing gets this one wrong; the optimum does not."""
    truth = np.array([[0.0, 0.0, 0.0], [0.010, 0.0, 0.0]])
    # Estimate 0 is nearest to truth 1, but taking that pair forces a 10 mm error
    # on the other; the optimal assignment is the identity at 6 + 4 mm.
    estimate = np.array([[0.006, 0.0, 0.0], [0.014, 0.0, 0.0]])
    match = match_sources(estimate, truth)
    assert match.assignment.tolist() == [0, 1]
    np.testing.assert_allclose(match.errors_mm, [6.0, 4.0], atol=1e-9)


@pytest.mark.parametrize("k", [2, 3, 4])
def test_permuting_the_estimate_leaves_the_error_set_unchanged(k: int) -> None:
    rng = np.random.default_rng(4)
    truth = rng.uniform(-0.05, 0.05, (k, 3))
    estimate = truth + rng.normal(0.0, 0.002, (k, 3))
    reference = np.sort(match_sources(estimate, truth).errors_mm)
    for _ in range(5):
        order = rng.permutation(k)
        shuffled = np.sort(match_sources(estimate[order], truth).errors_mm)
        np.testing.assert_allclose(shuffled, reference, atol=1e-9)


def test_scipy_and_brute_force_agree_above_the_enumeration_cutoff() -> None:
    """The K > 7 path is a different solver; it must give the same total cost."""
    from scipy.optimize import linear_sum_assignment

    rng = np.random.default_rng(11)
    k = 8
    truth = rng.uniform(-0.05, 0.05, (k, 3))
    estimate = rng.uniform(-0.05, 0.05, (k, 3))
    match = match_sources(estimate, truth)
    cost = np.linalg.norm(estimate[:, None] - truth[None], axis=-1) * 1e3
    _, columns = linear_sum_assignment(cost)
    assert match.errors_mm.sum() == pytest.approx(cost[np.arange(k), columns].sum())


def test_collapse_is_detected_and_separation_is_reported() -> None:
    truth = np.array([[0.03, 0.0, 0.0], [-0.03, 0.0, 0.0]])
    collapsed = np.array([[0.03, 0.0, 0.0], [0.031, 0.0, 0.0]])
    match = match_sources(collapsed, truth)
    assert match.collapsed
    assert match.min_separation_mm == pytest.approx(1.0)
    assert match.max_error_mm > 50.0

    separated = np.array([[0.03, 0.0, 0.0], [-0.03, 0.0, 0.0]])
    assert not match_sources(separated, truth).collapsed


def test_min_separation_is_in_millimetres() -> None:
    points = np.array([[0.0, 0.0, 0.0], [0.0, 0.012, 0.0], [0.05, 0.0, 0.0]])
    assert min_separation_mm(points) == pytest.approx(12.0)
    assert min_separation_mm(points[:1]) == float("inf")
    assert COLLAPSE_THRESHOLD_MM > 0.0


def test_mismatched_counts_are_refused() -> None:
    with pytest.raises(ValueError, match="same shape"):
        match_sources(np.zeros((2, 3)), np.zeros((3, 3)))
