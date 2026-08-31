# Copyright 2026 NeuroLocate contributors
# SPDX-License-Identifier: Apache-2.0
r"""The batched headfield mode: same physics, same derivative, one solver call.

``mode="localize_batch"`` exists for throughput, and a throughput change that
alters an answer is a bug rather than an optimization. So it is checked against
the two constructions that already worked, in value *and* in gradient:

**The loop.** ``B`` separate ``localize`` calls, one per batch entry. This is what
the batched mode replaces, and it is the definition of the right answer.

**The flattened set.** One ``localize`` call over all ``B*K`` positions treated as
a single source set, with the off-block time courses zeroed so each output batch
entry sees only its own sources. Mathematically identical, and quadratic in ``B``
in the array it has to carry — which is why it is a test rather than the
implementation.

The sphere backend is used throughout: the claim under test is about the batching
algebra and the derivative bookkeeping, not about the BEM, and the sphere solver
makes the test run anywhere in milliseconds.
"""

from __future__ import annotations

import numpy as np
import pytest
from neurolayout_shared.source_model import (
    backward,
    backward_batched,
    forward,
    forward_batched,
)
from neurolayout_shared.sphere_model import SphereHead, sphere_source_gain

BATCH, SOURCES, TIMES, CHANNELS = 3, 2, 5, 16


@pytest.fixture(name="gain_fn")
def fixture_gain_fn():
    """The analytic sphere gain, as the backend callable."""
    head = SphereHead(radius=0.09, sigma=0.33, n_channels=CHANNELS)

    def gain(positions: np.ndarray) -> np.ndarray:
        return sphere_source_gain(head, positions, reference=True)

    return gain


@pytest.fixture(name="problem")
def fixture_problem():
    """A deterministic batch of independent source sets."""
    rng = np.random.default_rng(11)
    positions = rng.uniform(-0.04, 0.04, (BATCH, SOURCES, 3))
    timecourses = rng.normal(0.0, 1e-8, (BATCH, SOURCES, 3, TIMES))
    cotangent = rng.normal(0.0, 1.0, (BATCH, CHANNELS, TIMES))
    return positions, timecourses, cotangent


def _loop_forward(gain_fn, positions, timecourses):
    """``B`` single-set forwards, stacked."""
    return np.concatenate(
        [
            forward(gain_fn, positions[b], timecourses[b : b + 1])["eeg"]
            for b in range(positions.shape[0])
        ],
        axis=0,
    )


def _flattened(positions, timecourses):
    """The batch as one ``B*K``-source set with block-diagonal time courses."""
    batch, n_sources = positions.shape[0], positions.shape[1]
    flat_positions = positions.reshape(-1, 3)
    flat = np.zeros((batch, batch * n_sources, 3, timecourses.shape[3]))
    for b in range(batch):
        flat[b, b * n_sources : (b + 1) * n_sources] = timecourses[b]
    return flat_positions, flat


def test_batched_forward_matches_the_loop(gain_fn, problem) -> None:
    """One call over B*K dipoles gives what B calls over K dipoles give."""
    positions, timecourses, _ = problem
    batched = forward_batched(gain_fn, positions, timecourses)["eeg"]
    looped = _loop_forward(gain_fn, positions, timecourses)
    assert batched.shape == (BATCH, CHANNELS, TIMES)
    np.testing.assert_allclose(batched, looped, rtol=0, atol=1e-30)


def test_batched_forward_matches_the_flattened_set(gain_fn, problem) -> None:
    """And what the quadratic single-set construction gives."""
    positions, timecourses, _ = problem
    flat_positions, flat_timecourses = _flattened(positions, timecourses)
    reference = forward(gain_fn, flat_positions, flat_timecourses)["eeg"]
    batched = forward_batched(gain_fn, positions, timecourses)["eeg"]
    np.testing.assert_allclose(batched, reference, rtol=0, atol=1e-30)


def test_batched_position_cotangent_matches_the_loop(gain_fn, problem) -> None:
    """The expensive half: 6*B*K perturbed dipoles in one call, same numbers."""
    positions, timecourses, cotangent = problem
    cache = forward_batched(gain_fn, positions, timecourses)
    grads = backward_batched(cache, gain_fn, positions, timecourses, cotangent)

    for b in range(BATCH):
        single = forward(gain_fn, positions[b], timecourses[b : b + 1])
        expected = backward(
            single,
            gain_fn,
            positions[b],
            timecourses[b : b + 1],
            cotangent[b : b + 1],
        )
        np.testing.assert_allclose(
            grads["source_positions_batch"][b], expected["source_positions"], rtol=1e-12
        )
        np.testing.assert_allclose(
            grads["source_timecourses"][b],
            expected["source_timecourses"][0],
            rtol=1e-12,
        )


def test_batched_gradient_agrees_with_finite_differences(gain_fn, problem) -> None:
    """An end-to-end check that does not reuse the VJP being tested.

    ``w^T J e_i`` against a central difference of ``w^T f`` in the same direction,
    for one coordinate of every source in the batch.
    """
    positions, timecourses, cotangent = problem
    cache = forward_batched(gain_fn, positions, timecourses)
    grads = backward_batched(cache, gain_fn, positions, timecourses, cotangent)

    step = 1e-6
    for b in range(BATCH):
        for k in range(SOURCES):
            for axis in range(3):
                shifted = positions.copy()
                shifted[b, k, axis] += step
                high = forward_batched(gain_fn, shifted, timecourses)["eeg"]
                shifted[b, k, axis] -= 2.0 * step
                low = forward_batched(gain_fn, shifted, timecourses)["eeg"]
                numeric = float(np.sum(cotangent * (high - low)) / (2.0 * step))
                analytic = float(grads["source_positions_batch"][b, k, axis])
                assert abs(numeric - analytic) <= 1e-5 * max(abs(numeric), 1.0)


def test_batched_forward_rejects_a_mismatched_shape(gain_fn, problem) -> None:
    """A [K, 3] position array in batch mode is an error, not a broadcast."""
    positions, timecourses, _ = problem
    with pytest.raises(ValueError, match=r"\[B, K, 3\]"):
        forward_batched(gain_fn, positions[0], timecourses)
    with pytest.raises(ValueError, match="timecourses must be"):
        forward_batched(gain_fn, positions, timecourses[:, :1])
