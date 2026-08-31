# Copyright 2026 NeuroLocate contributors
# SPDX-License-Identifier: Apache-2.0
"""The benchmark's design has to be what it claims: off grid, banded, and fixed.

These tests are cheap and they guard claims that are easy to break silently — a
truth that lands on a grid vertex, a "close" condition that quietly drew a far
pair, a channel subset that shrank because a name went missing, or a seed that
stopped being shared between two conditions that are supposed to be comparable.
"""

from __future__ import annotations

import numpy as np
import pytest
from neurolayout.benchmark import (
    CONDITIONS,
    SEPARATION_REGIMES,
    Condition,
    conditions_by_name,
    depth_mm,
    draw_truth,
    near_initialization,
    offgrid_truth,
    random_initialization,
    source_spacing_mm,
    stable_seed,
)
from neurolayout.channel_subsets import (
    CHANNEL_SUBSETS,
    SUBSET_NAMES,
    subset_indices,
    subset_names,
)
from neurolayout.localize import Containment
from neurolayout.mismatch import (
    INFERENCE_SKULL_CONDUCTIVITY,
    MISMATCH_LEVELS,
    IndependentForward,
    _displaced_electrodes,
)
from neurolayout.montage import CANONICAL_CHANNELS
from neurolayout_shared.openmeeg_model import HeadGeometry, default_artifact_path


@pytest.fixture(scope="module")
def geometry() -> HeadGeometry:
    path = default_artifact_path()
    if not path.exists():
        pytest.skip("no OpenMEEG head-model artifact")
    return HeadGeometry.load(path)


#
# Channel subsets
#


def test_subsets_are_nested_and_present_in_the_canonical_array() -> None:
    assert set(SUBSET_NAMES) == set(CHANNEL_SUBSETS)
    assert subset_indices("all") is None
    cap32 = subset_indices("cap32")
    clinical16 = subset_indices("clinical16")
    assert len(cap32) == 32
    assert len(clinical16) == 16
    assert set(clinical16) < set(cap32)
    for name in ("cap32", "clinical16"):
        assert set(subset_names(name)) == set(CHANNEL_SUBSETS[name])
        # Every name really exists, so a subset can never silently shrink.
        assert len(subset_names(name)) == len(CHANNEL_SUBSETS[name])


def test_clinical16_is_left_right_symmetric() -> None:
    """Mislateralization is the failure mode; an asymmetric array would invite it."""
    names = set(CHANNEL_SUBSETS["clinical16"])
    for channel in names:
        if channel[-1].isdigit():
            index = int(channel[-1])
            mirror = f"{channel[:-1]}{index + 1 if index % 2 else index - 1}"
            assert mirror in names, f"{channel} has no counterpart {mirror}"


def test_unknown_subset_is_refused() -> None:
    with pytest.raises(KeyError, match="unknown channel subset"):
        subset_names("cap64")


def test_indices_select_the_named_channels() -> None:
    for name in ("cap32", "clinical16"):
        indices = subset_indices(name)
        selected = [CANONICAL_CHANNELS[i] for i in indices]
        assert selected == list(subset_names(name))


#
# Condition matrix
#


def test_condition_names_are_unique_and_every_axis_is_populated() -> None:
    table = conditions_by_name()
    assert len(table) == len(CONDITIONS)
    axes = {c.axis for c in CONDITIONS}
    assert {"mismatch", "spine", "separation", "channels", "prior"} <= axes
    for condition in CONDITIONS:
        assert condition.channels in CHANNEL_SUBSETS
        assert condition.mismatch in MISMATCH_LEVELS


def test_a_multi_source_condition_without_a_separation_regime_is_refused() -> None:
    with pytest.raises(ValueError, match="separation regime"):
        Condition("bad", 2)
    with pytest.raises(ValueError, match="unknown mismatch"):
        Condition("bad", 1, mismatch="nonexistent")


def test_comparable_conditions_share_their_truths_and_starting_points() -> None:
    """The mismatch ladder is only a ladder if every rung sees the same problem."""
    table = conditions_by_name()
    ladder = [table[name] for name in
              ("k1-matched", "k1-solver", "k1-skull", "k1-electrodes", "k1-full")]
    seeds = {c.truth_seed(3) for c in ladder}
    assert len(seeds) == 1
    starts = {c.initialization_seed(3, "far") for c in ladder}
    assert len(starts) == 1
    # The 20 dB spine cell shares them too, since it is the same problem.
    assert table["k1-20db"].truth_seed(3) == ladder[0].truth_seed(3)
    # A different K is a different problem and must not share.
    assert table["k2-20db"].truth_seed(3) != ladder[0].truth_seed(3)


def test_conditions_differing_only_in_forward_model_share_the_noise_realization() -> None:
    table = conditions_by_name()
    assert table["k1-solver"].noise(1).seed == table["k1-full"].noise(1).seed
    assert table["k4-20db"].noise(1).seed == table["k4-cap32"].noise(1).seed
    # A different SNR or colour is a different realization.
    assert table["k2-20db"].noise(1).seed != table["k2-10db"].noise(1).seed
    assert table["k2-20db"].noise(1).seed != table["k2-20db-white"].noise(1).seed


def test_stable_seed_does_not_depend_on_the_interpreter() -> None:
    assert stable_seed("truth", 2, "moderate", 0) == stable_seed("truth", 2, "moderate", 0)
    assert stable_seed("a") != stable_seed("b")
    # Pinned so a change to the derivation is a visible test failure rather than a
    # silent reshuffle of every trial in the benchmark.
    assert stable_seed("truth", 1, None, "known", None, 0) == 1283576049157444969


#
# Ground truth
#


def test_truth_is_off_grid_but_close_to_it(geometry: HeadGeometry) -> None:
    spacing = source_spacing_mm(geometry.source_space)
    assert 1.5 < spacing < 3.5, f"unexpected source spacing {spacing:.2f} mm"
    for trial in range(8):
        truth = draw_truth(
            geometry.source_space, geometry.source_normals, 1,
            separation=None, seed=stable_seed("t", trial),
        )
        # Off the grid, but not off the cortex: between 0.3 and 0.7 of the spacing.
        assert 0.25 * spacing < truth.offgrid_mm[0] < 0.75 * spacing
        assert np.isclose(np.linalg.norm(truth.moments_am[0]), 25e-9)


def test_offgrid_truth_is_between_two_neighbours(geometry: HeadGeometry) -> None:
    position, normal, offgrid = offgrid_truth(
        geometry.source_space, geometry.source_normals, 5000, 0.5
    )
    assert offgrid > 0.0
    assert np.isclose(np.linalg.norm(normal), 1.0)
    # The midpoint of an edge is at least a third of the edge from either end.
    distances = np.sort(np.linalg.norm(geometry.source_space - position, axis=1))
    assert distances[0] == pytest.approx(distances[1], rel=0.35)


#: The (K, regime) pairs the matrix actually uses. Narrow bands are only reachable
#: for K = 2: asking four sources to be mutually 12-18 mm apart is a specific
#: tetrahedron that uniform rejection sampling over the cortex will not find, and
#: `draw_truth` correctly refuses rather than widening the band.
DRAWABLE = [(2, "far"), (2, "moderate"), (2, "close"), (4, "spread")]


@pytest.mark.parametrize(("k", "regime"), DRAWABLE)
def test_drawn_sources_land_inside_the_requested_band(
    geometry: HeadGeometry, k, regime
) -> None:
    low, high = SEPARATION_REGIMES[regime]
    for trial in range(4):
        truth = draw_truth(
            geometry.source_space, geometry.source_normals, k,
            separation=regime, seed=stable_seed(regime, k, trial),
        )
        distance = (
            np.linalg.norm(
                truth.positions_m[:, None] - truth.positions_m[None], axis=-1
            )
            * 1e3
        )
        offdiag = distance[~np.eye(k, dtype=bool)]
        # The band is enforced on the grid vertices; the off-grid displacement can
        # move a pair by up to ~1 mm, so the check allows that slack and no more.
        assert offdiag.min() > low - 2.0
        assert offdiag.max() < high + 2.0


def test_draw_truth_is_deterministic_and_seed_dependent(geometry: HeadGeometry) -> None:
    first = draw_truth(geometry.source_space, geometry.source_normals, 2,
                       separation="moderate", seed=99)
    again = draw_truth(geometry.source_space, geometry.source_normals, 2,
                       separation="moderate", seed=99)
    other = draw_truth(geometry.source_space, geometry.source_normals, 2,
                       separation="moderate", seed=100)
    np.testing.assert_array_equal(first.positions_m, again.positions_m)
    assert not np.allclose(first.positions_m, other.positions_m)


def test_an_impossible_separation_band_fails_loudly(geometry: HeadGeometry) -> None:
    """Silently running a different regime would be worse than crashing."""
    with pytest.raises(ValueError, match="no K=2 configuration"):
        draw_truth(
            geometry.source_space, geometry.source_normals, 2,
            separation="close", seed=0, max_attempts=1,
        )
    # And a band that is geometrically unreachable at this K fails too, rather
    # than quietly returning a wider configuration.
    with pytest.raises(ValueError, match="no K=4 configuration"):
        draw_truth(
            geometry.source_space, geometry.source_normals, 4,
            separation="close", seed=0, max_attempts=2000,
        )


def test_every_condition_in_the_matrix_can_actually_be_drawn(
    geometry: HeadGeometry,
) -> None:
    """A regime that cannot be sampled at this K must not be in the matrix."""
    for condition in CONDITIONS:
        draw_truth(
            geometry.source_space, geometry.source_normals, condition.n_sources,
            separation=condition.separation, seed=condition.truth_seed(0),
        )


#
# Initializations
#


def test_near_initialization_is_the_stated_distance_and_stays_inside(
    geometry: HeadGeometry,
) -> None:
    containment = Containment.from_points(geometry.vertices[0])
    truth = draw_truth(geometry.source_space, geometry.source_normals, 4,
                       separation="spread", seed=7)
    starts = near_initialization(geometry.vertices[0], truth.positions_m)
    assert starts.shape == (4, 3)
    offsets = np.linalg.norm(starts - truth.positions_m, axis=1)
    np.testing.assert_allclose(offsets, 0.024, rtol=1e-9)
    # Displacing inward rather than along a fixed vector keeps every start inside
    # the head, which a fixed offset would not do for superficial sources.
    assert all(containment.contains(p) for p in starts)


def test_random_initialization_is_far_from_every_truth_and_not_collapsed(
    geometry: HeadGeometry,
) -> None:
    containment = Containment.from_points(geometry.vertices[0])
    truth = draw_truth(geometry.source_space, geometry.source_normals, 4,
                       separation="spread", seed=11)
    rng = np.random.default_rng(0)
    starts = random_initialization(containment, truth.positions_m, rng)
    assert starts.shape == (4, 3)
    assert all(containment.contains(p) for p in starts)
    distance = np.linalg.norm(truth.positions_m[:, None] - starts[None], axis=-1)
    assert distance.min() > 0.03
    mutual = np.linalg.norm(starts[:, None] - starts[None], axis=-1)
    assert mutual[~np.eye(4, dtype=bool)].min() > 0.02


def test_depth_is_measured_from_the_inner_skull(geometry: HeadGeometry) -> None:
    centre = geometry.vertices[0].mean(axis=0)
    surface = geometry.vertices[0][0]
    assert depth_mm(geometry.vertices[0], surface) == pytest.approx(0.0, abs=1e-9)
    assert depth_mm(geometry.vertices[0], centre) > 30.0


#
# Mismatch specifications
#


def test_the_mismatch_ladder_is_graded() -> None:
    assert MISMATCH_LEVELS["matched"].is_matched
    for name in ("solver", "skull", "electrodes", "full"):
        assert not MISMATCH_LEVELS[name].is_matched
        assert MISMATCH_LEVELS[name].ico == 4
    assert MISMATCH_LEVELS["solver"].skull_conductivity == INFERENCE_SKULL_CONDUCTIVITY
    assert MISMATCH_LEVELS["solver"].electrode_error_mm == 0.0
    assert MISMATCH_LEVELS["skull"].skull_conductivity != INFERENCE_SKULL_CONDUCTIVITY
    assert MISMATCH_LEVELS["skull"].electrode_error_mm == 0.0
    assert MISMATCH_LEVELS["electrodes"].electrode_error_mm > 0.0
    assert MISMATCH_LEVELS["electrodes"].skull_conductivity == INFERENCE_SKULL_CONDUCTIVITY
    assert MISMATCH_LEVELS["full"].skull_conductivity != INFERENCE_SKULL_CONDUCTIVITY
    assert MISMATCH_LEVELS["full"].electrode_error_mm > 0.0
    assert MISMATCH_LEVELS["skull"].skull_ratio > 50.0


def test_matched_has_no_independent_forward(geometry: HeadGeometry) -> None:
    """Serving the crime from this class would disguise it as a control."""
    with pytest.raises(ValueError, match="no independent forward"):
        IndependentForward(
            MISMATCH_LEVELS["matched"],
            geometry.channel_names,
            geometry.sensor_xyz,
            geometry.vertices[2],
        )


def test_electrode_displacement_hits_its_rms_and_stays_on_the_scalp(
    geometry: HeadGeometry,
) -> None:
    for target in (3.0, 5.0, 8.0):
        displaced = _displaced_electrodes(
            geometry.sensor_xyz, geometry.vertices[2], target, seed=7
        )
        offsets = displaced - geometry.sensor_xyz
        realized = float(np.sqrt(np.mean(np.sum(offsets**2, axis=1))) * 1e3)
        assert realized == pytest.approx(target, rel=0.25)
        # Tangential, so the radius from the scalp centroid barely changes.
        centre = geometry.vertices[2].mean(axis=0)
        before = np.linalg.norm(geometry.sensor_xyz - centre, axis=1)
        after = np.linalg.norm(displaced - centre, axis=1)
        assert np.abs(after - before).max() * 1e3 < 0.5 * target


def test_electrode_displacement_is_reproducible(geometry: HeadGeometry) -> None:
    kwargs = dict(rms_mm=5.0)
    first = _displaced_electrodes(geometry.sensor_xyz, geometry.vertices[2], seed=7, **kwargs)
    again = _displaced_electrodes(geometry.sensor_xyz, geometry.vertices[2], seed=7, **kwargs)
    other = _displaced_electrodes(geometry.sensor_xyz, geometry.vertices[2], seed=8, **kwargs)
    np.testing.assert_array_equal(first, again)
    assert not np.allclose(first, other)
