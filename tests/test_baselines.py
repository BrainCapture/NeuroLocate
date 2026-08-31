# Copyright 2026 NeuroLocate contributors
# SPDX-License-Identifier: Apache-2.0
r"""The classical baselines, and the claim that they run on identical physics.

The comparison in ``docs/BENCHMARK.md`` is only meaningful if the
baselines really are given NeuroLocate's forward operator. That is done by
substituting OpenMEEG's gain into MNE's ``Forward`` bookkeeping, which is exactly
the kind of surgery that can silently half-work — so the substitution is checked
here rather than trusted.

Also here: the independent generator actually differs from the inference model
(a "mismatch" that produced identical data would be the crime with extra steps),
and the peak read-out cannot report ``K`` detections of one blob.

Most of these need MNE and a downloaded fsaverage and take a few seconds; the ones
that build a BEM are kept to ico3 for that reason, with the shipped ico4
specification checked separately in ``tests/test_benchmark_design.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from neurolayout.baselines import (
    DEFAULT_DEPTH,
    DEFAULT_LOOSE,
    DipoleDictionary,
    MneInverse,
    peak_locations,
)
from neurolayout.benchmark import draw_truth
from neurolayout.localize import LocalizeConfig, make_waveform
from neurolayout.matching import match_sources
from neurolayout.mismatch import MISMATCH_LEVELS, IndependentForward, MismatchSpec
from neurolayout.montage import CANONICAL_CHANNELS
from neurolayout_shared.openmeeg_model import (
    HeadGeometry,
    OpenMEEGForward,
    default_artifact_path,
    load_sensor_operator,
)
from neurolayout_shared.source_model import average_reference_operator

#: Keeping the dictionary to every 40th candidate makes it ~512 locations and a
#: few seconds to build. It coarsens the resolution floor by the same factor, so
#: these tests assert "found the right region", never a millimetre figure.
TEST_STRIDE = 40


@pytest.fixture(scope="module")
def geometry() -> HeadGeometry:
    path = default_artifact_path()
    if not path.exists():
        pytest.skip("no OpenMEEG head-model artifact")
    return HeadGeometry.load(path)


@pytest.fixture(scope="module")
def unreferenced(geometry: HeadGeometry) -> OpenMEEGForward:
    path = default_artifact_path()
    return OpenMEEGForward(
        geometry, load_sensor_operator(path, geometry), reference=False
    )


@pytest.fixture(scope="module")
def dictionary(unreferenced: OpenMEEGForward) -> DipoleDictionary:
    return DipoleDictionary(unreferenced, stride=TEST_STRIDE)


def _observation(
    unreferenced: OpenMEEGForward,
    geometry: HeadGeometry,
    positions: np.ndarray,
    moments: np.ndarray,
    waveform: np.ndarray,
) -> np.ndarray:
    """``[C, T]`` clean, referenced EEG for given sources — the generator's job."""
    reference = average_reference_operator(geometry.n_channels)
    gain = unreferenced.gain(positions)
    return np.einsum("cd,dkj,kj,t->ct", reference, gain, moments, waveform)


#
# Reading K locations out of a map
#


def test_peak_locations_excludes_a_neighbourhood_around_each_peak() -> None:
    positions = np.stack([np.linspace(0.0, 0.1, 21), np.zeros(21), np.zeros(21)], axis=1)
    # One broad blob at index 5 and a smaller one at 15, 50 mm apart.
    power = np.exp(-0.5 * ((np.arange(21) - 5) / 1.5) ** 2)
    power += 0.5 * np.exp(-0.5 * ((np.arange(21) - 15) / 1.5) ** 2)
    found, n_found = peak_locations(power, positions, 2, exclusion_mm=15.0)
    assert n_found == 2
    np.testing.assert_allclose(found[0], positions[5])
    np.testing.assert_allclose(found[1], positions[15])


def test_peak_locations_will_not_split_one_blob_into_two_detections() -> None:
    """Without the exclusion, adjacent vertices of one blob score as K hits."""
    positions = np.stack([np.linspace(0.0, 0.02, 21), np.zeros(21), np.zeros(21)], axis=1)
    power = np.exp(-0.5 * ((np.arange(21) - 10) / 2.0) ** 2)
    _, with_exclusion = peak_locations(power, positions, 3, exclusion_mm=15.0)
    _, without = peak_locations(power, positions, 3, exclusion_mm=0.0)
    assert with_exclusion == 1
    assert without == 3


def test_peak_locations_pads_when_it_runs_out() -> None:
    positions = np.eye(3)
    power = np.array([1.0, 0.0, 0.0])
    found, n_found = peak_locations(power, positions, 3, exclusion_mm=1e6)
    assert n_found == 1
    assert np.isfinite(found[0]).all()
    assert np.isnan(found[1:]).all()


#
# The discrete scan
#


def test_scan_finds_the_region_of_a_single_source(
    dictionary: DipoleDictionary, unreferenced: OpenMEEGForward, geometry: HeadGeometry
) -> None:
    config = LocalizeConfig(n_times=16)
    waveform = make_waveform(config)
    index = dictionary.n_locations // 3
    position = dictionary.positions[index : index + 1]
    moment = 25e-9 * geometry.source_normals[index * TEST_STRIDE][None]
    observed = _observation(unreferenced, geometry, position, moment, waveform)

    result = dictionary.scan(observed, waveform, 1)
    assert result.n_found == 1
    assert result.residual_fraction < 1e-12  # the source is a dictionary entry
    np.testing.assert_allclose(result.positions_m[0], position[0], atol=1e-12)


def test_scan_extends_to_two_sources_by_matching_pursuit(
    dictionary: DipoleDictionary, unreferenced: OpenMEEGForward, geometry: HeadGeometry
) -> None:
    """Both sources found, moments refit jointly, residual driven down."""
    config = LocalizeConfig(n_times=16)
    waveform = make_waveform(config)
    # Two well-separated dictionary entries, one per hemisphere.
    left = int(np.argmin(dictionary.positions[:, 0]))
    right = int(np.argmax(dictionary.positions[:, 0]))
    positions = dictionary.positions[[left, right]]
    moments = 25e-9 * np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    observed = _observation(unreferenced, geometry, positions, moments, waveform)

    single = dictionary.scan(observed, waveform, 1)
    both = dictionary.scan(observed, waveform, 2)
    assert both.n_found == 2
    # Explaining two sources with two dipoles must beat explaining them with one.
    assert both.residual_fraction < single.residual_fraction
    assert both.residual_fraction < 1e-12
    match = match_sources(both.positions_m, positions)
    assert match.max_error_mm < 1e-6
    assert not match.collapsed


def test_scan_respects_the_exclusion_radius(
    dictionary: DipoleDictionary, unreferenced: OpenMEEGForward, geometry: HeadGeometry
) -> None:
    """Greedy pursuit must not spend two sources on one topography."""
    config = LocalizeConfig(n_times=16)
    waveform = make_waveform(config)
    index = dictionary.n_locations // 2
    position = dictionary.positions[index : index + 1]
    moment = 25e-9 * geometry.source_normals[index * TEST_STRIDE][None]
    observed = _observation(unreferenced, geometry, position, moment, waveform)
    result = dictionary.scan(observed, waveform, 2, exclusion_mm=30.0)
    separation = np.linalg.norm(result.positions_m[0] - result.positions_m[1]) * 1e3
    assert separation >= 30.0


def test_scan_honours_a_channel_subset(
    dictionary: DipoleDictionary, unreferenced: OpenMEEGForward, geometry: HeadGeometry
) -> None:
    from neurolayout.channel_subsets import subset_indices

    config = LocalizeConfig(n_times=16)
    waveform = make_waveform(config)
    channels = subset_indices("clinical16")
    index = dictionary.n_locations // 4
    position = dictionary.positions[index : index + 1]
    moment = 25e-9 * geometry.source_normals[index * TEST_STRIDE][None]

    # A 16-channel observation carries a 16-channel reference, not a restriction
    # of the 64-channel one -- which is what dictionary.gain(channels) builds.
    gain = dictionary.gain(channels)
    observed = np.einsum("ckj,kj,t->ct", gain[:, index : index + 1, :], moment, waveform)
    assert observed.shape == (len(channels), config.n_times)
    result = dictionary.scan(observed, waveform, 1, channels=channels)
    np.testing.assert_allclose(result.positions_m[0], position[0], atol=1e-12)


def test_dictionary_requires_an_unreferenced_forward(geometry: HeadGeometry) -> None:
    """Referenced-then-subset would give a 16-channel run the wrong reference."""
    referenced = OpenMEEGForward(
        geometry, load_sensor_operator(default_artifact_path(), geometry), reference=True
    )
    with pytest.raises(ValueError, match="unreferenced"):
        DipoleDictionary(referenced, stride=TEST_STRIDE)


def test_dictionary_cache_round_trips_and_refuses_a_foreign_geometry(
    unreferenced: OpenMEEGForward, tmp_path
) -> None:
    cache = tmp_path / "dictionary.npz"
    # Written by hand rather than by building the real thing: the full-stride
    # dictionary is two minutes of solver time, and what is under test here is the
    # cache contract, not the gain.
    np.savez(
        cache,
        gain_unreferenced=np.zeros((unreferenced.geometry.n_channels,
                                    unreferenced.geometry.n_sources, 3)),
        fingerprint=np.array(unreferenced.geometry.fingerprint()),
    )
    loaded = DipoleDictionary(unreferenced, cache=cache)
    assert loaded.n_locations == unreferenced.geometry.n_sources
    assert loaded.from_cache

    np.savez(
        cache,
        gain_unreferenced=np.zeros((unreferenced.geometry.n_channels,
                                    unreferenced.geometry.n_sources, 3)),
        fingerprint=np.array("not-this-head-model"),
    )
    with pytest.raises(ValueError, match="another head model"):
        DipoleDictionary(unreferenced, cache=cache)


#
# The MNE estimators, and the gain substitution they depend on
#


#: Where the full-stride dictionary is cached. The MNE estimators live on the ico5
#: source space MNE itself builds, so they cannot be given the strided dictionary
#: the scan tests use -- the columns would not line up. Building it costs ~2 min of
#: solver time the first time and nothing afterwards, which is the price of
#: actually checking that the baselines run on NeuroLocate's physics rather than
#: MNE's own.
DICTIONARY_CACHE = (
    Path(__file__).resolve().parents[1] / "results" / "dipole_dictionary.npz"
)


@pytest.fixture(scope="module")
def full_dictionary(unreferenced: OpenMEEGForward) -> DipoleDictionary:
    return DipoleDictionary(unreferenced, cache=DICTIONARY_CACHE)


@pytest.fixture(scope="module")
def mne_inverse(full_dictionary: DipoleDictionary):
    pytest.importorskip("mne")
    return MneInverse(full_dictionary, tuple(CANONICAL_CHANNELS))


def test_the_substituted_forward_carries_openmeeg_s_gain(mne_inverse) -> None:
    """If this fails, the baselines are quietly running MNE's own BEM."""
    forward = mne_inverse.forward_for(None)
    expected = mne_inverse.dictionary.gain(None)
    actual = np.asarray(forward["sol"]["data"]).reshape(expected.shape)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
    # `convert_forward_solution` rebuilds `sol` from `_orig_sol`, so that copy has
    # to carry the substitution too or the inverse solvers would undo it.
    np.testing.assert_allclose(
        np.asarray(forward["_orig_sol"]).reshape(expected.shape), expected, rtol=0, atol=0
    )


def test_the_substituted_forward_survives_mne_s_orientation_conversion(mne_inverse) -> None:
    import mne

    forward = mne_inverse.forward_for(None)
    expected = mne_inverse.dictionary.gain(None)
    converted = mne.convert_forward_solution(
        forward, surf_ori=True, force_fixed=False, use_cps=True, copy=True, verbose="ERROR"
    )
    got = np.asarray(converted["sol"]["data"]).reshape(expected.shape)
    # The conversion rotates each source's three columns into a surface-aligned
    # frame, so the columns change but the topography subspace must not.
    for index in (0, expected.shape[1] // 2, expected.shape[1] - 1):
        before = np.linalg.svd(expected[:, index, :], compute_uv=False)
        after = np.linalg.svd(got[:, index, :], compute_uv=False)
        np.testing.assert_allclose(after, before, rtol=1e-10)


def test_a_channel_subset_forward_has_the_right_shape_and_reference(mne_inverse) -> None:
    from neurolayout.channel_subsets import subset_indices

    channels = subset_indices("cap32")
    forward = mne_inverse.forward_for(channels)
    gain = mne_inverse.dictionary.gain(channels)
    assert np.asarray(forward["sol"]["data"]).shape == (len(channels), 3 * gain.shape[1])
    assert list(forward["info"]["ch_names"]) == [CANONICAL_CHANNELS[i] for i in channels]
    # Referenced over the subset: every topography sums to zero across 32 channels.
    assert np.abs(gain.sum(axis=0)).max() < 1e-12 * np.abs(gain).max()


@pytest.mark.parametrize("method", ["irmxne", "dspm"])
def test_the_mne_estimators_run_and_return_scorable_output(
    mne_inverse, unreferenced, geometry, method
) -> None:
    """A smoke test with a real single source: shapes, finiteness, and a region."""
    config = LocalizeConfig(n_times=16)
    waveform = make_waveform(config)
    truth = draw_truth(
        geometry.source_space, geometry.source_normals, 1, separation=None, seed=4242
    )
    observed = _observation(
        unreferenced, geometry, truth.positions_m, truth.moments_am, waveform
    )
    runner = getattr(mne_inverse, method)
    result = runner(observed, 1, noise_rms_v=0.0, alpha=30.0) if method == "irmxne" \
        else runner(observed, 1, noise_rms_v=0.0)

    assert result.name == method
    assert result.positions_m.shape == (1, 3)
    assert result.seconds > 0.0
    assert result.detail["loose"] == DEFAULT_LOOSE
    assert result.detail["depth"] == DEFAULT_DEPTH
    if result.n_found == 0:  # a documented failure mode, not an error
        assert np.isnan(result.positions_m).all()
        return
    assert np.isfinite(result.positions_m).all()
    # The estimate must at least be a cortical location on the right side of the
    # head. Accuracy is the benchmark's business, not this test's.
    error_mm = float(np.linalg.norm(result.positions_m[0] - truth.positions_m[0]) * 1e3)
    assert error_mm < 80.0, f"{method} landed {error_mm:.0f} mm away"


def test_irmxne_reports_a_solver_failure_instead_of_raising(mne_inverse) -> None:
    """A failed baseline is a datum; crashing the sweep is not."""
    config = LocalizeConfig(n_times=16)
    observed = np.zeros((64, config.n_times))
    result = mne_inverse.irmxne(observed, 2, noise_rms_v=1e-9, alpha=99.0)
    assert result.name == "irmxne"
    assert result.positions_m.shape == (2, 3)
    assert result.n_found <= 2


#
# The mismatch really is a mismatch
#


def test_the_independent_generator_disagrees_with_the_inference_model(
    geometry: HeadGeometry, unreferenced: OpenMEEGForward
) -> None:
    """A generator that matched to round-off would be the crime with extra steps."""
    pytest.importorskip("mne")
    reference = average_reference_operator(geometry.n_channels)
    # ico3 rather than the shipped ico4: same two implementations, a tenth of the
    # build time. The shipped specification is checked in test_benchmark_design.
    spec = MismatchSpec(name="test-solver", ico=3, description="unit-test generator")
    generator = IndependentForward(
        spec, geometry.channel_names, geometry.sensor_xyz, geometry.vertices[2]
    )
    indices = list(range(500, 20000, 4000))
    positions = geometry.source_space[indices]
    normals = geometry.source_normals[indices]

    theirs = generator.gain(positions)
    ours = np.einsum("cd,dpj->cpj", reference, unreferenced.gain(positions))
    assert theirs.shape == ours.shape

    for index in range(len(indices)):
        a = np.einsum("cj,j->c", ours[:, index, :], normals[index])
        b = np.einsum("cj,j->c", theirs[:, index, :], normals[index])
        cosine = float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
        # The same physics, so the topographies must be close...
        assert cosine > 0.99, f"source {indices[index]}: cosine {cosine:.4f}"
        # ...but two different formulations at different resolutions, so not equal
        # even after the best global rescaling a free moment could absorb.
        scale = float(a @ b / (b @ b))
        residual = float(np.linalg.norm(a - scale * b) / np.linalg.norm(a))
        assert residual > 1e-3, (
            f"source {indices[index]}: generator agrees to {residual:.2e} after "
            "rescaling — that is not a mismatch"
        )


def test_the_generator_reports_what_it_did(geometry: HeadGeometry) -> None:
    pytest.importorskip("mne")
    spec = MISMATCH_LEVELS["electrodes"]
    generator = IndependentForward(
        MismatchSpec(
            name=spec.name, ico=3, electrode_error_mm=spec.electrode_error_mm,
            description=spec.description,
        ),
        geometry.channel_names,
        geometry.sensor_xyz,
        geometry.vertices[2],
    )
    provenance = generator.provenance()
    assert provenance["realized_electrode_rms_mm"] == pytest.approx(5.0, rel=0.25)
    assert "openmeeg" in provenance["inference"]
    assert "mne" in provenance["generator"]
    assert provenance["coord_frame"] == "mne-head"
    # The generator's electrodes are not the ones inference assumes.
    assert not np.allclose(generator.sensor_xyz, generator.canonical_sensor_xyz)
