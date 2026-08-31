# Copyright 2026 NeuroLocate contributors
# SPDX-License-Identifier: Apache-2.0
"""Gates C and D for source localization.

Gate C is composition: a JAX scalar loss whose evaluation crosses into a
compiled BEM solver, differentiated with ``jax.value_and_grad``, matching
central differences of the composed forward.

Gate D is the thing the project claims: start the source in the wrong place, and
the optimizer moves it to the right one. Ground truth is exact here, so the
assertion is in millimetres rather than in "the loss went down".

Both backends are exercised. The sphere path takes no artifact and no OpenMEEG,
which is what makes it a real fallback.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest
from neurolayout.clients import open_component
from neurolayout.localize import (
    Containment,
    LocalizeConfig,
    Separation,
    SourceParams,
    least_squares_moment,
    localization_error_mm,
    make_loss,
    make_waveform,
    predict_eeg,
    run_localization,
    sensor_positions,
    simulate,
)
from neurolayout.noise import NoiseSpec
from neurolayout_shared.openmeeg_model import HeadGeometry, default_artifact_path


@pytest.fixture(scope="module")
def headfield():
    with open_component("headfield", "local") as tesseract:
        yield tesseract


@pytest.fixture(scope="module")
def geometry() -> HeadGeometry:
    path = default_artifact_path()
    if not path.exists():
        pytest.skip("no OpenMEEG head-model artifact")
    return HeadGeometry.load(path)


#: A sphere-backend problem that needs nothing on disk.
SPHERE_CONFIG = LocalizeConfig(backend="sphere", n_times=4, n_channels=32, steps=150)
SPHERE_CONTAINMENT = Containment(centre_cm=np.zeros(3), semi_axes_cm=np.full(3, 7.0))
SPHERE_TRUTH = SourceParams.from_si(
    np.array([0.02, -0.01, 0.03]), 25e-9 * np.array([0.0, 0.0, 1.0])
)


def test_containment_is_zero_inside_and_grows_outside() -> None:
    """A guardrail that is silently active at the answer would bias the result."""
    containment = Containment(centre_cm=np.zeros(3), semi_axes_cm=np.array([7.0, 8.0, 6.0]))
    assert float(containment.penalty(np.array([[0.0, 0.0, 0.0]]))) == 0.0
    assert float(containment.penalty(np.array([[6.0, 0.0, 0.0]]))) == 0.0
    outside = float(containment.penalty(np.array([[14.0, 0.0, 0.0]])))
    further = float(containment.penalty(np.array([[21.0, 0.0, 0.0]])))
    assert 0.0 < outside < further


def test_containment_fits_inside_the_brain(geometry: HeadGeometry) -> None:
    """The fitted ellipsoid must contain the cortical source space it guards."""
    containment = Containment.from_points(geometry.vertices[0])
    inside = [containment.contains(point) for point in geometry.source_space[::400]]
    assert np.mean(inside) > 0.9


def test_least_squares_moment_is_exact_at_the_true_position(headfield) -> None:
    """Linearity again: with the position right, the moment needs no optimizer."""
    observation = simulate(headfield, SPHERE_TRUTH, SPHERE_CONFIG)
    recovered = least_squares_moment(
        headfield, np.asarray(SPHERE_TRUTH.position_cm), observation, SPHERE_CONFIG
    )
    np.testing.assert_allclose(
        np.asarray(recovered.moment_nam),
        np.asarray(SPHERE_TRUTH.moment_nam),
        rtol=1e-6,
        atol=1e-9,
    )


@pytest.mark.parametrize("kind", ["white", "correlated"])
def test_noise_is_applied_at_the_requested_snr(headfield, kind) -> None:
    clean = simulate(headfield, SPHERE_TRUTH, SPHERE_CONFIG)
    noisy_config = LocalizeConfig(
        backend="sphere",
        n_times=4,
        n_channels=32,
        noise=NoiseSpec(snr_db=20.0, kind=kind, seed=3),
    )
    noisy = simulate(headfield, SPHERE_TRUTH, noisy_config)
    residual = np.asarray(noisy.eeg) - np.asarray(clean.eeg)
    measured_db = 20.0 * np.log10(clean.clean_rms / np.sqrt(np.mean(residual**2)))
    assert measured_db == pytest.approx(20.0, abs=1e-6)
    assert noisy.provenance["noise"]["kind"] == kind
    # The reference operator is applied to the noise as well as the signal.
    assert np.abs(residual.mean(axis=1)).max() < 1e-18


@pytest.mark.parametrize("backend", ["sphere", "openmeeg"])
def test_composed_gradient_matches_finite_differences(headfield, geometry, backend) -> None:
    """Gate C: ``jax.grad`` through the served component, versus differences of it."""
    if backend == "sphere":
        config, containment, truth = SPHERE_CONFIG, SPHERE_CONTAINMENT, SPHERE_TRUTH
    else:
        config = LocalizeConfig(backend="openmeeg", n_times=4)
        containment = Containment.from_points(geometry.vertices[0])
        truth = SourceParams.from_si(
            geometry.source_space[5000], 25e-9 * geometry.source_normals[5000]
        )

    observation = simulate(headfield, truth, config)
    loss_fn, _ = make_loss(headfield, observation, config, containment)
    start = SourceParams(
        position_cm=truth.position_cm + np.array([[1.4, -0.8, 0.6]]),
        moment_nam=truth.moment_nam * 0.6 + 3.0,
    )

    value, grads = jax.value_and_grad(loss_fn)(start)
    assert np.isfinite(float(value))
    assert np.isfinite(np.asarray(grads.position_cm)).all()
    assert np.abs(np.asarray(grads.position_cm)).max() > 0
    assert np.abs(np.asarray(grads.moment_nam)).max() > 0

    step = 1e-4
    for name in ("position_cm", "moment_nam"):
        numeric = np.zeros((1, 3))
        for axis in range(3):
            delta = np.zeros((1, 3))
            delta[0, axis] = step
            plus = SourceParams(
                **{
                    key: (getattr(start, key) + delta if key == name else getattr(start, key))
                    for key in ("position_cm", "moment_nam")
                }
            )
            minus = SourceParams(
                **{
                    key: (getattr(start, key) - delta if key == name else getattr(start, key))
                    for key in ("position_cm", "moment_nam")
                }
            )
            numeric[0, axis] = (float(loss_fn(plus)) - float(loss_fn(minus))) / (2 * step)
        analytic = np.asarray(getattr(grads, name))
        error = np.abs(analytic - numeric).max() / np.abs(numeric).max()
        assert error < 1e-5, f"{backend}/{name} composed gradient error {error:.2e}"


def test_sphere_backend_recovers_a_known_source(headfield) -> None:
    """Gate D on the fallback solver — no artifact, no OpenMEEG, no excuses."""
    observation = simulate(headfield, SPHERE_TRUTH, SPHERE_CONFIG)
    start = least_squares_moment(
        headfield,
        np.asarray(SPHERE_TRUTH.position_cm) + np.array([[1.2, 0.8, -0.6]]),
        observation,
        SPHERE_CONFIG,
    )
    result = run_localization(
        headfield, observation, start, SPHERE_CONFIG, SPHERE_CONTAINMENT, record_every=25
    )
    assert result["initial_error_mm"] > 10.0
    assert result["final_error_mm"] < 1.0
    assert result["final_loss"] < result["initial_loss"] * 1e-3
    assert result["final_containment"] == 0.0
    # The trajectory has to actually be a trajectory.
    positions = np.asarray(result["history"]["position_cm"])
    assert positions.shape[0] > 3
    assert np.linalg.norm(positions[-1] - positions[0]) > 1.0


def test_openmeeg_backend_recovers_a_known_source(headfield, geometry) -> None:
    """Gate D through the C++ BEM: the headline claim, in millimetres."""
    config = LocalizeConfig(backend="openmeeg", n_times=4, steps=80)
    containment = Containment.from_points(geometry.vertices[0])
    truth = SourceParams.from_si(
        geometry.source_space[5000], 25e-9 * geometry.source_normals[5000]
    )
    observation = simulate(headfield, truth, config)
    start = least_squares_moment(
        headfield,
        np.asarray(truth.position_cm) + np.array([[1.5, -0.8, 0.4]]),
        observation,
        config,
    )
    result = run_localization(
        headfield, observation, start, config, containment, record_every=40
    )
    assert result["initial_error_mm"] > 15.0
    assert result["final_error_mm"] < 1.0, f"recovered {result['final_error_mm']:.2f} mm away"
    assert result["final_loss"] < 1e-3
    assert result["final_containment"] == 0.0


def test_prediction_is_linear_in_the_moment(headfield) -> None:
    """Scaling the source scales the prediction — no hidden nonlinearity."""
    waveform = make_waveform(SPHERE_CONFIG)
    doubled = SourceParams(
        position_cm=SPHERE_TRUTH.position_cm, moment_nam=2.0 * SPHERE_TRUTH.moment_nam
    )
    single = np.asarray(predict_eeg(headfield, SPHERE_TRUTH, waveform, SPHERE_CONFIG))
    double = np.asarray(predict_eeg(headfield, doubled, waveform, SPHERE_CONFIG))
    np.testing.assert_allclose(double, 2.0 * single, rtol=1e-12)


#
# Multi-source plumbing. These run on the sphere backend so they need no
# artifact; the OpenMEEG multi-source path is covered by the benchmark tests.
#


def _sphere_truth(k: int) -> SourceParams:
    """``K`` well-separated sources inside the test sphere."""
    positions = np.array(
        [[0.03, 0.0, 0.02], [-0.03, 0.0, 0.02], [0.0, 0.035, 0.01], [0.0, -0.035, 0.01]]
    )[:k]
    moments = 25e-9 * np.array(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.5], [0.3, 0.0, -1.0]]
    )[:k]
    moments /= np.linalg.norm(moments, axis=1, keepdims=True) / 25e-9
    return SourceParams.from_si(positions, moments)


@pytest.mark.parametrize("k", [1, 2, 4])
def test_forward_and_gradient_shapes_follow_k(headfield, k) -> None:
    """K enters only through array shapes; nothing else may need to know it."""
    truth = _sphere_truth(k)
    assert truth.n_sources == k
    observation = simulate(headfield, truth, SPHERE_CONFIG)
    assert np.asarray(observation.eeg).shape == (1, 32, SPHERE_CONFIG.n_times)

    loss_fn, parts_fn = make_loss(headfield, observation, SPHERE_CONFIG, SPHERE_CONTAINMENT)
    value, grads = jax.value_and_grad(loss_fn)(truth)
    assert np.asarray(grads.position_cm).shape == (k, 3)
    assert np.asarray(grads.moment_nam).shape == (k, 3)
    assert float(value) == pytest.approx(0.0, abs=1e-20)
    assert np.isfinite(np.asarray(grads.position_cm)).all()


@pytest.mark.parametrize("k", [2, 4])
def test_superposition_holds_across_sources(headfield, k) -> None:
    """K sources must be exactly the sum of K single-source forwards."""
    truth = _sphere_truth(k)
    waveform = make_waveform(SPHERE_CONFIG)
    together = np.asarray(predict_eeg(headfield, truth, waveform, SPHERE_CONFIG))
    apart = sum(
        np.asarray(
            predict_eeg(
                headfield,
                SourceParams(
                    position_cm=truth.position_cm[i : i + 1],
                    moment_nam=truth.moment_nam[i : i + 1],
                ),
                waveform,
                SPHERE_CONFIG,
            )
        )
        for i in range(k)
    )
    np.testing.assert_allclose(together, apart, rtol=1e-12)


@pytest.mark.parametrize("k", [2, 4])
def test_least_squares_moment_recovers_all_k_moments(headfield, k) -> None:
    """The forward is jointly linear in K moments, so one solve gets them all."""
    truth = _sphere_truth(k)
    observation = simulate(headfield, truth, SPHERE_CONFIG)
    recovered = least_squares_moment(
        headfield, np.asarray(truth.position_cm), observation, SPHERE_CONFIG
    )
    np.testing.assert_allclose(
        np.asarray(recovered.moment_nam), np.asarray(truth.moment_nam), rtol=1e-5, atol=1e-6
    )


def test_multi_source_recovery_reports_matched_errors(headfield) -> None:
    """Gate D at K=2: both sources found, scored under the optimal assignment."""
    truth = _sphere_truth(2)
    config = LocalizeConfig(
        backend="sphere", n_times=8, n_channels=32, steps=250, separation_weight=1.0
    )
    observation = simulate(headfield, truth, config)
    # Start with the two sources swapped and displaced, so a positional score
    # would be badly wrong even at a perfect solution.
    start_positions = np.asarray(truth.position_cm)[::-1] + np.array(
        [[0.9, -0.6, 0.5], [-0.7, 0.8, -0.4]]
    )
    start = least_squares_moment(headfield, start_positions, observation, config)
    result = run_localization(
        headfield, observation, start, config, SPHERE_CONTAINMENT, record_every=25
    )
    assert result["n_sources"] == 2
    assert len(result["final_error_mm_per_source"]) == 2
    assert result["final_error_mm_max"] < 2.0, result["final_error_mm_per_source"]
    assert result["final_data_loss"] < 1e-4
    assert not result["match"]["collapsed"]
    assert result["final_separation"] == 0.0
    assert np.asarray(result["history"]["position_cm"]).shape[1:] == (2, 3)


def test_separation_penalty_is_zero_when_apart_and_grows_when_close() -> None:
    separation = Separation(min_distance_cm=1.5)
    far = np.array([[3.0, 0.0, 0.0], [-3.0, 0.0, 0.0]])
    assert float(separation.penalty(far)) == 0.0
    near = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    nearer = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]])
    assert 0.0 < float(separation.penalty(near)) < float(separation.penalty(nearer))
    # A single source has no pairs, so the penalty cannot fire.
    assert float(separation.penalty(np.zeros((1, 3)))) == 0.0
    # Three sources contribute three pairs, not nine and not six. (The 1 µm floor
    # is the epsilon that keeps the distance's derivative finite at zero.)
    triple = np.zeros((3, 3))
    assert float(separation.penalty(triple)) == pytest.approx(3 * 1.5**2, rel=1e-5)


def test_separation_penalty_pushes_collapsed_sources_apart(headfield) -> None:
    """The regularizer has to have a usable gradient, not just a value."""
    truth = _sphere_truth(2)
    config = LocalizeConfig(
        backend="sphere", n_times=4, n_channels=32, steps=1, separation_weight=100.0
    )
    observation = simulate(headfield, truth, config)
    collapsed = SourceParams(
        position_cm=jax.numpy.asarray([[1.0, 0.0, 2.0], [1.2, 0.0, 2.0]]),
        moment_nam=truth.moment_nam,
    )
    loss_fn, _ = make_loss(headfield, observation, config, SPHERE_CONTAINMENT)
    grads = jax.grad(loss_fn)(collapsed)
    # The penalty acts along the separation axis, pushing the pair apart.
    along_x = np.asarray(grads.position_cm)[:, 0]
    assert along_x[0] > 0.0 > along_x[1]


#
# Channel subsets
#


def test_channel_subset_changes_the_channel_count_and_its_reference(headfield) -> None:
    """A 16-channel run must carry a 16-channel average reference."""
    subset = tuple(range(0, 32, 2))
    config = LocalizeConfig(
        backend="sphere", n_times=4, n_channels=32, channel_subset=subset
    )
    truth = _sphere_truth(1)
    waveform = make_waveform(config)
    eeg = np.asarray(predict_eeg(headfield, truth, waveform, config))
    assert eeg.shape == (1, len(subset), config.n_times)
    assert np.abs(eeg.mean(axis=1)).max() < 1e-20
    assert sensor_positions(headfield, config).shape == (len(subset), 3)

    # It is not a row selection of the 64-channel referenced forward: that would
    # leave a non-zero channel mean.
    full = np.asarray(predict_eeg(headfield, truth, waveform, SPHERE_CONFIG))
    restricted = full[:, list(subset), :]
    assert np.abs(restricted.mean(axis=1)).max() > 1e-12
    # Both describe the same physics up to the reference, so the differences
    # between channels are identical.
    np.testing.assert_allclose(
        eeg - eeg.mean(axis=1, keepdims=True),
        restricted - restricted.mean(axis=1, keepdims=True),
        rtol=1e-10,
        atol=1e-20,
    )


def test_localization_error_is_in_millimetres() -> None:
    a = SourceParams.from_si(np.array([0.0, 0.0, 0.0]), np.zeros(3))
    b = SourceParams.from_si(np.array([0.003, 0.004, 0.0]), np.zeros(3))
    np.testing.assert_allclose(localization_error_mm(a, b), [5.0], rtol=1e-9)


def test_least_squares_moment_never_returns_a_vanishing_moment(headfield) -> None:
    """A zero moment would zero the position gradient — observed happening once."""
    truth = _sphere_truth(1)
    config = LocalizeConfig(backend="sphere", n_times=16, n_channels=32)
    observation = simulate(headfield, truth, config)
    # Assume a temporal shape orthogonal to the one that generated the data: the
    # least-squares moment is then numerically nothing.
    waveform = make_waveform(config)
    orthogonal = np.zeros_like(waveform)
    orthogonal[0] = 1.0
    orthogonal -= (orthogonal @ waveform) / (waveform @ waveform) * waveform
    assert abs(float(orthogonal @ waveform)) < 1e-12

    warm = least_squares_moment(
        headfield, np.asarray(truth.position_cm), observation, config, orthogonal
    )
    magnitude = float(np.linalg.norm(np.asarray(warm.moment_nam)[0]))
    assert magnitude >= 1.0, "the floor did not engage"

    # And with the right waveform the floor must not interfere with the fit.
    exact = least_squares_moment(
        headfield, np.asarray(truth.position_cm), observation, config, waveform
    )
    np.testing.assert_allclose(
        np.asarray(exact.moment_nam), np.asarray(truth.moment_nam), rtol=1e-5, atol=1e-6
    )


def test_predict_eeg_accepts_a_waveform_per_source(headfield) -> None:
    """Per-source time courses, which is what the unknown-waveform study needs."""
    truth = _sphere_truth(2)
    config = LocalizeConfig(backend="sphere", n_times=16, n_channels=32)
    shared = make_waveform(config)
    from_shared = np.asarray(predict_eeg(headfield, truth, shared, config))
    from_tiled = np.asarray(
        predict_eeg(headfield, truth, np.tile(shared, (2, 1)), config)
    )
    np.testing.assert_allclose(from_tiled, from_shared, rtol=1e-12)

    # Two different shapes must not reduce to either single-shape prediction.
    distinct = np.stack([shared, np.roll(shared, 4)])
    from_distinct = np.asarray(predict_eeg(headfield, truth, distinct, config))
    assert not np.allclose(from_distinct, from_shared)
