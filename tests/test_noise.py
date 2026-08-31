# Copyright 2026 NeuroLocate contributors
# SPDX-License-Identifier: Apache-2.0
"""The noise model has to be reproducible, correlated, and honest about its SNR."""

from __future__ import annotations

import numpy as np
import pytest
from neurolayout.noise import (
    NoiseSpec,
    add_sensor_noise,
    sensor_covariance,
    snr_db,
)

#: A small ring of electrodes on a 9 cm sphere, spaced ~3.5 cm apart.
ANGLES = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
SENSORS = 0.09 * np.stack(
    [np.cos(ANGLES), np.sin(ANGLES), np.zeros_like(ANGLES)], axis=1
)


def _clean(rng: np.random.Generator, shape: tuple[int, ...] = (1, 16, 32)) -> np.ndarray:
    return 1e-6 * rng.standard_normal(shape)


def test_white_covariance_is_the_identity() -> None:
    np.testing.assert_allclose(sensor_covariance(SENSORS, "white"), np.eye(16))


def test_correlated_covariance_is_a_valid_decaying_kernel() -> None:
    covariance = sensor_covariance(SENSORS, "correlated", 0.04)
    np.testing.assert_allclose(np.diag(covariance), 1.0)
    np.testing.assert_allclose(covariance, covariance.T)
    assert np.linalg.eigvalsh(covariance).min() > 0.0
    # Neighbouring electrodes are more correlated than antipodal ones.
    assert covariance[0, 1] > covariance[0, 4] > covariance[0, 8]


def test_shorter_correlation_length_decorrelates_faster() -> None:
    near = sensor_covariance(SENSORS, "correlated", 0.01)
    far = sensor_covariance(SENSORS, "correlated", 0.10)
    assert near[0, 1] < far[0, 1]


def test_realized_snr_matches_the_request() -> None:
    rng = np.random.default_rng(0)
    clean = _clean(rng)
    for target in (20.0, 10.0, 0.0):
        for kind in ("white", "correlated"):
            noisy, report = add_sensor_noise(
                clean, SENSORS, NoiseSpec(snr_db=target, kind=kind, seed=1)
            )
            assert report["realized_snr_db"] == pytest.approx(target, abs=1e-9)
            assert snr_db(clean, noisy - clean) == pytest.approx(target, abs=1e-9)


def test_noise_is_reproducible_from_the_seed_and_varies_with_it() -> None:
    rng = np.random.default_rng(0)
    clean = _clean(rng)
    spec = NoiseSpec(snr_db=15.0, kind="correlated", seed=7)
    first, _ = add_sensor_noise(clean, SENSORS, spec)
    again, _ = add_sensor_noise(clean, SENSORS, spec)
    other, _ = add_sensor_noise(clean, SENSORS, NoiseSpec(snr_db=15.0, seed=8))
    np.testing.assert_array_equal(first, again)
    assert not np.allclose(first, other)


def test_correlated_noise_is_spatially_smoother_than_white_noise() -> None:
    """The point of the correlated model: adjacent channels move together."""
    clean = np.zeros((1, 16, 4096))
    ones = np.ones((1, 16, 4096))

    def adjacent_correlation(noise: np.ndarray) -> float:
        flat = noise[0]
        centred = flat - flat.mean(axis=1, keepdims=True)
        normed = centred / np.linalg.norm(centred, axis=1, keepdims=True)
        return float(np.mean([normed[i] @ normed[i + 1] for i in range(15)]))

    white, _ = add_sensor_noise(clean + ones, SENSORS, NoiseSpec(20.0, "white", seed=3))
    correlated, _ = add_sensor_noise(
        clean + ones, SENSORS, NoiseSpec(20.0, "correlated", 0.04, seed=3)
    )
    # Adjacent ring electrodes are 3.5 cm apart, so the kernel predicts
    # exp(-0.035/0.04) = 0.42, and the realized sample correlation must find it.
    expected = float(sensor_covariance(SENSORS, "correlated", 0.04)[0, 1])
    assert adjacent_correlation(correlated - ones) == pytest.approx(expected, abs=0.05)
    assert abs(adjacent_correlation(white - ones)) < 0.1


def test_noise_respects_the_reference_operator() -> None:
    """Referenced noise must survive re-referencing unchanged, or the SNR lies."""
    rng = np.random.default_rng(0)
    clean = _clean(rng)
    reference = np.eye(16) - np.full((16, 16), 1.0 / 16)
    noisy, _ = add_sensor_noise(
        clean, SENSORS, NoiseSpec(10.0, "correlated", seed=2), reference_operator=reference
    )
    noise = noisy - clean
    assert np.abs(noise.mean(axis=1)).max() < 1e-18
    unreferenced, _ = add_sensor_noise(clean, SENSORS, NoiseSpec(10.0, "correlated", seed=2))
    assert np.abs((unreferenced - clean).mean(axis=1)).max() > 1e-12


def test_clean_spec_is_a_no_op() -> None:
    rng = np.random.default_rng(0)
    clean = _clean(rng)
    noisy, report = add_sensor_noise(clean, SENSORS, NoiseSpec(snr_db=None))
    np.testing.assert_array_equal(noisy, clean)
    assert report["realized_snr_db"] is None
    assert NoiseSpec(snr_db=None).tag == "clean"
    assert NoiseSpec(snr_db=20.0, kind="white").tag == "20dB-white"
