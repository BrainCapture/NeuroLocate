# Copyright 2026 NeuroLocate contributors
# SPDX-License-Identifier: Apache-2.0
r"""Sensor-noise models for the synthetic inverse benchmark.

Independent white noise is the easiest thing to add to a simulated EEG epoch and
the least like real EEG. Real sensor noise is **spatially correlated**: nearby
electrodes share reference drift, movement artifact, and — dominantly —
background brain activity that volume conduction has already smeared across the
scalp. Correlated noise is also the harder case for an inverse method, because
a spatially smooth noise field looks more like a distant source than white noise
does.

Two models are supported, both reproducible from an integer seed:

``white``
    :math:`\Sigma = I`. The baseline.

``correlated``
    :math:`\Sigma_{ij} = \exp(-d_{ij}/\lambda)`, with :math:`d_{ij}` the
    Euclidean distance between electrodes ``i`` and ``j`` on the scalp and
    :math:`\lambda` a correlation length (default 4 cm, the scale over which
    scalp potentials from a cortical patch decorrelate). This is the standard
    exponential (Ornstein–Uhlenbeck) kernel; it is positive definite for any
    point set, needs no fitting, and is written into the artifact so a reader can
    reconstruct it exactly.

Both are generated *after* the forward model and then passed through the same
average-reference operator the forward carries, so the noise lives in the same
63-dimensional subspace as the signal. Noise that ignored the reference would be
partly removable by re-referencing, which would make the stated SNR a fiction.

Definition of SNR
-----------------
.. math::

    \mathrm{SNR}_{\mathrm{dB}} = 20 \log_{10}
        \frac{\mathrm{rms}_{c,t}(\text{clean})}{\mathrm{rms}_{c,t}(\text{noise})}

The root-mean-square runs over *all* channels and time samples of the epoch
(after referencing), not per channel and not over a baseline window. It is a
broadband amplitude ratio of the whole epoch, which is the quantity a reader can
recompute from the stored arrays; it is deliberately not the "SNR" of any
particular EEG convention. The realized value is measured after the fact and
stored, so the number in a results table is what the data actually has.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

__all__ = [
    "DEFAULT_CORRELATION_LENGTH_M",
    "NoiseSpec",
    "sensor_covariance",
    "add_sensor_noise",
    "snr_db",
]

#: Correlation length of the exponential sensor-noise kernel, in metres.
DEFAULT_CORRELATION_LENGTH_M = 0.04

NoiseKind = Literal["white", "correlated"]


@dataclass(frozen=True)
class NoiseSpec:
    """A reproducible sensor-noise setting.

    Attributes:
        snr_db: Target SNR in dB, or ``None`` for a noise-free observation.
        kind: ``"white"`` or ``"correlated"``.
        correlation_length_m: Kernel length scale; ignored when ``kind`` is
            ``"white"``.
        seed: Seed for the noise realization.
    """

    snr_db: float | None = None
    kind: NoiseKind = "correlated"
    correlation_length_m: float = DEFAULT_CORRELATION_LENGTH_M
    seed: int = 0

    @property
    def tag(self) -> str:
        """Short identifier used in result keys and figure labels."""
        if self.snr_db is None:
            return "clean"
        return f"{self.snr_db:g}dB-{self.kind}"

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable description, for provenance in result files."""
        return {
            "snr_db": self.snr_db,
            "kind": self.kind,
            "correlation_length_m": (
                None if self.kind == "white" else self.correlation_length_m
            ),
            "seed": self.seed,
        }


def sensor_covariance(
    sensor_xyz: np.ndarray,
    kind: NoiseKind = "correlated",
    correlation_length_m: float = DEFAULT_CORRELATION_LENGTH_M,
) -> np.ndarray:
    """Unit-diagonal sensor-noise covariance for an electrode array.

    Args:
        sensor_xyz: ``[C, 3]`` electrode positions, metres.
        kind: ``"white"`` (identity) or ``"correlated"`` (exponential kernel).
        correlation_length_m: Kernel length scale in metres.

    Returns:
        ``[C, C]`` covariance with unit diagonal.
    """
    positions = np.asarray(sensor_xyz, dtype=np.float64)
    n_channels = positions.shape[0]
    if kind == "white":
        return np.eye(n_channels)
    if correlation_length_m <= 0.0:
        raise ValueError(f"correlation_length_m must be positive, got {correlation_length_m}")
    distance = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    return np.exp(-distance / correlation_length_m)


def snr_db(clean: np.ndarray, noise: np.ndarray) -> float:
    """Broadband amplitude SNR in dB, per the definition in the module docstring."""
    clean_rms = float(np.sqrt(np.mean(np.asarray(clean) ** 2)))
    noise_rms = float(np.sqrt(np.mean(np.asarray(noise) ** 2)))
    if noise_rms == 0.0:
        return float("inf")
    return 20.0 * np.log10(clean_rms / noise_rms)


def add_sensor_noise(
    clean: np.ndarray,
    sensor_xyz: np.ndarray,
    spec: NoiseSpec,
    *,
    reference_operator: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Add noise at a requested SNR, returning the data and what was done to it.

    Args:
        clean: ``[..., C, T]`` noise-free sensor signals.
        sensor_xyz: ``[C, 3]`` electrode positions, metres.
        spec: The noise setting.
        reference_operator: ``[C, C]`` referencing matrix applied to the noise so
            it occupies the same subspace as the referenced forward. Omit for an
            unreferenced forward.

    Returns:
        ``(noisy, report)`` where ``report`` records the spec, the realized SNR
        and the noise RMS in volts.
    """
    clean = np.asarray(clean, dtype=np.float64)
    if spec.snr_db is None:
        return clean.copy(), {**spec.to_dict(), "realized_snr_db": None, "noise_rms_v": 0.0}

    n_channels = clean.shape[-2]
    covariance = sensor_covariance(sensor_xyz, spec.kind, spec.correlation_length_m)
    if covariance.shape[0] != n_channels:
        raise ValueError(
            f"clean has {n_channels} channels but sensor_xyz has {covariance.shape[0]}"
        )
    # Jitter keeps the factorization well conditioned; the exponential kernel is
    # positive definite in theory but can be numerically singular for close
    # electrode pairs.
    factor = np.linalg.cholesky(covariance + 1e-10 * np.eye(n_channels))

    rng = np.random.default_rng(spec.seed)
    white = rng.standard_normal(clean.shape)
    noise = np.einsum("ij,...jt->...it", factor, white)
    if reference_operator is not None:
        noise = np.einsum("ij,...jt->...it", np.asarray(reference_operator), noise)

    clean_rms = float(np.sqrt(np.mean(clean**2)))
    noise_rms = float(np.sqrt(np.mean(noise**2)))
    if noise_rms == 0.0:  # degenerate array; nothing sensible to scale
        raise ValueError("noise realization has zero RMS")
    target_rms = clean_rms * 10.0 ** (-spec.snr_db / 20.0)
    noise = noise * (target_rms / noise_rms)

    noisy = clean + noise
    return noisy, {
        **spec.to_dict(),
        "realized_snr_db": snr_db(clean, noise),
        "noise_rms_v": float(np.sqrt(np.mean(noise**2))),
        "clean_rms_v": clean_rms,
    }
