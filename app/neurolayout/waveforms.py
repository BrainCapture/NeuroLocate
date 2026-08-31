# Copyright 2026 NeuroLocate contributors
# SPDX-License-Identifier: Apache-2.0
r"""A synthetic family of cortical source time courses, and its sampler.

This is the temporal distribution the proposal network's synthetic training
observations are drawn from, and it is written here — in the application, not in
the component — because it is a modelling assumption about neural activity that
belongs next to the experiment rather than buried inside a trained artifact.

The family is deliberately *generic*. A network trained on the exact waveform the
benchmark uses would only have memorized the answer, so the family spans

``burst``
    a Gaussian-windowed oscillation — the canonical evoked/induced transient;
``rhythm``
    a sustained 8–13 Hz mu/alpha oscillation with slow amplitude drift, the
    rhythm motor imagery actually modulates;
``am``
    an amplitude-modulated carrier, where the envelope and the carrier are at
    different rates;
``transient``
    a single biphasic pulse with no oscillatory content at all;
``mixture``
    a weighted sum of two of the above, so the prior cannot assume unimodality.

and the benchmark's own stimulus (a Hann-tapered 10 Hz burst) is a *member of the
family's support but not a training sample*: it is one particular burst among a
continuum, which is exactly the generalization the comparison needs.

Every waveform is scaled to unit RMS. Amplitude is not the prior's business — it
lives in the dipole moment, where the forward is exactly linear in it and the
derivative is exact.

Correlated pairs
----------------
:func:`sample_pair` draws two waveforms with a controlled temporal correlation.
Correlated sources are the classical hard case for sparse inverse methods: when
two source time courses are collinear, their topographies add and the pair is
indistinguishable from a single source somewhere in between. The pair sampler
exists so the prior sees that regime during training and so the benchmark can
create it on purpose.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

__all__ = [
    "KINDS",
    "DEFAULT_N_TIMES",
    "DEFAULT_SFREQ",
    "unit_rms",
    "sample_waveform",
    "sample_pair",
    "sample_dataset",
]

WaveformKind = Literal["burst", "rhythm", "am", "transient", "mixture"]

#: The elementary shapes, sampled uniformly unless weights are given.
KINDS: tuple[WaveformKind, ...] = ("burst", "rhythm", "am", "transient", "mixture")

#: Epoch length and sampling rate the prior is trained at. 32 samples at 160 Hz is
#: 200 ms — the EEGBCI sampling rate, and long enough to hold two cycles of mu.
DEFAULT_N_TIMES = 32
DEFAULT_SFREQ = 160.0


def unit_rms(waveform: np.ndarray) -> np.ndarray:
    """Scale to unit root-mean-square along the last axis."""
    waveform = np.asarray(waveform, dtype=np.float64)
    rms = np.sqrt(np.mean(waveform**2, axis=-1, keepdims=True))
    return waveform / np.maximum(rms, 1e-12)


def _times(n_times: int, sfreq: float) -> np.ndarray:
    return np.arange(n_times) / sfreq


def _burst(rng: np.random.Generator, n_times: int, sfreq: float) -> np.ndarray:
    times = _times(n_times, sfreq)
    duration = times[-1] if n_times > 1 else 1.0 / sfreq
    frequency = rng.uniform(6.0, 30.0)
    centre = rng.uniform(0.25, 0.75) * duration
    width = rng.uniform(0.12, 0.35) * duration
    phase = rng.uniform(0.0, 2.0 * np.pi)
    envelope = np.exp(-0.5 * ((times - centre) / width) ** 2)
    return envelope * np.sin(2.0 * np.pi * frequency * times + phase)


def _rhythm(rng: np.random.Generator, n_times: int, sfreq: float) -> np.ndarray:
    times = _times(n_times, sfreq)
    duration = times[-1] if n_times > 1 else 1.0 / sfreq
    frequency = rng.uniform(8.0, 13.0)
    phase = rng.uniform(0.0, 2.0 * np.pi)
    # A slow amplitude drift across the epoch: mu rhythm is not stationary, and a
    # prior that assumes it is would be a worse model than a Fourier basis.
    drift = 1.0 + rng.uniform(-0.6, 0.6) * (times / max(duration, 1e-12) - 0.5) * 2.0
    return drift * np.sin(2.0 * np.pi * frequency * times + phase)


def _am(rng: np.random.Generator, n_times: int, sfreq: float) -> np.ndarray:
    times = _times(n_times, sfreq)
    carrier_hz = rng.uniform(12.0, 30.0)
    envelope_hz = rng.uniform(2.0, 6.0)
    depth = rng.uniform(0.4, 1.0)
    phase = rng.uniform(0.0, 2.0 * np.pi)
    envelope = 1.0 - depth * 0.5 * (1.0 + np.cos(2.0 * np.pi * envelope_hz * times))
    return envelope * np.sin(2.0 * np.pi * carrier_hz * times + phase)


def _transient(rng: np.random.Generator, n_times: int, sfreq: float) -> np.ndarray:
    times = _times(n_times, sfreq)
    duration = times[-1] if n_times > 1 else 1.0 / sfreq
    centre = rng.uniform(0.3, 0.7) * duration
    width = rng.uniform(0.05, 0.15) * duration
    scaled = (times - centre) / width
    # Derivative of a Gaussian: one positive and one negative lobe, no carrier.
    return -scaled * np.exp(-0.5 * scaled**2) * rng.choice([-1.0, 1.0])


_SAMPLERS = {
    "burst": _burst,
    "rhythm": _rhythm,
    "am": _am,
    "transient": _transient,
}


def sample_waveform(
    rng: np.random.Generator,
    *,
    n_times: int = DEFAULT_N_TIMES,
    sfreq: float = DEFAULT_SFREQ,
    kind: WaveformKind | None = None,
) -> np.ndarray:
    """Draw one unit-RMS source time course.

    Args:
        rng: Draw source.
        n_times: Epoch length ``T``.
        sfreq: Sampling rate, Hz.
        kind: Force a particular shape, or ``None`` to draw one uniformly.

    Returns:
        ``[T]`` unit-RMS waveform.
    """
    choice = kind if kind is not None else str(rng.choice(KINDS))
    if choice == "mixture":
        first, second = rng.choice(list(_SAMPLERS), 2, replace=False)
        weight = rng.uniform(0.3, 0.7)
        blended = weight * unit_rms(_SAMPLERS[first](rng, n_times, sfreq)) + (
            1.0 - weight
        ) * unit_rms(_SAMPLERS[second](rng, n_times, sfreq))
        return unit_rms(blended)
    return unit_rms(_SAMPLERS[choice](rng, n_times, sfreq))


def sample_pair(
    rng: np.random.Generator,
    correlation: float,
    *,
    n_times: int = DEFAULT_N_TIMES,
    sfreq: float = DEFAULT_SFREQ,
) -> np.ndarray:
    """Two unit-RMS waveforms with an approximately prescribed correlation.

    Built by Gram–Schmidt: draw two independent waveforms, orthogonalize the
    second against the first, and recombine at the requested cosine. The realized
    correlation is exact up to the mean-removal the RMS normalization does not do,
    so callers that need the number should measure it.

    Args:
        rng: Draw source.
        correlation: Target cosine between the two, in ``[-1, 1]``.
        n_times: Epoch length ``T``.
        sfreq: Sampling rate, Hz.

    Returns:
        ``[2, T]`` unit-RMS waveforms.
    """
    if not -1.0 <= correlation <= 1.0:
        raise ValueError(f"correlation must be in [-1, 1], got {correlation}")
    first = sample_waveform(rng, n_times=n_times, sfreq=sfreq)
    other = sample_waveform(rng, n_times=n_times, sfreq=sfreq)
    residual = other - (other @ first) / (first @ first) * first
    norm = np.linalg.norm(residual)
    if norm < 1e-12:  # degenerate draw; fall back to an independent second
        return np.stack([first, other])
    residual /= norm
    unit_first = first / np.linalg.norm(first)
    second = correlation * unit_first + np.sqrt(max(1.0 - correlation**2, 0.0)) * residual
    return np.stack([unit_rms(first), unit_rms(second)])


def sample_dataset(
    n_samples: int,
    *,
    n_times: int = DEFAULT_N_TIMES,
    sfreq: float = DEFAULT_SFREQ,
    seed: int = 0,
    pair_fraction: float = 0.25,
) -> np.ndarray:
    """A training set of unit-RMS waveforms.

    A fraction of the set comes from :func:`sample_pair` at a random correlation,
    so the prior's training distribution contains the collinear pairs that make
    multi-source localization hard, rather than only independent draws.

    Args:
        n_samples: Number of waveforms.
        n_times: Epoch length ``T``.
        sfreq: Sampling rate, Hz.
        seed: Deterministic seed.
        pair_fraction: Fraction of samples drawn as halves of correlated pairs.

    Returns:
        ``[n_samples, T]`` unit-RMS waveforms.
    """
    rng = np.random.default_rng(seed)
    samples: list[np.ndarray] = []
    while len(samples) < n_samples:
        if rng.uniform() < pair_fraction:
            pair = sample_pair(
                rng, float(rng.uniform(-0.95, 0.95)), n_times=n_times, sfreq=sfreq
            )
            samples.extend(pair)
        else:
            samples.append(sample_waveform(rng, n_times=n_times, sfreq=sfreq))
    return np.stack(samples[:n_samples])
