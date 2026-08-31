# Copyright 2026 NeuroLocate contributors
# SPDX-License-Identifier: Apache-2.0
r"""Training observations, drawn from the gain bank.

One sample is: ``K`` source positions with a separation drawn from a stated band,
a moment per source, a time course per source with a stated mutual correlation,
one of the bank's head models, and sensor noise at a stated SNR. The EEG is then

.. math::  y[c, t] = \sum_k G_v(p_k)[c, :] \cdot m_k \; a_k(t) \; + \; n[c, t]

with every ``G_v(p_k)`` a real OpenMEEG column read out of the bank. Nothing here
approximates the physics; the only thing that is generated on the fly is the
linear algebra the map is linear in.

The correlation axis is the whole point
---------------------------------------
``docs/BENCHMARK.md`` measures a 14x difference between four sources with
distinct time courses (1.35 mm) and four sources sharing one (19.02 mm), at the
same ``K``, the same SNR and the same physics. That is larger than every other
effect in that benchmark put together, and it is the regime this package exists
for. So the training distribution samples the correlation *deliberately* and
heavily, rather than drawing time courses independently and hoping.

At correlation 1 the data matrix is exactly rank one however many sources there
are, and no method can separate them from their time courses; all that is left is
the spatial structure of a sum of ``K`` topographies. That is a real, hard,
identifiable-only-by-prior problem, and it is the one the proposal network is
being asked to learn.

Separation is drawn, not fixed
------------------------------
Two sources 8 cm apart are two easy problems. Two sources 15 mm apart are one hard
one. Training draws the pairwise separation log-uniformly across the whole range,
so the network sees the hard band often enough to learn it without the easy band
disappearing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from neurolayout.hybrid.bank import Bank
from neurolayout.noise import sensor_covariance
from neurolayout.waveforms import sample_waveform, unit_rms

__all__ = [
    "CorrelationMode",
    "SynthSpec",
    "DEFAULT_SPEC",
    "Sample",
    "Synthesizer",
    "correlated_waveforms",
]

#: How the ``K`` source time courses relate to one another.
#:
#: ``distinct``
#:     Independent draws from the waveform family. The easy case, and the one the
#:     existing benchmark's ``fam-`` conditions use.
#: ``correlated``
#:     A prescribed mutual cosine, drawn in ``[0.6, 0.98]``. The realistic case:
#:     cortical sources driven by a common input are correlated but not identical.
#: ``shared``
#:     One time course for every source, so the data matrix is exactly rank one.
#:     The classical worst case, and the one the spine of the existing benchmark
#:     reports 19 mm on at ``K = 4``.
CorrelationMode = Literal["distinct", "correlated", "shared"]


def correlated_waveforms(
    rng: np.random.Generator,
    n_sources: int,
    correlation: float,
    *,
    n_times: int,
    sfreq: float,
) -> np.ndarray:
    """``[K, T]`` unit-RMS time courses with an approximately equal mutual cosine.

    Generalizes :func:`neurolayout.waveforms.sample_pair` past two sources. ``K``
    independent draws are orthonormalized, then recombined against the Cholesky
    factor of the exchangeable correlation matrix
    ``(1 - rho) I + rho 11ᵀ``, which is positive definite for every
    ``rho in [0, 1)`` and rank one at ``rho = 1``.

    Args:
        rng: Draw source.
        n_sources: ``K``.
        correlation: Target mutual cosine in ``[0, 1]``. Exactly 1 returns ``K``
            copies of one waveform.
        n_times: Epoch length ``T``.
        sfreq: Sampling rate, Hz.

    Returns:
        ``[K, T]`` unit-RMS waveforms.
    """
    if not 0.0 <= correlation <= 1.0:
        raise ValueError(f"correlation must be in [0, 1], got {correlation}")
    draws = np.stack(
        [sample_waveform(rng, n_times=n_times, sfreq=sfreq) for _ in range(n_sources)]
    )
    if n_sources == 1:
        return draws
    if correlation >= 1.0:
        return np.repeat(draws[:1], n_sources, axis=0)
    # An orthonormal basis spanning the same K waveforms; QR on the transpose so
    # the rows come out orthonormal.
    basis, _ = np.linalg.qr(draws.T)
    basis = basis.T[:n_sources]
    if basis.shape[0] < n_sources:  # degenerate draw; fall back to independence
        return unit_rms(draws)
    target = (1.0 - correlation) * np.eye(n_sources) + correlation
    factor = np.linalg.cholesky(target)
    return unit_rms(factor @ basis)


@dataclass(frozen=True)
class SynthSpec:
    """The training distribution.

    Attributes:
        n_times: Epoch length ``T``. Matched to the benchmark's 32 samples.
        sfreq: Sampling rate, Hz.
        k_choices: The source counts to draw from.
        k_weights: Relative frequency of each ``K``. ``K = 2`` and ``K = 4`` carry
            most of the weight because they are what the benchmark measures;
            ``K = 1`` and ``K = 3`` are kept so the network is not a two-count
            classifier in disguise.
        separation_mm: Range of pairwise distances, sampled log-uniformly. The
            lower end is below what 64 electrodes resolve on purpose.
        correlation_weights: Relative frequency of each :data:`CorrelationMode`.
        correlation_range: Bounds for the ``correlated`` mode's cosine.
        amplitude_nam: Range of dipole magnitudes, nano-ampere-metres,
            log-uniform. A source that is ten times weaker than its neighbour is
            a different problem from two equal ones.
        orientation_jitter_deg: Angle by which a source's moment may depart from
            the local cortical normal. Not zero: the estimator is free-orientation
            and must not learn that the normal is the answer.
        snr_db: Range of sensor SNRs, uniform in dB.
        clean_fraction: Share of samples with no sensor noise at all.
        white_noise_fraction: Share of noisy samples with white rather than
            spatially correlated noise.
        dropout_prob: Probability that a sample drops channels.
        max_dropped: Most channels a sample may drop. Dropped channels are
            re-referenced over the survivors, which is the reference a recording
            with that many electrodes would carry.
    """

    n_times: int = 32
    sfreq: float = 160.0
    k_choices: tuple[int, ...] = (1, 2, 3, 4)
    k_weights: tuple[float, ...] = (0.15, 0.35, 0.15, 0.35)
    separation_mm: tuple[float, float] = (10.0, 140.0)
    correlation_weights: dict[str, float] = field(
        default_factory=lambda: {"distinct": 0.3, "correlated": 0.35, "shared": 0.35}
    )
    correlation_range: tuple[float, float] = (0.6, 0.98)
    amplitude_nam: tuple[float, float] = (8.0, 60.0)
    orientation_jitter_deg: float = 25.0
    snr_db: tuple[float, float] = (0.0, 30.0)
    clean_fraction: float = 0.1
    white_noise_fraction: float = 0.3
    dropout_prob: float = 0.15
    max_dropped: int = 8

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable record."""
        return {
            "n_times": self.n_times,
            "sfreq": self.sfreq,
            "k_choices": list(self.k_choices),
            "k_weights": list(self.k_weights),
            "separation_mm": list(self.separation_mm),
            "correlation_weights": dict(self.correlation_weights),
            "correlation_range": list(self.correlation_range),
            "amplitude_nam": list(self.amplitude_nam),
            "orientation_jitter_deg": self.orientation_jitter_deg,
            "snr_db": list(self.snr_db),
            "clean_fraction": self.clean_fraction,
            "white_noise_fraction": self.white_noise_fraction,
            "dropout_prob": self.dropout_prob,
            "max_dropped": self.max_dropped,
        }


#: The default training distribution, as a singleton so it can be a default
#: argument without being rebuilt on every call.
DEFAULT_SPEC = SynthSpec()


@dataclass(frozen=True)
class Sample:
    """One synthetic observation and the answer it was made from.

    Attributes:
        eeg: ``[C, T]`` sensor signals, volts, average-referenced over the
            retained channels.
        mask: ``[C]`` 1.0 for a retained channel, 0.0 for a dropped one.
        positions_m: ``[K, 3]`` true source positions, metres, head frame.
        moments_nam: ``[K, 3]`` true dipole moments, nano-ampere-metres.
        waveforms: ``[K, T]`` true unit-RMS time courses.
        n_sources: ``K``.
        variant: Which head model generated it.
        correlation_mode: Which correlation regime.
        realized_correlation: The measured mean off-diagonal cosine.
        snr_db: The requested SNR, or ``None`` for a clean sample.
        separation_mm: Minimum pairwise distance, or ``inf`` for ``K = 1``.
    """

    eeg: np.ndarray
    mask: np.ndarray
    positions_m: np.ndarray
    moments_nam: np.ndarray
    waveforms: np.ndarray
    n_sources: int
    variant: str
    correlation_mode: str
    realized_correlation: float
    snr_db: float | None
    separation_mm: float


class Synthesizer:
    """Draws training observations from a gain bank.

    Args:
        bank: The gain bank.
        spec: The training distribution.
        indices: Which bank positions this synthesizer may use. Passing the two
            halves of :meth:`neurolayout.hybrid.bank.Bank.split` gives a
            validation set whose *sources* the network has never had a gain for,
            not merely ones it has not seen in this combination of moment,
            waveform and noise.
        variants: Which head models to draw from, by name. ``None`` is all of
            them; holding one out is how a forward model becomes a test condition.
    """

    #: Attempts to place one source in the requested separation band before the
    #: draw gives up and accepts the closest it found. A band that is empty for a
    #: particular anchor is a property of the head's geometry, not a bug: a source
    #: near the occipital pole has no bank neighbour 140 mm away.
    _PLACEMENT_ATTEMPTS = 24

    def __init__(
        self,
        bank: Bank,
        spec: SynthSpec = DEFAULT_SPEC,
        *,
        indices: np.ndarray | None = None,
        variants: tuple[str, ...] | None = None,
    ) -> None:
        """Cache the per-variant views and the noise factorizations."""
        self.bank = bank
        self.spec = spec
        self.indices = (
            np.arange(bank.n_positions) if indices is None else np.asarray(indices)
        )
        self.variant_indices = (
            np.arange(len(bank.variants))
            if variants is None
            else np.array([bank.variant_index(name) for name in variants])
        )
        self.positions = np.asarray(bank.positions_m)[self.indices]
        self.normals = np.asarray(bank.normals)[self.indices]
        self._tree = _build_tree(self.positions)
        self._noise_factor = {
            kind: np.linalg.cholesky(
                sensor_covariance(bank.sensor_xyz, kind)
                + 1e-10 * np.eye(bank.n_channels)
            )
            for kind in ("white", "correlated")
        }
        self._weights = np.asarray(spec.k_weights, dtype=np.float64)
        self._weights = self._weights / self._weights.sum()
        modes = list(spec.correlation_weights)
        self._modes = modes
        mode_weights = np.array([spec.correlation_weights[m] for m in modes])
        self._mode_weights = mode_weights / mode_weights.sum()

    #
    # The draw
    #

    def draw(
        self,
        rng: np.random.Generator,
        *,
        n_sources: int | None = None,
        correlation_mode: str | None = None,
        separation_mm: float | None = None,
        snr_db: float | None = None,
        clean: bool = False,
        variant: str | None = None,
    ) -> Sample:
        """One observation. Every axis can be forced, for tests and for figures.

        ``snr_db=None`` means *draw* one, which is the training behaviour; ``clean``
        is the separate request for no sensor noise at all. Two flags rather than a
        sentinel, because "unspecified" and "none" are different requests and
        conflating them made a test silently skip itself.
        """
        n_sources = (
            int(rng.choice(self.spec.k_choices, p=self._weights))
            if n_sources is None
            else int(n_sources)
        )
        picks, separation = self._place(rng, n_sources, separation_mm)
        moments = self._moments(rng, picks)
        mode = (
            str(rng.choice(self._modes, p=self._mode_weights))
            if correlation_mode is None
            else correlation_mode
        )
        waveforms, correlation = self._waveforms(rng, n_sources, mode)

        variant_index = (
            int(rng.choice(self.variant_indices))
            if variant is None
            else self.bank.variant_index(variant)
        )
        gains = np.asarray(self.bank.gains[variant_index, self.indices[picks]])  # [K,C,3]
        columns = np.einsum("kcj,kj->kc", gains.astype(np.float64), moments * 1e-9)
        clean_eeg = columns.T @ waveforms  # [C, T]

        mask = self._mask(rng)
        noisy, realized_snr = self._noise(rng, clean_eeg, snr_db, clean)
        eeg = _apply_mask(noisy, mask)
        return Sample(
            eeg=eeg,
            mask=mask,
            positions_m=self.positions[picks],
            moments_nam=moments,
            waveforms=waveforms,
            n_sources=n_sources,
            variant=self.bank.variants[variant_index].name,
            correlation_mode=mode,
            realized_correlation=correlation,
            snr_db=realized_snr,
            separation_mm=separation,
        )

    def batch(
        self,
        rng: np.random.Generator,
        size: int,
        *,
        with_samples: bool = True,
        **forced: Any,
    ) -> dict[str, Any]:
        """``size`` samples, padded to the largest ``K`` in the batch.

        Returns arrays rather than :class:`Sample` objects, because the training
        loop wants tensors and the padding has to be explicit: ``source_mask``
        says which of the ``K_max`` slots is a real source.

        Args:
            rng: Draw source.
            size: Batch size.
            with_samples: Also return the :class:`Sample` objects under
                ``"samples"``. Evaluation needs them (for the correlation regime
                and ``K`` of each trial); the training loader must not, because a
                dataclass does not survive a DataLoader worker's collation.
            forced: Any draw axis to hold fixed, forwarded to :meth:`draw`.
        """
        samples = [self.draw(rng, **forced) for _ in range(size)]
        k_max = max(sample.n_sources for sample in samples)
        n_channels, n_times = samples[0].eeg.shape
        positions = np.zeros((size, k_max, 3))
        moments = np.zeros((size, k_max, 3))
        source_mask = np.zeros((size, k_max), dtype=np.float32)
        eeg = np.zeros((size, n_channels, n_times))
        mask = np.zeros((size, n_channels), dtype=np.float32)
        for index, sample in enumerate(samples):
            k = sample.n_sources
            positions[index, :k] = sample.positions_m
            moments[index, :k] = sample.moments_nam
            source_mask[index, :k] = 1.0
            eeg[index] = sample.eeg
            mask[index] = sample.mask
        return {
            "eeg": eeg,
            "channel_mask": mask,
            "positions_m": positions,
            "moments_nam": moments,
            "source_mask": source_mask,
            "n_sources": np.array([s.n_sources for s in samples], dtype=np.int64),
            **({"samples": samples} if with_samples else {}),
        }

    #
    # The pieces
    #

    def _place(
        self, rng: np.random.Generator, n_sources: int, separation_mm: float | None
    ) -> tuple[np.ndarray, float]:
        """Indices (into :attr:`indices`) of ``K`` sources in a separation band.

        The first source is uniform over the bank. Each further source is placed
        by drawing a direction and a distance and snapping to the nearest bank
        position, which is a vectorized tree query rather than rejection sampling
        over pairwise distances — at 15 mm separation the rejection rate would be
        four nines.
        """
        picks = [int(rng.integers(len(self.positions)))]
        if n_sources == 1:
            return np.array(picks), float("inf")
        low, high = self.spec.separation_mm
        for _ in range(n_sources - 1):
            wanted = (
                float(np.exp(rng.uniform(np.log(low), np.log(high))))
                if separation_mm is None
                else float(separation_mm)
            )
            picks.append(self._nearby(rng, picks, wanted))
        chosen = np.array(picks)
        points = self.positions[chosen]
        distance = np.linalg.norm(points[:, None] - points[None], axis=-1) * 1e3
        return chosen, float(distance[~np.eye(n_sources, dtype=bool)].min())

    def _nearby(self, rng: np.random.Generator, taken: list[int], wanted_mm: float) -> int:
        """A bank position about ``wanted_mm`` from the last one, and not already taken."""
        anchor = self.positions[taken[-1]]
        best: int | None = None
        best_error = np.inf
        for _ in range(self._PLACEMENT_ATTEMPTS):
            direction = rng.standard_normal(3)
            direction /= max(np.linalg.norm(direction), 1e-30)
            target = anchor + direction * wanted_mm * 1e-3
            candidate = int(_query(self._tree, target))
            if candidate in taken:
                continue
            error = abs(
                float(np.linalg.norm(self.positions[candidate] - anchor)) * 1e3 - wanted_mm
            )
            if error < best_error:
                best, best_error = candidate, error
            if error < 0.15 * wanted_mm:
                return candidate
        if best is None:  # every direction landed on a taken source
            free = int(rng.integers(len(self.positions)))
            while free in taken:
                free = int(rng.integers(len(self.positions)))
            return free
        return best

    def _moments(self, rng: np.random.Generator, picks: np.ndarray) -> np.ndarray:
        """``[K, 3]`` moments: the local normal, jittered, at a log-uniform magnitude."""
        low, high = self.spec.amplitude_nam
        magnitude = np.exp(rng.uniform(np.log(low), np.log(high), len(picks)))
        directions = _jitter_directions(
            rng, self.normals[picks], self.spec.orientation_jitter_deg
        )
        return directions * magnitude[:, None]

    def _waveforms(
        self, rng: np.random.Generator, n_sources: int, mode: str
    ) -> tuple[np.ndarray, float]:
        """``([K, T], realized mean off-diagonal cosine)``."""
        if mode == "distinct":
            correlation = 0.0
        elif mode == "shared":
            correlation = 1.0
        else:
            correlation = float(rng.uniform(*self.spec.correlation_range))
        waveforms = correlated_waveforms(
            rng,
            n_sources,
            correlation,
            n_times=self.spec.n_times,
            sfreq=self.spec.sfreq,
        )
        return waveforms, _mean_cosine(waveforms)

    def _mask(self, rng: np.random.Generator) -> np.ndarray:
        """``[C]`` channel mask. Mostly all ones."""
        mask = np.ones(self.bank.n_channels, dtype=np.float32)
        if self.spec.dropout_prob <= 0.0 or rng.uniform() >= self.spec.dropout_prob:
            return mask
        count = int(rng.integers(1, self.spec.max_dropped + 1))
        mask[rng.choice(self.bank.n_channels, count, replace=False)] = 0.0
        return mask

    def _noise(
        self,
        rng: np.random.Generator,
        clean: np.ndarray,
        snr_db: float | None,
        forced_clean: bool = False,
    ) -> tuple[np.ndarray, float | None]:
        """Add sensor noise at a drawn SNR; return the data and the SNR used."""
        if forced_clean:
            return clean, None
        if snr_db is None and rng.uniform() < self.spec.clean_fraction:
            return clean, None
        target = (
            float(rng.uniform(*self.spec.snr_db)) if snr_db is None else float(snr_db)
        )
        kind = "white" if rng.uniform() < self.spec.white_noise_fraction else "correlated"
        white = rng.standard_normal(clean.shape)
        noise = self._noise_factor[kind] @ white
        # The forward is average-referenced, so the noise is too: otherwise the
        # network could detect a source by the one thing that is not in the
        # reference subspace.
        noise = noise - noise.mean(axis=0, keepdims=True)
        clean_rms = float(np.sqrt(np.mean(clean**2)))
        noise_rms = float(np.sqrt(np.mean(noise**2)))
        if noise_rms <= 0.0 or clean_rms <= 0.0:
            return clean, None
        noise *= clean_rms * 10.0 ** (-target / 20.0) / noise_rms
        return clean + noise, target


def _apply_mask(eeg: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Zero the dropped channels and re-reference over the survivors.

    A subset of an average-referenced array does not carry the average reference
    its own channel count implies, and the difference is exactly a constant per
    time sample — which is why removing it again is legitimate: the reference
    operator annihilates constants, so re-referencing the *already referenced*
    subset gives the same answer as referencing the raw subset would have.
    """
    if mask.all():
        return eeg
    kept = mask > 0.0
    out = np.array(eeg, dtype=np.float64, copy=True)
    out[~kept] = 0.0
    out[kept] -= out[kept].mean(axis=0, keepdims=True)
    return out


def _jitter_directions(
    rng: np.random.Generator, normals: np.ndarray, jitter_deg: float
) -> np.ndarray:
    """Unit directions within ``jitter_deg`` of each normal, uniform on the cap."""
    normals = np.asarray(normals, dtype=np.float64)
    normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-30)
    if jitter_deg <= 0.0:
        return normals
    cos_max = np.cos(np.radians(jitter_deg))
    cosine = rng.uniform(cos_max, 1.0, len(normals))
    sine = np.sqrt(np.maximum(1.0 - cosine**2, 0.0))
    angle = rng.uniform(0.0, 2.0 * np.pi, len(normals))
    # An orthonormal frame around each normal, from the least-aligned axis.
    helper = np.zeros_like(normals)
    helper[np.arange(len(normals)), np.argmin(np.abs(normals), axis=1)] = 1.0
    first = np.cross(normals, helper)
    first /= np.maximum(np.linalg.norm(first, axis=1, keepdims=True), 1e-30)
    second = np.cross(normals, first)
    return (
        cosine[:, None] * normals
        + (sine * np.cos(angle))[:, None] * first
        + (sine * np.sin(angle))[:, None] * second
    )


def _mean_cosine(waveforms: np.ndarray) -> float:
    """Mean off-diagonal cosine of a set of time courses."""
    if len(waveforms) < 2:
        return float("nan")
    unit = waveforms / np.maximum(
        np.linalg.norm(waveforms, axis=1, keepdims=True), 1e-30
    )
    gram = unit @ unit.T
    return float(np.abs(gram[~np.eye(len(waveforms), dtype=bool)]).mean())


def _build_tree(points: np.ndarray) -> Any:
    """A KD-tree over the bank positions, for the separation-band placement."""
    from scipy.spatial import cKDTree

    return cKDTree(np.asarray(points, dtype=np.float64))


def _query(tree: Any, point: np.ndarray) -> int:
    """Index of the bank position nearest a target point."""
    return int(tree.query(np.asarray(point, dtype=np.float64))[1])
