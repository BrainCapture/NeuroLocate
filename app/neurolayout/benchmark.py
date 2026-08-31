# Copyright 2026 NeuroLocate contributors
# SPDX-License-Identifier: Apache-2.0
r"""The design of the synthetic inverse benchmark: trials, and the condition matrix.

Everything that decides *what* is measured lives here, separately from the code
that measures it, so the experiment can be read without reading the runner — and
so the runner cannot quietly change the experiment.

Ground truth is **off grid**
----------------------------
True sources are never placed on an ico5 source-space vertex. If they were, the
discrete-scan and sparse-classical baselines could find them exactly and the
comparison would measure nothing but grid alignment. Each truth is placed at an
interior point of the segment joining a vertex to its nearest neighbour, with the
orientation interpolated between their cortical normals, at a deterministic
mixing weight. The median nearest-neighbour spacing is 2.1 mm, so this puts every
truth roughly 0.7–1.1 mm off the nearest candidate — exactly the regime where a
continuous estimator can do something a grid cannot.

Source separation is a controlled variable
------------------------------------------
For ``K > 1`` the interesting axis is not "how many sources" but "how far apart".
Two sources 8 cm apart are two easy problems; two sources 15 mm apart are one hard
one, because their topographies are nearly collinear at 64 electrodes.
:data:`SEPARATION_REGIMES` fixes three bands and the trial drawer rejects any
configuration outside the requested one.

The condition matrix is bounded on purpose
------------------------------------------
:data:`CONDITIONS` is not the Cartesian product of every axis. It is a spine —
``K`` crossed with SNR at 64 channels — plus one-at-a-time excursions along model
mismatch, source separation and electrode count. That keeps the whole sweep to a
few hundred trials, which fits in an evening, while still letting each axis be
read separately.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from neurolayout.mismatch import MISMATCH_LEVELS, MismatchSpec
from neurolayout.noise import NoiseSpec

__all__ = [
    "stable_seed",
    "MOMENT_AM",
    "NEAR_DISTANCE_M",
    "SEPARATION_REGIMES",
    "Truth",
    "Condition",
    "CONDITIONS",
    "conditions_by_name",
    "offgrid_truth",
    "draw_truth",
    "near_initialization",
    "random_initialization",
    "depth_mm",
    "source_spacing_mm",
]


def stable_seed(*parts: Any) -> int:
    """A seed derived from the parts, identical across processes and versions.

    ``hash()`` is randomized per interpreter run, so a benchmark seeded with it
    would not be reproducible. This is a SHA-256 digest of the parts' string
    forms, truncated to 63 bits.
    """
    key = "\x00".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") % (2**63)


#: Dipole strength of every ground-truth source, in A·m. 25 nA·m is a
#: conventional equivalent-dipole magnitude for a focal cortical patch.
MOMENT_AM = 25e-9

#: Distance of the "near" initialization from each true source, in metres.
NEAR_DISTANCE_M = 0.024

#: Pairwise-distance bands the trial drawer enforces, in millimetres. "far" is two
#: independent problems; "close" is at the edge of what 64 electrodes resolve.
SEPARATION_REGIMES: dict[str, tuple[float, float]] = {
    "far": (60.0, 140.0),
    "moderate": (28.0, 40.0),
    "close": (12.0, 18.0),
    "spread": (30.0, 140.0),
}


@dataclass(frozen=True)
class Truth:
    """The hidden answer to one trial.

    Attributes:
        positions_m: ``[K, 3]`` source positions, metres, MNE head frame. Off grid
            by construction.
        moments_am: ``[K, 3]`` dipole moments, A·m, along interpolated cortical
            normals at :attr:`MOMENT_AM` strength.
        vertices: ``[K]`` source-space indices the truths were derived from, for
            provenance only.
        offgrid_mm: ``[K]`` distance from each truth to its nearest source-space
            vertex, millimetres — the resolution a grid method cannot beat.
    """

    positions_m: np.ndarray
    moments_am: np.ndarray
    vertices: tuple[int, ...]
    offgrid_mm: np.ndarray

    @property
    def n_sources(self) -> int:
        """``K``."""
        return int(self.positions_m.shape[0])

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable record."""
        return {
            "positions_m": self.positions_m.tolist(),
            "moments_am": self.moments_am.tolist(),
            "vertices": list(self.vertices),
            "offgrid_mm": [float(x) for x in self.offgrid_mm],
        }


@dataclass(frozen=True)
class Condition:
    """One cell of the benchmark matrix.

    Attributes:
        name: Unique key, used in result files and figure labels.
        n_sources: ``K``.
        mismatch: Which generating forward model to use, by
            :data:`neurolayout.mismatch.MISMATCH_LEVELS` key.
        snr_db: Sensor SNR, or ``None`` for noise-free.
        noise_kind: ``"white"`` or ``"correlated"``.
        channels: A :mod:`neurolayout.channel_subsets` name.
        separation: A :data:`SEPARATION_REGIMES` key, or ``None`` for ``K = 1``.
        n_trials: Deterministic trials in this cell.
        initializations: Which starting-point regimes to run.
        axis: Which experimental axis this cell is an excursion along, for
            grouping in the report.
        waveform_source: ``"known"`` — the fixed benchmark burst, which the
            estimator is told; or ``"family"`` — a held-out draw from
            :mod:`neurolayout.waveforms`, which it is not. The ``family``
            conditions are the ones where the free-versus-learned-prior comparison
            means anything.
        waveform_correlation: For ``family`` conditions with ``K = 2``, the target
            temporal correlation between the two source time courses. ``None``
            draws them independently.
    """

    name: str
    n_sources: int
    mismatch: str = "solver"
    snr_db: float | None = 20.0
    noise_kind: Literal["white", "correlated"] = "correlated"
    channels: str = "all"
    separation: str | None = None
    n_trials: int = 6
    initializations: tuple[str, ...] = ("near", "far")
    axis: str = "spine"
    waveform_source: Literal["known", "family"] = "known"
    waveform_correlation: float | None = None

    def __post_init__(self) -> None:
        """Reject a condition that cannot be run, at import time rather than hour three."""
        if self.mismatch not in MISMATCH_LEVELS:
            raise ValueError(f"condition {self.name!r}: unknown mismatch {self.mismatch!r}")
        if self.n_sources > 1 and self.separation is None:
            raise ValueError(f"condition {self.name!r}: K > 1 needs a separation regime")
        if self.separation is not None and self.separation not in SEPARATION_REGIMES:
            raise ValueError(f"condition {self.name!r}: unknown regime {self.separation!r}")

    @property
    def mismatch_spec(self) -> MismatchSpec:
        """The resolved generator specification."""
        return MISMATCH_LEVELS[self.mismatch]

    @property
    def problem_key(self) -> tuple[Any, ...]:
        r"""What makes two conditions the *same inverse problem* up to the model.

        Seeds are derived from this rather than from the condition name, so every
        condition with the same ``K``, separation regime and waveform setting sees
        **the same true sources** and the same starting points. That is what makes
        the mismatch ladder and the SNR spine comparable trial by trial: a
        difference between ``k1-solver`` and ``k1-skull`` is the skull conductivity,
        not a different draw of sources.
        """
        return (
            self.n_sources,
            self.separation,
            self.waveform_source,
            self.waveform_correlation,
        )

    def truth_seed(self, trial: int) -> int:
        """Seed for this trial's ground truth, shared across comparable conditions."""
        return stable_seed("truth", *self.problem_key, trial)

    def waveform_seed(self, trial: int) -> int:
        """Seed for this trial's source time courses."""
        return stable_seed("waveform", *self.problem_key, trial)

    def initialization_seed(self, trial: int, initialization: str) -> int:
        """Seed for this trial's starting point."""
        return stable_seed("init", *self.problem_key, trial, initialization)

    def noise(self, trial: int) -> NoiseSpec:
        """The noise setting for one trial.

        Seeded from the noise setting rather than the condition name, so two
        conditions that differ only in forward model or electrode count get the
        same noise realization and can be compared directly.
        """
        return NoiseSpec(
            snr_db=self.snr_db,
            kind=self.noise_kind,
            seed=stable_seed(
                "noise", *self.problem_key, self.snr_db, self.noise_kind, trial
            )
            % (2**31),
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable record."""
        return {
            "name": self.name,
            "n_sources": self.n_sources,
            "mismatch": self.mismatch,
            "snr_db": self.snr_db,
            "noise_kind": self.noise_kind,
            "channels": self.channels,
            "separation": self.separation,
            "n_trials": self.n_trials,
            "initializations": list(self.initializations),
            "axis": self.axis,
            "waveform_source": self.waveform_source,
            "waveform_correlation": self.waveform_correlation,
        }


#: The benchmark matrix. Read it as: a spine of K x SNR at 64 channels under the
#: solver mismatch, plus one-at-a-time excursions.
CONDITIONS: tuple[Condition, ...] = (
    # --- mismatch ladder at K=1: what each source of model error costs ---------
    # Ordered by measured severity (scale-free topography residual): 0% / 1.0% /
    # 6.7% / 7.5% / 9.3%, so the table reads as the ladder it is.
    Condition("k1-matched", 1, mismatch="matched", axis="mismatch"),
    Condition("k1-solver", 1, mismatch="solver", axis="mismatch"),
    Condition("k1-electrodes", 1, mismatch="electrodes", axis="mismatch"),
    Condition("k1-skull", 1, mismatch="skull", axis="mismatch"),
    Condition("k1-full", 1, mismatch="full", axis="mismatch"),
    # --- spine: K x SNR, 64 channels, solver mismatch -------------------------
    Condition("k1-clean", 1, snr_db=None),
    Condition("k1-20db", 1, snr_db=20.0),
    Condition("k1-10db", 1, snr_db=10.0),
    Condition("k2-clean", 2, separation="moderate", snr_db=None),
    Condition("k2-20db", 2, separation="moderate", snr_db=20.0),
    Condition("k2-10db", 2, separation="moderate", snr_db=10.0),
    Condition("k4-clean", 4, separation="spread", snr_db=None),
    Condition("k4-20db", 4, separation="spread", snr_db=20.0),
    Condition("k4-10db", 4, separation="spread", snr_db=10.0),
    # --- excursion: source separation at K=2, 20 dB ---------------------------
    Condition("k2-far", 2, separation="far", axis="separation"),
    Condition("k2-close", 2, separation="close", axis="separation"),
    # --- excursion: electrode count at K=4, 20 dB -----------------------------
    Condition("k4-cap32", 4, separation="spread", channels="cap32", axis="channels"),
    Condition("k4-clinical16", 4, separation="spread", channels="clinical16", axis="channels"),
    # --- excursion: noise colour ----------------------------------------------
    Condition("k2-20db-white", 2, separation="moderate", noise_kind="white", axis="noise"),
    # --- the free-vs-learned-prior study: unknown source time courses ---------
    # These are the only conditions where the temporal model matters, because they
    # are the only ones where the waveform is not handed to the estimator.
    Condition("fam-k1-20db", 1, waveform_source="family", axis="prior"),
    Condition("fam-k2-20db", 2, separation="moderate", waveform_source="family",
              axis="prior"),
    Condition("fam-k2-10db", 2, separation="moderate", snr_db=10.0,
              waveform_source="family", axis="prior"),
    Condition("fam-k2-corr", 2, separation="moderate", waveform_source="family",
              waveform_correlation=0.9, axis="prior"),
    Condition("fam-k4-20db", 4, separation="spread", waveform_source="family",
              axis="prior"),
    Condition("fam-k4-clinical16", 4, separation="spread", channels="clinical16",
              waveform_source="family", axis="prior"),
)


def conditions_by_name() -> dict[str, Condition]:
    """The matrix as a lookup, checking that every name is unique."""
    table: dict[str, Condition] = {}
    for condition in CONDITIONS:
        if condition.name in table:
            raise ValueError(f"duplicate condition name {condition.name!r}")
        table[condition.name] = condition
    return table


#
# Trial construction
#


def offgrid_truth(
    source_space: np.ndarray,
    source_normals: np.ndarray,
    index: int,
    weight: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Place a source *between* two neighbouring source-space vertices.

    Args:
        source_space: ``[S, 3]`` candidate locations, metres.
        source_normals: ``[S, 3]`` unit cortical normals.
        index: The vertex to start from.
        weight: Mixing weight toward its nearest neighbour, in ``(0, 1)``.

    Returns:
        ``(position_m, unit_normal, offgrid_mm)`` where ``offgrid_mm`` is the
        distance to the nearest source-space vertex.
    """
    vertex = source_space[index]
    distances = np.linalg.norm(source_space - vertex, axis=1)
    distances[index] = np.inf
    neighbour = int(np.argmin(distances))
    position = (1.0 - weight) * vertex + weight * source_space[neighbour]
    normal = (1.0 - weight) * source_normals[index] + weight * source_normals[neighbour]
    offgrid = float(np.linalg.norm(source_space - position, axis=1).min() * 1e3)
    return position, normal / np.linalg.norm(normal), offgrid


def draw_truth(
    source_space: np.ndarray,
    source_normals: np.ndarray,
    n_sources: int,
    *,
    separation: str | None,
    seed: int,
    moment_am: float = MOMENT_AM,
    max_attempts: int = 20000,
) -> Truth:
    """Draw ``K`` off-grid sources whose pairwise distances lie in one regime.

    Rejection sampling: draw ``K`` candidate vertices uniformly from the source
    space, keep the draw only if every pairwise distance falls inside the band.
    Uniform sampling over the cortical source space means depth and laterality
    vary across trials on their own, which is what we want — the benchmark should
    not be a hand-picked set of easy superficial sources.

    Args:
        source_space: ``[S, 3]`` candidate locations, metres.
        source_normals: ``[S, 3]`` unit cortical normals.
        n_sources: ``K``.
        separation: A :data:`SEPARATION_REGIMES` key; ignored for ``K = 1``.
        seed: Deterministic seed for this trial.
        moment_am: Dipole strength for every source.
        max_attempts: Give up after this many rejected draws.

    Returns:
        A :class:`Truth`.

    Raises:
        ValueError: If no configuration in the requested band was found. Better a
            loud failure than a benchmark that silently ran a different regime.
    """
    rng = np.random.default_rng(seed)
    n_candidates = source_space.shape[0]
    low, high = (
        (0.0, np.inf) if n_sources == 1 or separation is None
        else SEPARATION_REGIMES[separation]
    )

    for _ in range(max_attempts):
        picks = rng.choice(n_candidates, n_sources, replace=False)
        points = source_space[picks]
        if n_sources > 1:
            distance = np.linalg.norm(points[:, None] - points[None], axis=-1) * 1e3
            offdiag = distance[~np.eye(n_sources, dtype=bool)]
            if offdiag.min() < low or offdiag.max() > high:
                continue
        positions, normals, offgrid = [], [], []
        for index in picks:
            position, normal, distance_mm = offgrid_truth(
                source_space, source_normals, int(index), float(rng.uniform(0.35, 0.65))
            )
            positions.append(position)
            normals.append(normal)
            offgrid.append(distance_mm)
        return Truth(
            positions_m=np.stack(positions),
            moments_am=moment_am * np.stack(normals),
            vertices=tuple(int(i) for i in picks),
            offgrid_mm=np.asarray(offgrid),
        )

    raise ValueError(
        f"no K={n_sources} configuration with pairwise separation in "
        f"[{low}, {high}] mm after {max_attempts} draws"
    )


def near_initialization(
    brain_vertices: np.ndarray,
    truth_positions_m: np.ndarray,
    *,
    distance_m: float = NEAR_DISTANCE_M,
) -> np.ndarray:
    """Starting positions ``distance_m`` from each truth, displaced inward.

    The direction is two thirds inward (toward the brain centroid) and one third
    along a fixed tangential axis, rather than a fixed offset vector. A fixed
    vector pushes superficial sources straight out through the skull, where the
    forward model is meaningless and the trial measures the containment penalty
    instead of the method.
    """
    centre = np.asarray(brain_vertices, dtype=np.float64).mean(axis=0)
    starts = []
    for position in np.atleast_2d(np.asarray(truth_positions_m, dtype=np.float64)):
        inward = centre - position
        inward /= max(np.linalg.norm(inward), 1e-12)
        tangential = np.cross(inward, [0.0, 0.0, 1.0])
        tangential /= max(np.linalg.norm(tangential), 1e-12)
        direction = 2.0 * inward + tangential
        starts.append(position + distance_m * direction / np.linalg.norm(direction))
    return np.stack(starts)


def random_initialization(
    containment: Any,
    truth_positions_m: np.ndarray,
    rng: np.random.Generator,
    *,
    min_distance_m: float = 0.03,
    min_mutual_m: float = 0.02,
) -> np.ndarray:
    """Starting positions drawn inside the brain, far from every truth.

    Args:
        containment: The :class:`neurolayout.localize.Containment` ellipsoid.
        truth_positions_m: ``[K, 3]`` the answers, used only to reject starts that
            happen to land near one — "far" has to actually be far.
        rng: Draw source.
        min_distance_m: Minimum distance from every true source.
        min_mutual_m: Minimum distance between two starting points, so the run
            does not begin already collapsed.

    Returns:
        ``[K, 3]`` starting positions, metres.
    """
    from neurolayout.localize import METRES_PER_CM

    truth = np.atleast_2d(np.asarray(truth_positions_m, dtype=np.float64))
    starts: list[np.ndarray] = []
    while len(starts) < truth.shape[0]:
        candidate = (
            containment.centre_cm + 0.6 * containment.semi_axes_cm * rng.standard_normal(3)
        ) * METRES_PER_CM
        if not containment.contains(candidate):
            continue
        if np.linalg.norm(truth - candidate, axis=1).min() <= min_distance_m:
            continue
        if starts and np.linalg.norm(np.stack(starts) - candidate, axis=1).min() <= min_mutual_m:
            continue
        starts.append(candidate)
    return np.stack(starts)


def depth_mm(brain_vertices: np.ndarray, position_m: np.ndarray) -> float:
    """Distance from a point to the nearest inner-skull vertex, in millimetres.

    A superficial source is where BEM discretizations disagree most, so this is
    reported for every trial and is one axis of the identifiability analysis.
    """
    return float(np.linalg.norm(np.asarray(brain_vertices) - position_m, axis=1).min() * 1e3)


def source_spacing_mm(source_space: np.ndarray, sample: int = 300, seed: int = 0) -> float:
    """Median nearest-neighbour distance in the source space, in millimetres.

    A grid method cannot localize better than about half of this, by construction.
    It is the number the continuous method is being compared against.
    """
    rng = np.random.default_rng(seed)
    points = np.asarray(source_space, dtype=np.float64)
    nearest = []
    for index in rng.choice(points.shape[0], sample, replace=False):
        distances = np.linalg.norm(points - points[index], axis=1)
        distances[index] = np.inf
        nearest.append(distances.min())
    return float(np.median(nearest) * 1e3)


#: Convenience: the matrix grouped by which axis each condition varies.
CONDITIONS_BY_AXIS: dict[str, tuple[Condition, ...]] = {
    axis: tuple(c for c in CONDITIONS if c.axis == axis)
    for axis in dict.fromkeys(c.axis for c in CONDITIONS)
}
