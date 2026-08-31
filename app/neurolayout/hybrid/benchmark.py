# Copyright 2026 NeuroLocate contributors
# SPDX-License-Identifier: Apache-2.0
r"""The hard-case benchmark, in its own namespace.

This matrix answers one question and is built to answer only that one:

    Does a learned global proposal, refined through OpenMEEG, localize correlated
    and shared-dynamics multi-source configurations better than either half alone?

So it is deliberately small. ``docs/BENCHMARK.md`` already maps the
mismatch ladder, the electrode count, the noise colour and the temporal model over
25 conditions; none of that is re-run here, and none of it is touched. What this
matrix varies is the **temporal correlation between sources**, which is the axis
that separated a 1.35 mm result from a 19.02 mm one at the same ``K``, the same
SNR and the same physics — the largest single effect in that benchmark, and the
reason this package exists.

Reading the correlation axis
----------------------------
``distinct``
    Independent draws from the waveform family. The easy case; the existing
    ``fam-`` conditions already reach 1.3–1.5 mm here.
``correlated``
    A prescribed mutual cosine of 0.9. Physiologically the common case, and the
    one classical subspace methods start to lose.
``shared``
    One time course for every source. The data matrix is exactly rank one, and no
    method can separate the sources temporally: all that is left is the spatial
    structure of a sum of ``K`` topographies. This is the case the spine of the
    existing benchmark reports 8.6 mm (K=2) and 19.0 mm (K=4) on.

Everything else is held fixed
-----------------------------
One generating forward model (MNE linear-collocation BEM on ico4 with a wrong
skull conductivity and displaced electrodes — the ``full`` rung of the existing
ladder), 64 channels, one epoch length. Every trial's truth is off grid by
construction. The seeds derive from the *problem*, not the condition name, so
``k2-distinct``, ``k2-correlated`` and ``k2-shared`` see the **same** sources at
the same separations, and the difference between their rows is the correlation and
nothing else.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal

from neurolayout.benchmark import SEPARATION_REGIMES, stable_seed
from neurolayout.mismatch import MISMATCH_LEVELS
from neurolayout.noise import NoiseSpec

__all__ = [
    "CorrelationRegime",
    "CORRELATION_VALUES",
    "HybridCondition",
    "CONDITIONS",
    "METHODS",
    "N_RESTARTS",
    "conditions_by_name",
    "conditions_fingerprint",
    "fingerprint",
]

#: The correlation regimes, and the mutual cosine each one asks for.
CorrelationRegime = Literal["distinct", "correlated", "shared"]

#: ``None`` means "draw independently"; a number is the target mutual cosine.
CORRELATION_VALUES: dict[str, float | None] = {
    "distinct": None,
    "correlated": 0.9,
    "shared": 1.0,
}

#: The methods the matrix compares. Order is the order they are reported in.
#:
#: ``rapmusic``
#:     Recursively applied MUSIC on the same OpenMEEG gain, cortically
#:     constrained. The strongest classical multi-source method there is, and the
#:     one that is *designed* for the correlated case.
#: ``scan``
#:     The discrete OpenMEEG dipole scan extended to K sources by orthogonal
#:     matching pursuit, carried over from the existing benchmark.
#: ``gradient``
#:     The continuous OpenMEEG refinement from the uninformed initialization the
#:     frozen benchmark has always used. This is NeuroLocate as it stands.
#: ``gradient_restarts``
#:     The same, from four independent uninformed starts, keeping the one with the
#:     best *data fit* — never the one closest to the truth. This is here because
#:     "the learned proposal is only buying restarts" is the obvious objection to
#:     the whole hypothesis, and it should be answered by a measurement rather than
#:     by an argument. It costs four times the compute of `hybrid`.
#: ``proposal``
#:     The network's output, with no physics applied to it at all.
#: ``hybrid``
#:     The network's output, refined by exactly the same loop ``gradient`` runs.
#:     The main method.
#: ``hybrid_stopgrad``
#:     The same, from a network trained without the through-solver gradient and
#:     with the same total budget. This is what separates "the solver's gradient
#:     helped" from "more training helped".
METHODS: tuple[str, ...] = (
    "rapmusic",
    "scan",
    "gradient",
    "gradient_restarts",
    "proposal",
    "hybrid",
    "hybrid_stopgrad",
)

#: Independent uninformed starts the ``gradient_restarts`` arm draws.
N_RESTARTS = 4


@dataclass(frozen=True)
class HybridCondition:
    """One cell of the hard-case matrix.

    Attributes:
        name: Unique key, used in result files and figure labels.
        n_sources: ``K``.
        correlation: Which :data:`CORRELATION_VALUES` regime.
        separation: A :data:`neurolayout.benchmark.SEPARATION_REGIMES` key, or
            ``None`` for ``K = 1``.
        snr_db: Sensor SNR in dB, or ``None`` for noise-free.
        mismatch: Generating forward model, by
            :data:`neurolayout.mismatch.MISMATCH_LEVELS` key. Every cell uses
            ``full`` — the hardest rung — because the question here is not what
            model error costs; that is already measured.
        n_trials: Deterministic trials in this cell.
        axis: Which experimental axis this cell varies, for grouping.
    """

    name: str
    n_sources: int
    correlation: CorrelationRegime
    separation: str | None = None
    snr_db: float | None = 20.0
    mismatch: str = "full"
    n_trials: int = 8
    axis: str = "core"

    def __post_init__(self) -> None:
        """Reject a cell that cannot be run, at import time."""
        if self.correlation not in CORRELATION_VALUES:
            raise ValueError(f"{self.name}: unknown correlation {self.correlation!r}")
        if self.mismatch not in MISMATCH_LEVELS:
            raise ValueError(f"{self.name}: unknown mismatch {self.mismatch!r}")
        if self.n_sources > 1 and self.separation is None:
            raise ValueError(f"{self.name}: K > 1 needs a separation regime")
        if self.separation is not None and self.separation not in SEPARATION_REGIMES:
            raise ValueError(f"{self.name}: unknown regime {self.separation!r}")
        if self.n_sources == 1 and self.correlation != "distinct":
            raise ValueError(
                f"{self.name}: one source has nothing to correlate with; use "
                "'distinct'"
            )

    @property
    def mismatch_spec(self) -> Any:
        """The resolved generating forward model."""
        return MISMATCH_LEVELS[self.mismatch]

    @property
    def correlation_value(self) -> float | None:
        """The target mutual cosine, or ``None`` for independent draws."""
        return CORRELATION_VALUES[self.correlation]

    @property
    def problem_key(self) -> tuple[Any, ...]:
        """What makes two cells the *same* inverse problem up to the dynamics.

        Deliberately excludes the correlation regime and the SNR, so
        ``k2-distinct``, ``k2-correlated`` and ``k2-shared`` are handed the same
        sources at the same separations and the row-to-row difference is the
        thing being varied.
        """
        return (self.n_sources, self.separation)

    def truth_seed(self, trial: int) -> int:
        """Seed for this trial's ground truth, shared across comparable cells."""
        return stable_seed("hybrid-truth", *self.problem_key, trial)

    def waveform_seed(self, trial: int) -> int:
        """Seed for this trial's source time courses."""
        return stable_seed(
            "hybrid-waveform", *self.problem_key, self.correlation, trial
        )

    def initialization_seed(self, trial: int) -> int:
        """Seed for the uninformed starting point the gradient-only method uses."""
        return stable_seed("hybrid-init", *self.problem_key, trial)

    def noise(self, trial: int) -> NoiseSpec:
        """The noise setting for one trial."""
        return NoiseSpec(
            snr_db=self.snr_db,
            kind="correlated",
            seed=stable_seed("hybrid-noise", *self.problem_key, self.snr_db, trial)
            % (2**31),
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable record."""
        return {
            "name": self.name,
            "n_sources": self.n_sources,
            "correlation": self.correlation,
            "correlation_value": self.correlation_value,
            "separation": self.separation,
            "snr_db": self.snr_db,
            "mismatch": self.mismatch,
            "n_trials": self.n_trials,
            "axis": self.axis,
        }


#: The matrix. Seven core cells — the ones the brief names — plus three
#: excursions, one per axis that could plausibly change the conclusion.
CONDITIONS: tuple[HybridCondition, ...] = (
    # --- control -------------------------------------------------------------
    HybridCondition("h-k1", 1, "distinct", axis="control"),
    # --- the seven core cells ------------------------------------------------
    HybridCondition("h-k2-distinct", 2, "distinct", separation="moderate"),
    HybridCondition("h-k2-correlated", 2, "correlated", separation="moderate"),
    HybridCondition("h-k2-shared", 2, "shared", separation="moderate"),
    HybridCondition("h-k4-distinct", 4, "distinct", separation="spread"),
    HybridCondition("h-k4-correlated", 4, "correlated", separation="spread"),
    HybridCondition("h-k4-shared", 4, "shared", separation="spread"),
    # --- excursion: separation, at the regime where topographies collide -----
    HybridCondition(
        "h-k2-shared-close", 2, "shared", separation="close", axis="separation"
    ),
    # --- excursion: SNR ------------------------------------------------------
    HybridCondition(
        "h-k2-shared-10db", 2, "shared", separation="moderate", snr_db=10.0, axis="snr"
    ),
    HybridCondition(
        "h-k4-shared-10db", 4, "shared", separation="spread", snr_db=10.0, axis="snr"
    ),
)


def conditions_by_name() -> dict[str, HybridCondition]:
    """The matrix as a lookup, checking that every name is unique."""
    table: dict[str, HybridCondition] = {}
    for condition in CONDITIONS:
        if condition.name in table:
            raise ValueError(f"duplicate condition name {condition.name!r}")
        table[condition.name] = condition
    return table


def conditions_fingerprint(
    conditions: tuple[HybridCondition, ...] = CONDITIONS,
) -> str:
    """SHA-256 over the condition definitions alone.

    This is what an *observation artifact* is bound to, and it is deliberately
    narrower than :func:`fingerprint`. The generator draws sources, dynamics,
    noise and a forward model; it has never heard of the method list, and binding
    its output to one would mean regenerating a hundred BEM solutions every time
    a baseline is added. Conflating the two would also create a quiet pressure not
    to add baselines, which is the wrong pressure to have.
    """
    payload = json.dumps(
        {
            "conditions": [condition.to_dict() for condition in conditions],
            "correlation_values": CORRELATION_VALUES,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def fingerprint(conditions: tuple[HybridCondition, ...] = CONDITIONS) -> str:
    """SHA-256 over the matrix definition and the method list.

    Committed **before** the results, and checked by the runner, so "the benchmark
    was adjusted until it said something" is a claim the repository's own history
    can refute. The stake is low here because the benchmark is synthetic, so it
    is reproducible from source rather than merely auditable.
    """
    payload = json.dumps(
        {
            "conditions": [condition.to_dict() for condition in conditions],
            "methods": list(METHODS),
            "n_restarts": N_RESTARTS,
            "correlation_values": CORRELATION_VALUES,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()
