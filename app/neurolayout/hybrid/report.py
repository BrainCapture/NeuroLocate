# Copyright 2026 NeuroLocate contributors
# SPDX-License-Identifier: Apache-2.0
r"""Aggregating the hard-case shards into the numbers the document reports.

What is reported, and why each one is there:

**Median and IQR, not a mean.** The error distribution on correlated sources is
strongly right-skewed: most trials land close and a few land on the wrong side of
the head. A mean is then a summary of the tail, and a median with an IQR beside it
says both things separately.

**Worst-source, not mean-source.** A ``K = 4`` trial that recovers three sources
perfectly and misses the fourth entirely is a failed trial, and averaging over its
four sources would report it as a mild one. Every per-trial number here is the
**maximum** over that trial's sources; the per-source distribution is reported
alongside, and the two are labelled.

**Catastrophic failure separately from accuracy.** ">20 mm" is a rate, not a
moment of the distribution, and a method can improve its median while losing more
trials outright.

**Paired differences.** Every method sees the same trials, so the comparison that
matters is per-trial and paired — a method can win on the median while losing on
most trials, and a bootstrap over trials says which happened.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from neurolayout.matching import COLLAPSE_THRESHOLD_MM

__all__ = [
    "load_shards",
    "summarize",
    "method_summary",
    "paired_comparison",
    "by_separation",
    "detection_summary",
    "RECOVERED_MM",
]


def load_shards(directory: str | Path) -> list[dict[str, Any]]:
    """Every trial record in a shard directory, with its shard context attached."""
    records: list[dict[str, Any]] = []
    for path in sorted(Path(directory).glob("*.json")):
        shard = json.loads(path.read_text())
        for trial in shard["trials"]:
            records.append({**trial, "_shard": {
                key: value for key, value in shard.items() if key != "trials"
            }})
    return records


def _values(records: list[dict], method: str, field: str) -> np.ndarray:
    """One number per trial for a method, dropping the trials it did not produce."""
    out = []
    for record in records:
        entry = record["methods"].get(method)
        if entry is None or entry.get(field) is None:
            continue
        out.append(float(entry[field]))
    return np.asarray(out)


def _bootstrap(values: np.ndarray, statistic, draws: int = 4000, seed: int = 5):
    """Percentile bootstrap interval over trials.

    Over *trials*, which are independent here by construction: each trial draws
    its own sources, its own dynamics and its own noise. A benchmark whose trials
    were nested inside a smaller number of heads would need a clustered interval;
    these are not, and saying so is why this is a plain bootstrap.
    """
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    samples = [
        statistic(values[rng.integers(0, len(values), len(values))]) for _ in range(draws)
    ]
    return (float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5)))


def method_summary(records: list[dict], method: str) -> dict[str, Any]:
    """The distribution of one method's error over a set of trials."""
    worst = _values(records, method, "worst_mm")
    per_source = np.concatenate(
        [
            np.asarray(record["methods"][method]["errors_mm"])
            for record in records
            if record["methods"].get(method, {}).get("errors_mm")
        ]
        or [np.array([])]
    )
    residual = _values(records, method, "sensor_residual")
    converged = [
        record["methods"][method]["converged"]
        for record in records
        if record["methods"].get(method, {}).get("converged") is not None
    ]
    seconds = _values(records, method, "seconds")
    attempted = sum(1 for record in records if method in record["methods"])
    if len(worst) == 0:
        return {"method": method, "n": 0, "attempted": attempted}
    low, high = _bootstrap(worst, np.median)
    return {
        "method": method,
        "n": int(len(worst)),
        "attempted": attempted,
        "failed": attempted - int(len(worst)),
        "median_mm": float(np.median(worst)),
        "ci_low_mm": low,
        "ci_high_mm": high,
        "q1_mm": float(np.percentile(worst, 25)),
        "q3_mm": float(np.percentile(worst, 75)),
        "mean_mm": float(worst.mean()),
        "worst_mm": float(worst.max()),
        "per_source_median_mm": (
            float(np.median(per_source)) if per_source.size else None
        ),
        "gross_failure_rate": float((worst > 20.0).mean()),
        "sensor_residual_median": (
            float(np.median(residual)) if residual.size else None
        ),
        "seconds_median": float(np.median(seconds)) if seconds.size else None,
        # Reported next to the error, never merged into it: a run can converge
        # cleanly onto the wrong sources, and at K=4 on shared dynamics many do.
        "converged_rate": (
            float(np.mean(converged)) if converged else None
        ),
    }


#: A source counts as recovered when its matched estimate is within this, in mm.
#:
#: The same 20 mm the frozen benchmark calls a gross failure, so "recovered" and
#: "not a gross failure" are the same statement at the source level rather than
#: two thresholds a reader has to reconcile.
RECOVERED_MM = 20.0


def detection_summary(records: list[dict], method: str) -> dict[str, Any]:
    """Recall, and the two ways a multi-source estimate fails without being wrong.

    **Recall equals precision here, and is reported once.** Every method in this
    matrix is handed ``K`` and returns exactly ``K`` estimates, so the assignment
    is a bijection: an estimate that misses is simultaneously a false positive and
    a missed source, and reporting the same number in two columns would suggest
    two measurements where there is one. If ``K`` were ever predicted rather than
    given, the two would separate and this function would have to.

    **Merged** counts trials where two estimates landed within
    :data:`neurolayout.matching.COLLAPSE_THRESHOLD_MM` of each other — the
    characteristic degenerate solution of multi-dipole fitting, where ``K``
    parameter sets explain one topography while a true source goes unexplained.
    It is not an error in millimetres and it does not show up as one: a collapsed
    pair can sit on top of a true source and score well on the estimate that
    matched it.

    **Missed** is the per-source complement of recall, and is the number that says
    what "97% gross failure at K = 4" actually means — whether a method is
    slightly wrong about everything or exactly right about most of it.
    """
    total_sources = recovered = merged = trials = 0
    for record in records:
        entry = record["methods"].get(method)
        if entry is None or not entry.get("errors_mm"):
            continue
        errors = np.asarray(entry["errors_mm"], dtype=float)
        trials += 1
        total_sources += errors.size
        recovered += int((errors <= RECOVERED_MM).sum())
        separation = entry.get("min_separation_mm")
        if separation is not None and float(separation) < COLLAPSE_THRESHOLD_MM:
            merged += 1
    if not trials:
        return {"method": method, "n_trials": 0}
    return {
        "method": method,
        "n_trials": trials,
        "n_sources": total_sources,
        "recall": recovered / total_sources,
        "missed_rate": 1.0 - recovered / total_sources,
        "merged_rate": merged / trials,
        "recovered_mm": RECOVERED_MM,
        "collapse_mm": COLLAPSE_THRESHOLD_MM,
    }


def paired_comparison(records: list[dict], method: str, reference: str) -> dict[str, Any]:
    """Per-trial difference ``method - reference``, with a bootstrap interval.

    Negative means ``method`` has the smaller error. Trials where either method
    produced nothing are dropped and counted, because a paired test needs a pair.
    """
    differences, dropped = [], 0
    for record in records:
        first = record["methods"].get(method, {}).get("worst_mm")
        second = record["methods"].get(reference, {}).get("worst_mm")
        if first is None or second is None:
            dropped += 1
            continue
        differences.append(float(first) - float(second))
    values = np.asarray(differences)
    if values.size == 0:
        return {"method": method, "reference": reference, "n": 0, "dropped": dropped}
    low, high = _bootstrap(values, np.median)
    return {
        "method": method,
        "reference": reference,
        "n": int(values.size),
        "dropped": dropped,
        "median_difference_mm": float(np.median(values)),
        "ci_low_mm": low,
        "ci_high_mm": high,
        "better_fraction": float((values < 0.0).mean()),
        "excludes_zero": bool(low * high > 0.0),
    }


def by_separation(
    records: list[dict], method: str, edges: tuple[float, ...] = (0.0, 20.0, 35.0, 1e9)
) -> list[dict[str, Any]]:
    """One summary per band of true source separation.

    A global median hides the axis that decides multi-source localization. Two
    sources 15 mm apart and two 80 mm apart are different problems, and a method
    that is good at one and useless at the other should not be reported as
    mediocre at both.
    """
    bands = []
    for low, high in zip(edges[:-1], edges[1:], strict=False):
        inside = [
            record
            for record in records
            if record.get("true_separation_mm") is not None
            and low <= float(record["true_separation_mm"]) < high
        ]
        if not inside:
            continue
        summary = method_summary(inside, method)
        bands.append({**summary, "separation_low_mm": low, "separation_high_mm": high})
    return bands


def summarize(records: list[dict], methods: tuple[str, ...]) -> dict[str, Any]:
    """The whole report: overall, per condition, per correlation regime, paired."""
    conditions = sorted({record["condition"] for record in records})
    per_condition = {
        name: {
            method: method_summary(
                [record for record in records if record["condition"] == name], method
            )
            for method in methods
        }
        for name in conditions
    }
    regimes = sorted({record["correlation"] for record in records})
    return {
        "n_trials": len(records),
        "conditions": conditions,
        "methods": list(methods),
        "overall": {method: method_summary(records, method) for method in methods},
        "by_condition": per_condition,
        "by_correlation": {
            regime: {
                method: method_summary(
                    [record for record in records if record["correlation"] == regime],
                    method,
                )
                for method in methods
            }
            for regime in regimes
        },
        "by_k": {
            str(k): {
                method: method_summary(
                    [record for record in records if record["n_sources"] == k], method
                )
                for method in methods
            }
            for k in sorted({record["n_sources"] for record in records})
        },
        "by_separation": {
            method: by_separation(records, method) for method in methods
        },
        "detection": {method: detection_summary(records, method) for method in methods},
        "detection_by_k": {
            str(k): {
                method: detection_summary(
                    [record for record in records if record["n_sources"] == k], method
                )
                for method in methods
            }
            for k in sorted({record["n_sources"] for record in records})
        },
        "paired": {
            f"{method}_vs_{reference}": paired_comparison(records, method, reference)
            for method, reference in (
                ("hybrid", "gradient"),
                ("hybrid", "gradient_restarts"),
                ("hybrid", "proposal"),
                ("hybrid", "rapmusic"),
                ("hybrid", "scan"),
                ("hybrid", "hybrid_stopgrad"),
                ("proposal", "gradient"),
            )
        },
        "paired_by_k": {
            str(k): {
                f"{method}_vs_{reference}": paired_comparison(
                    [record for record in records if record["n_sources"] == k],
                    method,
                    reference,
                )
                for method, reference in (
                    ("hybrid", "gradient"),
                    ("hybrid", "gradient_restarts"),
                    ("hybrid", "proposal"),
                    ("hybrid", "rapmusic"),
                    ("hybrid", "scan"),
                )
            }
            for k in sorted({record["n_sources"] for record in records})
        },
        "paired_by_correlation": {
            regime: {
                f"{method}_vs_{reference}": paired_comparison(
                    [record for record in records if record["correlation"] == regime],
                    method,
                    reference,
                )
                for method, reference in (
                    ("hybrid", "gradient"),
                    ("hybrid", "proposal"),
                    ("hybrid", "rapmusic"),
                )
            }
            for regime in regimes
        },
        "paired_hard_cases": {
            f"{method}_vs_{reference}": paired_comparison(
                [record for record in records if record["correlation"] != "distinct"],
                method,
                reference,
            )
            for method, reference in (
                ("hybrid", "gradient"),
                ("hybrid", "gradient_restarts"),
                ("hybrid", "proposal"),
                ("hybrid", "rapmusic"),
                ("hybrid", "hybrid_stopgrad"),
            )
        },
    }
