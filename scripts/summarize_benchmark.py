#!/usr/bin/env python
# Copyright 2026 NeuroLocate contributors
# SPDX-License-Identifier: Apache-2.0
"""Recompute every number ``docs/BENCHMARK.md`` quotes, from the committed shards.

The shards under ``results/hybrid/shards`` hold one record per trial per method.
This aggregates them and prints the four tables the README and the benchmark
document use, so a reader can check the published numbers against the artifacts
without re-running anything:

* per-source recovery at ``K = 2`` (the README's recovery table);
* the paired comparisons on the correlated and shared-dynamics cells;
* the paired comparison by source count, which is where the ``K = 4`` result is;
* median inference time per trial.

It also writes the full aggregate to ``results/hybrid/summary.json``, which is
the committed file the document was written from.

Usage::

    make summary
    python scripts/summarize_benchmark.py --json /dev/null
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "app"))

from neurolayout.hybrid.benchmark import CONDITIONS, METHODS, fingerprint  # noqa: E402
from neurolayout.hybrid.report import (  # noqa: E402
    RECOVERED_MM,
    load_shards,
    paired_comparison,
    summarize,
)

#: How the report names each method in a table.
LABELS = {
    "rapmusic": "RAP-MUSIC",
    "scan": "OpenMEEG dipole scan",
    "gradient": "uninformed initialization + refinement",
    "gradient_restarts": "four physical restarts",
    "proposal": "learned proposal alone",
    "hybrid": "proposal + physical refinement",
    "hybrid_stopgrad": "proposal (stop-gradient control) + refinement",
}

#: The order the tables read in, worst first.
ORDER = (
    "gradient",
    "rapmusic",
    "scan",
    "gradient_restarts",
    "proposal",
    "hybrid",
    "hybrid_stopgrad",
)


def half_up(value: float, places: int = 1) -> str:
    """Round half away from zero, so 41.25% prints as 41.3% and not 41.2%.

    Python's default rounds ties to even, and three of the recovery rates in this
    matrix land exactly on a tie because they are counts out of 80.
    """
    quantum = Decimal(1).scaleb(-places)
    return str(Decimal(value).quantize(quantum, rounding=ROUND_HALF_UP))


def table(rows: list[tuple[str, ...]], headers: tuple[str, ...]) -> str:
    """A fixed-width text table."""
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))
    ]
    line = "  ".join("-" * w for w in widths)
    out = ["  ".join(h.ljust(w) for h, w in zip(headers, widths, strict=True)), line]
    out += [
        "  ".join(cell.ljust(w) for cell, w in zip(row, widths, strict=True))
        for row in rows
    ]
    return "\n".join(out)


def main() -> int:
    """Aggregate and print."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shards", type=Path, default=REPO_ROOT / "results" / "hybrid" / "shards"
    )
    parser.add_argument(
        "--json", type=Path, default=REPO_ROOT / "results" / "hybrid" / "summary.json"
    )
    arguments = parser.parse_args()

    records = load_shards(arguments.shards)
    if not records:
        raise SystemExit(f"no shards in {arguments.shards}")

    stored = {record["_shard"].get("fingerprint") for record in records}
    if stored != {fingerprint()}:
        raise SystemExit(
            f"the shards were produced for matrix {stored}, but this source "
            f"declares {fingerprint()}"
        )
    expected = sum(condition.n_trials for condition in CONDITIONS)
    if len(records) != expected:
        raise SystemExit(f"{len(records)} trials in the shards, {expected} in the matrix")

    present = tuple(
        method for method in METHODS
        if any(method in record["methods"] for record in records)
    )
    report = summarize(records, present)
    report["fingerprint"] = fingerprint()
    report["numpy_version"] = np.__version__
    report["python"] = platform.python_version()
    for name in ("gradcheck", "batching", "training"):
        path = arguments.shards.parent / f"{name}.json"
        if path.exists():
            report[name] = json.loads(path.read_text())

    print(f"matrix fingerprint  {fingerprint()[:16]}")
    print(f"conditions          {len(report['conditions'])}")
    print(f"trials              {report['n_trials']}")
    print()

    print(f"K = 2 — per-source recovery (an estimate within {RECOVERED_MM:.0f} mm of "
          "the source it was matched with)")
    rows = []
    for method in ORDER:
        entry = report["detection_by_k"]["2"].get(method)
        if not entry or not entry.get("n_trials"):
            continue
        timing = report["by_k"]["2"][method]
        rows.append((
            LABELS[method],
            f"{half_up(entry['recall'] * 100)}%",
            f"{timing['median_mm']:.1f}",
            f"{timing['seconds_median']:.1f}",
        ))
    print(table(rows, ("method", "recovered", "median mm", "median s")))
    print()

    print("Median inference time per trial, every trial in the matrix")
    rows = [
        (LABELS[method], f"{report['overall'][method]['seconds_median']:.1f} s")
        for method in ORDER
        if method in present
    ]
    print(table(rows, ("method", "median s")))
    print()

    hard = [record for record in records if record["correlation"] in ("correlated", "shared")]
    print(f"Correlated and shared-dynamics cells ({len(hard)} trials) — paired "
          "difference, hybrid minus the named method")
    rows = []
    for method in ORDER:
        if method == "hybrid" or method not in present:
            continue
        paired = paired_comparison(hard, "hybrid", method)
        if not paired.get("n"):
            continue
        rows.append((
            LABELS[method],
            str(paired["n"]),
            f"{paired['median_difference_mm']:+.1f}",
            f"{paired['ci_low_mm']:+.1f} to {paired['ci_high_mm']:+.1f}",
            "yes" if paired["excludes_zero"] else "no",
        ))
    print(table(rows, ("vs", "n", "median mm", "95% CI", "excludes zero")))
    print("  negative means the composition is closer to the truth")
    print()

    print("By source count — paired difference, hybrid minus the named method")
    counts = sorted({record["n_sources"] for record in records})
    rows = []
    for method in ORDER:
        if method == "hybrid" or method not in present:
            continue
        cells = []
        for k in counts:
            subset = [record for record in records if record["n_sources"] == k]
            paired = paired_comparison(subset, "hybrid", method)
            cells.append(f"{paired['median_difference_mm']:+.1f}" if paired.get("n") else "—")
        rows.append((LABELS[method], *cells))
    print(table(rows, ("vs", *(f"K={k}" for k in counts))))
    print()

    arguments.json.parent.mkdir(parents=True, exist_ok=True)
    arguments.json.write_text(
        json.dumps(report, indent=1, sort_keys=True, default=str) + "\n"
    )
    print(f"wrote {arguments.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
