#!/usr/bin/env python
# Copyright 2026 NeuroLocate contributors
# SPDX-License-Identifier: Apache-2.0
"""The frozen K=2 benchmark, as one compact figure.

Two panels, no more. Left: how many of the sources each method actually found at
``K = 2``. Right: what that cost per trial. Both are recomputed from the shards
under ``results/hybrid/shards`` every time this runs, so the figure cannot drift
from the numbers ``make summary`` prints.

The K=4 column of the recovery panel is deliberately present: the composition
does not win there, and a summary figure that showed only K=2 would be choosing
its own result.

Usage::

    make figures
"""

from __future__ import annotations

import argparse
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "app"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import showcase_style as style  # noqa: E402
from neurolayout.hybrid.report import (  # noqa: E402
    RECOVERED_MM,
    detection_summary,
    load_shards,
    method_summary,
)

#: The rows, bottom to top, and the label each carries in the README.
ROWS = (
    ("gradient", "uninformed initialization\n+ refinement"),
    ("rapmusic", "RAP-MUSIC"),
    ("gradient_restarts", "four physical restarts"),
    ("proposal", "learned proposal alone"),
    ("hybrid", "proposal\n+ physical refinement"),
)

#: The two rows the README's argument is about get the accent; the rest are grey.
HIGHLIGHT = {"gradient": style.UNINFORMED, "hybrid": style.LEARNED}
NEUTRAL = "#b9bec7"


def half_up(value: float) -> str:
    """Round half away from zero, so 41.25% prints as 41.3% and not 41.2%.

    Three of these rates are counts out of 80 and land exactly on a tie, where
    Python's round-half-to-even would disagree with what `make summary` prints.
    """
    return str(Decimal(value).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def main() -> int:
    """Draw the figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shards", type=Path, default=REPO_ROOT / "results" / "hybrid" / "shards"
    )
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "docs" / "figures" / "benchmark.png"
    )
    arguments = parser.parse_args()

    records = load_shards(arguments.shards)
    if not records:
        raise SystemExit(f"no shards in {arguments.shards}")
    at_k2 = [record for record in records if record["n_sources"] == 2]
    at_k4 = [record for record in records if record["n_sources"] == 4]

    recovery = [detection_summary(at_k2, name)["recall"] * 100 for name, _ in ROWS]
    recovery_k4 = [detection_summary(at_k4, name)["recall"] * 100 for name, _ in ROWS]
    seconds = [method_summary(records, name)["seconds_median"] for name, _ in ROWS]
    colours = [HIGHLIGHT.get(name, NEUTRAL) for name, _ in ROWS]
    labels = [label for _, label in ROWS]
    y = np.arange(len(ROWS))

    plt = style.use_agg()
    with plt.rc_context(style.rc()):
        figure, (left, right) = plt.subplots(
            1, 2, figsize=(12.2, 4.3), width_ratios=[1.55, 1.0],
            facecolor=style.PAPER,
        )

        left.barh(y, recovery, height=0.62, color=colours, zorder=3)
        for index, (value, deeper) in enumerate(zip(recovery, recovery_k4, strict=True)):
            # A fixed column, so a label never lands on the K=4 marker.
            left.text(
                103, index, f"{half_up(value)}%", va="center", ha="left",
                fontsize=style.FONT["label"], fontweight="bold", color=style.INK,
            )
            left.scatter(
                deeper, index, s=52, facecolors=style.PAPER, edgecolors=style.MUTED,
                linewidths=1.4, zorder=4,
            )
        left.set_yticks(y, labels, fontsize=style.FONT["label"], color=style.INK)
        left.set_xlim(0, 124)
        left.set_xticks([0, 20, 40, 60, 80, 100])
        left.set_ylim(-0.7, len(ROWS) - 0.3)
        left.set_xlabel(
            f"sources recovered, within {RECOVERED_MM:.0f} mm",
            fontsize=style.FONT["label"], color=style.MUTED,
        )
        left.set_title(
            "sources recovered        bars  K = 2        ○  K = 4", loc="left",
            fontsize=style.FONT["label"], color=style.INK, pad=14,
        )
        style.thin_axes(left)
        left.tick_params(axis="y", length=0)
        left.grid(axis="x", color=style.RULE, linewidth=0.8, zorder=0)
        left.set_axisbelow(True)

        right.barh(y, seconds, height=0.62, color=colours, zorder=3)
        for index, value in enumerate(seconds):
            right.text(
                max(seconds) * 1.06, index,
                f"{value:.1f} s" if value >= 0.05 else "< 0.1 s",
                va="center", ha="left", fontsize=style.FONT["label"],
                fontweight="bold", color=style.INK,
            )
        right.set_yticks(y, [""] * len(ROWS))
        right.set_xlim(0, max(seconds) * 1.32)
        right.set_ylim(-0.7, len(ROWS) - 0.3)
        right.set_xlabel(
            "median seconds per trial", fontsize=style.FONT["label"],
            color=style.MUTED,
        )
        right.set_title(
            "inference time", loc="left", fontsize=style.FONT["label"],
            color=style.INK, pad=14,
        )
        style.thin_axes(right)
        right.tick_params(axis="y", length=0)
        right.grid(axis="x", color=style.RULE, linewidth=0.8, zorder=0)
        right.set_axisbelow(True)

        figure.text(
            0.008, 0.028,
            f"{len(records)} trials, 10 conditions, matrix fingerprint "
            "35b6e07a7e130731.  Every method is given K.  Training cost is not "
            "included in the times.",
            fontsize=style.FONT["small"], color=style.MUTED,
        )
        figure.subplots_adjust(left=0.175, right=0.985, top=0.90, bottom=0.20,
                               wspace=0.06)
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(arguments.out, dpi=200, facecolor=style.PAPER)
        plt.close(figure)
    print(f"wrote {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
