#!/usr/bin/env python
# Copyright 2026 NeuroLocate contributors
# SPDX-License-Identifier: Apache-2.0
"""The composition: what runs where, what crosses each boundary, and what returns.

Two of the four boxes are Tesseract components, and they are the two that own a
runtime the orchestrator cannot import. The figure's whole job is to make that
legible — which stack, which derivative rule, and where the cotangent enters.

Hand-drawn rather than a graph library: it is four boxes and six arrows, and the
placement is the design.

Usage::

    make figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import showcase_style as style  # noqa: E402

#: The forward path, left to right. ``tesseract`` marks a served component.
BOXES = (
    {
        "x": 0.30, "width": 2.35, "title": "observed EEG",
        "lines": ["64 channels x 32 samples"], "tesseract": False,
    },
    {
        "x": 3.35, "width": 3.75, "title": "proposal",
        "lines": ["PyTorch", "sensor covariance  →  K source positions",
                  "torch.autograd"],
        "tesseract": True,
    },
    {
        "x": 7.80, "width": 3.90, "title": "headfield",
        "lines": ["OpenMEEG C++ symmetric BEM",
                  "K positions and moments  →  scalp potentials",
                  "analytic + finite-difference VJP"],
        "tesseract": True,
    },
)

#: The derivative path is the one thing here worth an accent.
GRADIENT = style.LEARNED


def main() -> int:
    """Draw the figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "docs" / "figures" / "architecture.png"
    )
    arguments = parser.parse_args()

    plt = style.use_agg()
    with plt.rc_context(style.rc()):
        figure, axes = plt.subplots(figsize=(12.0, 4.0), facecolor=style.PAPER)
        axes.set_xlim(0, 12.0)
        axes.set_ylim(0, 4.0)
        axes.axis("off")

        top, height = 2.30, 1.35
        centres = []
        for box in BOXES:
            x, width = box["x"], box["width"]
            centres.append(x + width / 2)
            axes.add_patch(
                plt.Rectangle(
                    (x, top), width, height, facecolor=style.PAPER,
                    edgecolor=style.INK if box["tesseract"] else style.BRAIN_EDGE,
                    linewidth=1.6 if box["tesseract"] else 1.2, zorder=3,
                )
            )
            if box["tesseract"]:
                # A tag rather than a second box: the point is which two of these
                # are served components, and it should read at a glance.
                axes.add_patch(
                    plt.Rectangle(
                        (x, top + height), width, 0.30, facecolor=style.BRAIN_FILL,
                        edgecolor=style.INK, linewidth=1.6, zorder=3,
                    )
                )
                axes.text(
                    x + width / 2, top + height + 0.15,
                    "T E S S E R A C T   C O M P O N E N T", ha="center",
                    va="center", fontsize=style.FONT["small"] - 1.0,
                    color=style.MUTED, zorder=4,
                )
            axes.text(
                x + width / 2, top + height - 0.34, box["title"], ha="center",
                va="center", fontsize=style.FONT["caption"], fontweight="bold",
                color=style.INK, zorder=4,
            )
            for index, line in enumerate(box["lines"]):
                axes.text(
                    x + width / 2, top + height - 0.68 - 0.28 * index, line,
                    ha="center", va="center", fontsize=style.FONT["small"],
                    color=style.MUTED, zorder=4,
                )

        forward = dict(arrowstyle="-|>", color=style.INK, linewidth=1.7,
                       mutation_scale=15)
        backward = dict(arrowstyle="-|>", color=GRADIENT, linewidth=1.9,
                        mutation_scale=15, linestyle=(0, (4, 3)))

        for tail, head in ((2.68, 3.30), (7.13, 7.75)):
            axes.annotate("", xy=(head, top + 0.92), xytext=(tail, top + 0.92),
                          arrowprops=forward, zorder=5)
        for tail, head in ((3.30, 2.68), (7.75, 7.13)):
            axes.annotate("", xy=(head, top + 0.42), xytext=(tail, top + 0.42),
                          arrowprops=backward, zorder=5)

        loop_top, loop_height = 0.42, 0.90
        axes.add_patch(
            plt.Rectangle(
                (0.30, loop_top), 11.40, loop_height, facecolor=style.PAPER,
                edgecolor=style.BRAIN_EDGE, linewidth=1.2, zorder=3,
            )
        )
        axes.text(
            6.00, loop_top + loop_height - 0.32, "JAX / Optax", ha="center",
            va="center", fontsize=style.FONT["caption"], fontweight="bold",
            color=style.INK, zorder=4,
        )
        axes.text(
            6.00, loop_top + loop_height - 0.62,
            "one jax.grad of the sensor residual  ·  imports neither PyTorch "
            "nor OpenMEEG",
            ha="center", va="center", fontsize=style.FONT["small"],
            color=style.MUTED, zorder=4,
        )

        axes.annotate(
            "", xy=(10.60, loop_top + loop_height + 0.02), xytext=(10.60, top - 0.02),
            arrowprops=forward, zorder=5,
        )
        axes.text(
            10.72, (top + loop_top + loop_height) / 2, "predicted EEG",
            fontsize=style.FONT["small"], color=style.MUTED, ha="left", va="center",
        )
        axes.annotate(
            "", xy=(8.70, top - 0.02), xytext=(8.70, loop_top + loop_height + 0.02),
            arrowprops=backward, zorder=5,
        )
        axes.text(
            8.58, (top + loop_top + loop_height) / 2, "gradient",
            fontsize=style.FONT["small"], color=GRADIENT, ha="right", va="center",
        )

        axes.text(
            0.30, 0.12,
            "Each component keeps its own runtime and its own derivative rule. "
            "The gradient crosses both boundaries.",
            fontsize=style.FONT["small"], color=style.MUTED, ha="left", va="center",
        )

        figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(arguments.out, dpi=200, facecolor=style.PAPER,
                       bbox_inches="tight", pad_inches=0.12)
        plt.close(figure)
    print(f"wrote {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
