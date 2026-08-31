#!/usr/bin/env python
# Copyright 2026 NeuroLocate contributors
# SPDX-License-Identifier: Apache-2.0
r"""The README loop: one refinement, two initializations, at eight checkpoints.

Eight frames, a fixed camera and fixed geometry. Every frame is a readable static
figure; the only thing that moves is where each estimate currently sits and how
much trail is behind it.

Every number and every position comes from ``results/hybrid_k2_visual.npz``,
which :mod:`scripts.build_hybrid_k2_visual` writes only after checking each arm's
reproduced error and sensor residual against the frozen shard. Nothing here is
recomputed, and no optimizer position is interpolated: the checkpoints are seven
of the twenty-one recorded samples, subsampled.

Output
------
``docs/figures/hybrid_k2_visual.gif``

Usage::

    make k2-visual
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import showcase_style as style  # noqa: E402

#: Optimizer steps to show, all of them recorded samples. Coarse on purpose: the
#: point is progression, not every step.
CHECKPOINTS = (0, 30, 60, 105, 150, 225, 300)

#: Milliseconds per frame, and the long hold on the last one.
FRAME_MS = 780
HOLD_MS = 4200

#: The two arms, in drawing order, with the label each carries.
ARMS = (
    ("gradient", "uninformed initialization", style.UNINFORMED),
    ("hybrid", "learned initialization", style.LEARNED),
)

#: The component strip, styled after ``docs/figures/architecture.png``.
CHIPS = (
    ("proposal", "PyTorch"),
    ("headfield", "OpenMEEG C++"),
    ("refinement", "JAX / Optax"),
)

#: Sagittal window, millimetres. Fixed for every frame. The top is a little
#: above the cortex so the step counter has somewhere to sit that never moves.
VIEW_Y = (-78.0, 102.0)
VIEW_Z = (-20.0, 134.0)


def load(path: Path) -> dict:
    """The replayed trial, with its frozen record parsed out."""
    with np.load(path, allow_pickle=False) as data:
        out = {key: np.asarray(data[key]) for key in data.files if key != "frozen"}
        out["frozen"] = json.loads(str(data["frozen"]))
    return out


class Frames:
    """The figure, and a callable that redraws it at one checkpoint."""

    def __init__(self, data: dict, outline: list[np.ndarray]) -> None:
        """Lay the figure out once. Every element keeps its place in every frame."""
        self.data = data
        self.outline = outline
        self.frozen = data["frozen"]
        self.plt = style.use_agg()

        self.figure = self.plt.figure(figsize=(12.4, 8.0), facecolor=style.PAPER)
        self.brain = self.figure.add_axes((0.015, 0.185, 0.565, 0.680))
        self.side = self.figure.add_axes((0.615, 0.185, 0.370, 0.680))
        self.strip = self.figure.add_axes((0.015, 0.028, 0.970, 0.090))

        self.figure.text(
            0.015, 0.960, "NeuroLocate", fontsize=style.FONT["label"],
            color=style.MUTED, va="center",
        )
        self.figure.text(
            0.015, 0.913, "Same physical refinement, different initialization",
            fontsize=style.FONT["title"], fontweight="bold", color=style.INK,
            va="center",
        )
        self.figure.text(
            0.015, 0.874, "K = 2  ·  shared dynamics  ·  300 refinement steps",
            fontsize=style.FONT["label"], color=style.MUTED, va="center",
        )
        # Reserved on every frame so nothing below it shifts when it fills in.
        self.closing = self.figure.text(
            0.015, 0.145, "", fontsize=style.FONT["label"], color=style.MUTED,
            va="center",
        )

    # -- the stage ---------------------------------------------------------

    def _draw_brain(self, index: int) -> None:
        """Cortex, true sources, and each arm's trail and current estimate."""
        axes = self.brain
        axes.clear()
        style.bare(axes)
        axes.set_xlim(*VIEW_Y)
        axes.set_ylim(*VIEW_Z)
        axes.set_aspect("equal")

        style.draw_cortex(axes, self.outline)

        for name, _, colour in ARMS:
            path = self.data[f"{name}_path_m"][:, :, [1, 2]] * 1e3
            for source in range(path.shape[1]):
                trail = path[: index + 1, source]
                if len(trail) > 1:
                    axes.plot(
                        trail[:, 0], trail[:, 1], color=colour, linewidth=1.7,
                        alpha=0.55, zorder=3, solid_capstyle="round",
                    )
                axes.scatter(
                    path[0, source, 0], path[0, source, 1], marker="x", s=100,
                    color=colour, linewidths=2.0, zorder=4,
                )
                # Above the stars, and ringed in paper: an estimate that has
                # landed on a true source must stay visible on top of it.
                axes.scatter(
                    trail[-1, 0], trail[-1, 1], s=92, color=colour, zorder=8,
                    edgecolors=style.PAPER, linewidths=1.4,
                )

        truth = self.data["truth_m"][:, [1, 2]] * 1e3
        axes.scatter(
            truth[:, 0], truth[:, 1], marker="*", s=350, color=style.INK,
            edgecolors=style.PAPER, linewidths=1.0, zorder=7,
        )

        # Fixed position, fixed width: the counter must not shuffle frame to frame.
        axes.text(
            0.995, 0.995, f"step {int(self.data['steps'][index]):3d} / 300",
            transform=axes.transAxes, ha="right", va="top",
            fontsize=style.FONT["label"], color=style.MUTED, family="monospace",
        )

    # -- the side column ---------------------------------------------------

    def _draw_side(self, final: bool) -> None:
        """The key, and on the last frame the numbers it was all leading to."""
        axes = self.side
        axes.clear()
        style.bare(axes)
        axes.set_xlim(0, 1)
        axes.set_ylim(0, 1)

        marks = (
            ("*", 210, style.INK, "true source"),
            ("x", 95, style.MUTED, "initialization"),
            ("o", 70, style.MUTED, "current estimate"),
        )
        for row, (marker, size, colour, label) in enumerate(marks):
            y = 0.965 - row * 0.072
            extra = (
                {"edgecolors": style.PAPER, "linewidths": 1.0}
                if marker == "o" else {"linewidths": 2.0}
            )
            axes.scatter(0.045, y, marker=marker, s=size, color=colour, **extra)
            axes.text(
                0.135, y, label, va="center", ha="left",
                fontsize=style.FONT["label"], color=style.MUTED,
            )
        for row, (_, label, colour) in enumerate(ARMS):
            y = 0.735 - row * 0.072
            axes.plot([0.020, 0.075], [y, y], color=colour, linewidth=3.0)
            axes.text(
                0.135, y, label, va="center", ha="left",
                fontsize=style.FONT["label"], color=colour,
            )

        if not final:
            return

        methods = self.frozen["methods"]
        rows = (
            ("uninformed + refinement", methods["gradient"]["worst_mm"],
             style.UNINFORMED),
            ("learned + refinement", methods["hybrid"]["worst_mm"], style.LEARNED),
        )
        for row, (label, value, colour) in enumerate(rows):
            y = 0.470 - row * 0.105
            axes.text(
                0.020, y, label, va="center", ha="left",
                fontsize=style.FONT["label"], color=style.INK,
            )
            axes.text(
                1.000, y, f"{value:.1f} mm", va="center", ha="right",
                fontsize=style.FONT["hero"] * 0.60, fontweight="bold", color=colour,
            )

        axes.plot([0.020, 1.000], [0.235, 0.235], color=style.RULE, linewidth=1.0)
        axes.text(
            0.020, 0.170, "sensor residual", va="center", ha="left",
            fontsize=style.FONT["small"], color=style.MUTED,
        )
        for align, (name, _, colour) in zip(("left", "right"), ARMS, strict=True):
            axes.text(
                0.020 if align == "left" else 1.000, 0.078,
                f"{methods[name]['sensor_residual']:.4f}", va="center", ha=align,
                fontsize=style.FONT["caption"], fontweight="bold", color=colour,
            )

    # -- the component strip ----------------------------------------------

    def _draw_strip(self) -> None:
        """Three boxes, drawn like the ones in the architecture figure."""
        axes = self.strip
        axes.clear()
        style.bare(axes)
        axes.set_xlim(0, 1)
        axes.set_ylim(0, 1)

        width, gap = 0.235, 0.055
        left = (1.0 - (len(CHIPS) * width + (len(CHIPS) - 1) * gap)) / 2
        centres = []
        for index, (name, stack) in enumerate(CHIPS):
            x = left + index * (width + gap)
            centres.append(x + width / 2)
            axes.add_patch(
                self.plt.Rectangle(
                    (x, 0.10), width, 0.80, facecolor=style.PAPER,
                    edgecolor=style.INK, linewidth=1.6, zorder=2,
                )
            )
            axes.text(
                x + width / 2, 0.62, name, ha="center", va="center",
                fontsize=style.FONT["label"], fontweight="bold", color=style.INK,
                zorder=3,
            )
            axes.text(
                x + width / 2, 0.34, stack, ha="center", va="center",
                fontsize=style.FONT["small"], color=style.MUTED, zorder=3,
            )
        for index in range(len(centres) - 1):
            axes.annotate(
                "", xy=(centres[index + 1] - width / 2 - 0.006, 0.50),
                xytext=(centres[index] + width / 2 + 0.006, 0.50),
                arrowprops=dict(arrowstyle="-|>", color=style.FAINT, linewidth=1.1,
                                mutation_scale=10),
                zorder=2,
            )

    # -- one frame ---------------------------------------------------------

    def render(self, index: int, *, final: bool) -> None:
        """Draw the whole figure at one recorded checkpoint."""
        self.closing.set_text(
            "Similar sensor fit, different source configuration." if final else ""
        )
        self._draw_brain(index)
        self._draw_side(final)
        self._draw_strip()


def write_gif(frames: list, path: Path, durations: list[int]) -> None:
    """Save the frames onto one shared palette, optimized.

    The palette is built from a montage of every frame rather than from the
    first: the accents only exist once both arms are drawn, and quantizing on
    frame one alone would flatten the whole loop to greyscale.
    """
    from PIL import Image

    width, height = frames[0].size
    montage = Image.new("RGB", (width, height * len(frames)))
    for index, frame in enumerate(frames):
        montage.paste(frame, (0, index * height))
    palette = montage.quantize(colors=128, method=Image.MEDIANCUT)
    quantized = [
        frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    quantized[0].save(
        path, save_all=True, append_images=quantized[1:], duration=durations,
        loop=0, optimize=True,
    )


def main() -> int:
    """Render the eight frames and write the loop."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=REPO_ROOT / "results" / "hybrid_k2_visual.npz"
    )
    parser.add_argument(
        "--cortex", type=Path, default=REPO_ROOT / "results" / "cortex_ico5.npz"
    )
    parser.add_argument(
        "--out", type=Path,
        default=REPO_ROOT / "docs" / "figures" / "hybrid_k2_visual.gif",
    )
    parser.add_argument("--dpi", type=int, default=100)
    arguments = parser.parse_args()

    data = load(arguments.data)
    steps = list(data["steps"])
    for step in CHECKPOINTS:
        if step not in steps:
            raise SystemExit(f"step {step} is not a recorded sample: {steps}")
    indices = [steps.index(step) for step in CHECKPOINTS]

    with np.load(arguments.cortex) as mesh:
        outline = style.cortex_outline(mesh["vertices"])

    from PIL import Image

    plt = style.use_agg()
    with plt.rc_context(style.rc()):
        panel = Frames(data, outline)
        # The last checkpoint is drawn twice: once as the end of the progression,
        # once carrying the result, so the numbers arrive rather than sit there.
        plan = [(index, False) for index in indices] + [(indices[-1], True)]
        images = []
        for index, final in plan:
            panel.render(index, final=final)
            panel.figure.canvas.draw()
            images.append(
                Image.frombytes(
                    "RGBA", panel.figure.canvas.get_width_height(),
                    panel.figure.canvas.buffer_rgba(),
                ).convert("RGB")
            )
        plt.close(panel.figure)

    durations = [FRAME_MS] * (len(images) - 1) + [HOLD_MS]
    write_gif(images, arguments.out, durations)
    size = arguments.out.stat().st_size
    print(
        f"wrote {arguments.out}  —  {len(images)} frames, {images[0].size[0]}x"
        f"{images[0].size[1]}, {size / 1e6:.2f} MB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
