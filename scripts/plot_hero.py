#!/usr/bin/env python
# Copyright 2026 NeuroLocate contributors
# SPDX-License-Identifier: Apache-2.0
r"""The README hero: one refinement, two starting points, told as a sequence.

Every number drawn here comes from ``results/hybrid_k2_visual.npz``, which
:mod:`scripts.build_hybrid_k2_visual` writes only after checking each arm's
reproduced error and sensor residual against the frozen shard. This script adds
no computation of its own beyond interpolating between recorded samples.

The clip runs through five stages:

1. the problem — two sources sharing one time course, seen by 64 electrodes;
2. the sources are hidden, and only the traces remain;
3. two starting points appear, one of them a trained network's guess;
4. the same refinement runs from both, through the same OpenMEEG solver;
5. the two answers, 124.3 mm apart in anatomy and 0.0003 apart in data fit.

Outputs
-------
``docs/media/hero.mp4``   the clip, 1920x1080
``docs/figures/hero.gif`` the same clip for the README, 960x540
``docs/figures/hero.png`` its final frame, as a static fallback

Usage::

    make hero
    python scripts/plot_hero.py --no-video
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

FPS = 25

#: Stage boundaries in seconds. The clip is the last one.
SETUP, HIDE, STARTS, REFINE, ANSWER, END = 0.0, 3.2, 4.8, 7.4, 16.6, 22.0

#: One caption at a time, and never more than a line of it.
CAPTIONS = (
    (SETUP, "Two cortical sources, one shared time course, 64 scalp electrodes"),
    (HIDE, "The sources are hidden. Only the 64 traces are given."),
    (STARTS, "Two starting points. One is a trained network's guess."),
    (REFINE, "The same objective, the same OpenMEEG physics, the same 300 steps"),
    (ANSWER, "Same refinement, different start, different answer"),
)

#: The component chips along the bottom, as the gradient crosses them.
CHIPS = (
    ("proposal", "PyTorch", style.LEARNED),
    ("headfield", "OpenMEEG C++", style.INK),
    ("outer loop", "JAX / Optax", style.INK),
)

#: Sagittal window, millimetres, chosen to hold the cortex and both trajectories.
VIEW_Y = (-80.0, 105.0)
VIEW_Z = (-24.0, 126.0)

#: The zoom inset, centred on the true pair.
ZOOM_HALF_MM = 22.0


def load(path: Path) -> dict:
    """The replayed trial, with its frozen record parsed out."""
    with np.load(path, allow_pickle=False) as data:
        out = {key: np.asarray(data[key]) for key in data.files if key != "frozen"}
        out["frozen"] = json.loads(str(data["frozen"]))
    return out


def worst(errors: np.ndarray) -> np.ndarray:
    """``[n_records]`` worst-source error, which is what the benchmark scores."""
    return np.asarray(errors).max(axis=1)


def sample(path: np.ndarray, position: float) -> np.ndarray:
    """``[K, 3]`` linearly interpolated point at a fractional record index."""
    low = int(np.floor(position))
    if low >= len(path) - 1:
        return path[-1]
    weight = position - low
    return path[low] * (1.0 - weight) + path[low + 1] * weight


def caption_for(seconds: float) -> str:
    """The line showing at this moment."""
    current = CAPTIONS[0][1]
    for start, text in CAPTIONS:
        if seconds >= start:
            current = text
    return current


class Hero:
    """The figure, and a callable that redraws it at one moment in the clip."""

    def __init__(self, data: dict, outline: list[np.ndarray]) -> None:
        """Build the layout once; :meth:`render` fills it per frame."""
        self.data = data
        self.outline = outline
        self.frozen = data["frozen"]
        self.n_records = len(data["steps"])
        self.plt = style.use_agg()

        self.figure = self.plt.figure(figsize=(12.8, 7.2), facecolor=style.PAPER)
        self.brain = self.figure.add_axes((0.010, 0.150, 0.560, 0.675))
        self.zoom = self.figure.add_axes((0.028, 0.560, 0.150, 0.250))
        self.eeg = self.figure.add_axes((0.635, 0.215, 0.345, 0.575))
        self.error = self.figure.add_axes((0.635, 0.545, 0.345, 0.245))
        self.residual = self.figure.add_axes((0.635, 0.215, 0.345, 0.215))
        self.key = self.figure.add_axes((0.012, 0.105, 0.556, 0.056))
        self.strip = self.figure.add_axes((0.000, 0.000, 1.000, 0.100))

        self.title = self.figure.text(
            0.015, 0.950, "NeuroLocate", fontsize=style.FONT["title"],
            fontweight="bold", color=style.INK, va="center",
        )
        self.subtitle = self.figure.text(
            0.180, 0.952, "differentiable EEG source localization",
            fontsize=style.FONT["label"], color=style.MUTED, va="center",
        )
        self.caption = self.figure.text(
            0.015, 0.893, "", fontsize=style.FONT["caption"], color=style.INK,
            va="center",
        )
        self.punchline = self.figure.text(
            0.015, 0.848, "", fontsize=style.FONT["label"], color=style.MUTED,
            va="center",
        )
        self.counter = self.figure.text(
            0.980, 0.950, "", fontsize=style.FONT["label"], color=style.MUTED,
            va="center", ha="right", family="monospace",
        )

    # -- the stage ---------------------------------------------------------

    def _draw_brain(self, seconds: float, position: float) -> None:
        """Cortex, scalp, sources and both trajectories."""
        axes = self.brain
        axes.clear()
        style.bare(axes)
        axes.set_xlim(*VIEW_Y)
        axes.set_ylim(*VIEW_Z)
        axes.set_aspect("equal")

        appear = style.fade(0.15, 1.30, seconds)
        style.draw_cortex(axes, self.outline, alpha=0.95 * appear)
        axes.text(
            VIEW_Y[0] + 3, VIEW_Z[0] + 3, "left hemisphere, sagittal",
            ha="left", fontsize=style.FONT["small"], color=style.FAINT,
            alpha=0.9 * appear,
        )

        truth = self.data["truth_m"][:, [1, 2]] * 1e3
        # The sources are visible while the problem is being posed, hollow once
        # they are hidden, and solid again beside the answer.
        solid = style.fade(0.9, 2.1, seconds) * (1.0 - style.fade(HIDE, HIDE + 0.7, seconds))
        solid = max(solid, style.fade(ANSWER, ANSWER + 0.8, seconds))
        outline_only = style.fade(HIDE + 0.2, HIDE + 0.9, seconds)
        for point in truth:
            if outline_only > 0.02:
                axes.scatter(
                    *point, marker="*", s=520, facecolors="none",
                    edgecolors=style.INK, linewidths=1.4, alpha=outline_only, zorder=6,
                )
            if solid > 0.02:
                axes.scatter(
                    *point, marker="*", s=520, color=style.TRUTH, alpha=solid,
                    zorder=7, edgecolors=style.PAPER, linewidths=1.0,
                )

        self._draw_runs(axes, seconds, position, plane=(1, 2), scale=1.0)

        if seconds >= ANSWER:
            reveal = style.fade(ANSWER + 0.3, ANSWER + 1.2, seconds)
            lost = self.data["gradient_final_m"][
                int(np.argmax(self.data["gradient_error_mm"][-1]))
            ][[1, 2]] * 1e3
            axes.text(
                lost[0] + 32, lost[1] + 4,
                f"{worst(self.data['gradient_error_mm'])[-1]:.1f} mm",
                fontsize=style.FONT["hero"] * 0.66, fontweight="bold",
                color=style.UNINFORMED, alpha=reveal, ha="left", va="center",
            )
            axes.plot(
                [lost[0] + 6, lost[0] + 29], [lost[1] + 1, lost[1] + 3],
                color=style.UNINFORMED, linewidth=1.2, alpha=reveal * 0.7,
            )
            axes.text(
                truth[:, 0].mean() + 26, truth[:, 1].mean() + 8,
                f"{worst(self.data['hybrid_error_mm'])[-1]:.1f} mm",
                fontsize=style.FONT["hero"] * 0.66, fontweight="bold",
                color=style.LEARNED, alpha=reveal, ha="left", va="center",
            )
            axes.plot(
                [truth[:, 0].mean() + 6, truth[:, 0].mean() + 24],
                [truth[:, 1].mean() + 3, truth[:, 1].mean() + 7],
                color=style.LEARNED, linewidth=1.2, alpha=reveal * 0.7,
            )

    def _draw_runs(self, axes, seconds: float, position: float, plane, scale) -> None:
        """Both trajectories, grown to ``position``, in the given 2-D plane."""
        horizontal, vertical = plane
        cut = int(np.floor(position))
        starts = style.fade(STARTS + 0.1, STARTS + 0.7, seconds)
        second = style.fade(STARTS + 0.8, STARTS + 1.4, seconds)
        for name, colour, born in (
            ("gradient", style.UNINFORMED, starts),
            ("hybrid", style.LEARNED, second),
        ):
            if born < 0.02:
                continue
            path = self.data[f"{name}_path_m"] * 1e3
            head = sample(self.data[f"{name}_path_m"], position) * 1e3
            for source in range(path.shape[1]):
                drawn = np.vstack([path[: cut + 1, source], head[source]])
                if len(drawn) > 1 and seconds >= REFINE:
                    axes.plot(
                        drawn[:, horizontal], drawn[:, vertical], color=colour,
                        linewidth=2.6 * scale, alpha=0.95, zorder=4,
                        solid_capstyle="round",
                    )
                axes.scatter(
                    path[0, source, horizontal], path[0, source, vertical],
                    marker="x", s=150 * scale, color=colour, alpha=0.9 * born,
                    linewidths=2.4 * scale, zorder=5,
                )
                if seconds >= REFINE:
                    axes.scatter(
                        head[source, horizontal], head[source, vertical],
                        s=130 * scale, color=colour, zorder=6,
                        edgecolors=style.PAPER, linewidths=1.4,
                    )

    # -- the zoom ----------------------------------------------------------

    def _draw_zoom(self, seconds: float, position: float) -> None:
        """A 44 mm window on the true pair: the brain panel cannot show 6.9 mm."""
        axes = self.zoom
        axes.clear()
        alpha = style.fade(REFINE - 0.6, REFINE + 0.4, seconds)
        axes.set_xticks([])
        axes.set_yticks([])
        # Until the card is wanted it must not exist at all: an invisible axis
        # still paints its own background, which would punch a white hole in the
        # cortex behind it.
        axes.patch.set_visible(alpha > 0.02)
        axes.set_facecolor(style.PAPER)
        for spine in axes.spines.values():
            spine.set_visible(alpha > 0.02)
            spine.set_color(style.RULE)
        if alpha < 0.02:
            return

        truth = self.data["truth_m"][:, [1, 2]] * 1e3
        centre = truth.mean(axis=0)
        axes.set_xlim(centre[0] - ZOOM_HALF_MM, centre[0] + ZOOM_HALF_MM)
        axes.set_ylim(centre[1] - ZOOM_HALF_MM, centre[1] + ZOOM_HALF_MM)
        axes.set_aspect("equal")
        self._draw_runs(axes, seconds, position, plane=(1, 2), scale=0.62)
        axes.scatter(
            truth[:, 0], truth[:, 1], marker="*", s=300, color=style.TRUTH,
            edgecolors=style.PAPER, linewidths=1.0, zorder=7, alpha=alpha,
        )
        axes.plot(
            [centre[0] - 16, centre[0] - 6], [centre[1] - 18, centre[1] - 18],
            color=style.MUTED, linewidth=2.0, alpha=alpha,
        )
        axes.text(
            centre[0] - 11, centre[1] - 16.6, "10 mm", ha="center", va="bottom",
            fontsize=style.FONT["small"], color=style.MUTED, alpha=alpha,
        )
        axes.set_title(
            "detail", fontsize=style.FONT["small"], color=style.MUTED, pad=3,
            alpha=alpha,
        )

    # -- the right column --------------------------------------------------

    def _draw_eeg(self, seconds: float) -> None:
        """The 64 observed traces, drawn in and then handed over to the curves."""
        axes = self.eeg
        axes.clear()
        style.bare(axes)
        alpha = style.fade(0.6, 1.8, seconds) * (
            1.0 - style.fade(STARTS + 0.2, STARTS + 1.1, seconds)
        )
        axes.patch.set_visible(alpha >= 0.02)
        if alpha < 0.02:
            return
        traces = self.data["observed"] * 1e6
        n_times = traces.shape[1]
        shown = int(np.clip(style.fade(0.7, 2.6, seconds) * n_times, 1, n_times))
        time = np.arange(n_times)
        for row in traces:
            axes.plot(
                time[:shown], row[:shown], color=style.MUTED, linewidth=0.7,
                alpha=0.45 * alpha,
            )
        axes.set_xlim(0, n_times - 1)
        span = float(np.abs(traces).max()) * 1.15
        axes.set_ylim(-span, span)
        axes.text(
            0.0, 1.04, "observed EEG   64 channels · 32 samples",
            transform=axes.transAxes, fontsize=style.FONT["label"],
            color=style.INK, alpha=alpha,
        )

    def _draw_curves(self, seconds: float, position: float) -> None:
        """Worst-source error and sensor residual, growing with the refinement."""
        alpha = style.fade(STARTS + 0.6, STARTS + 1.6, seconds)
        cut = int(np.floor(position)) + 1
        steps = self.data["steps"]
        panels = (
            (self.error, "worst-source error", "mm", worst, (1.0, 400.0)),
            (self.residual, "sensor residual", "relative",
             lambda values: values, (6e-3, 1.2)),
        )
        for axes, title, unit, reduce_to, limits in panels:
            axes.clear()
            style.thin_axes(axes)
            if alpha < 0.02:
                for spine in axes.spines.values():
                    spine.set_visible(False)
                axes.set_xticks([])
                axes.set_yticks([])
                axes.set_xlabel("")
                axes.patch.set_visible(False)
                continue
            axes.patch.set_visible(True)
            for name, colour in (("gradient", style.UNINFORMED),
                                 ("hybrid", style.LEARNED)):
                key = "error_mm" if unit == "mm" else "residual"
                values = reduce_to(self.data[f"{name}_{key}"])
                axes.plot(steps[:cut], values[:cut], color=colour, linewidth=2.6,
                          alpha=alpha)
                if cut:
                    axes.scatter(steps[cut - 1], values[cut - 1], s=48, color=colour,
                                 zorder=5, edgecolors=style.PAPER, linewidths=1.2,
                                 alpha=alpha)
            axes.set_yscale("log")
            axes.set_xlim(-10, int(steps[-1]) + 10)
            axes.set_ylim(*limits)
            axes.set_ylabel(unit, fontsize=style.FONT["small"], color=style.MUTED)
            axes.text(
                0.0, 1.06, title, transform=axes.transAxes,
                fontsize=style.FONT["label"], color=style.INK, alpha=alpha,
            )
        if alpha >= 0.02:
            self.residual.set_xlabel(
                "optimizer step", fontsize=style.FONT["small"], color=style.MUTED
            )

        if seconds >= ANSWER:
            reveal = style.fade(ANSWER + 0.6, ANSWER + 1.4, seconds)
            for name, colour, shift in (("gradient", style.UNINFORMED, 14),
                                        ("hybrid", style.LEARNED, -20)):
                value = self.data[f"{name}_residual"][-1]
                self.residual.annotate(
                    f"{value:.4f}", xy=(steps[-1], value), xytext=(-8, shift),
                    textcoords="offset points", ha="right", color=colour,
                    fontsize=style.FONT["label"], fontweight="bold", alpha=reveal,
                )

    def _draw_key(self, seconds: float) -> None:
        """What the marks mean, on one line, under the stage."""
        axes = self.key
        axes.clear()
        style.bare(axes)
        axes.set_xlim(0, 1)
        axes.set_ylim(0, 1)
        alpha = style.fade(STARTS, STARTS + 0.8, seconds)
        if alpha < 0.02:
            return
        marks = (
            (0.010, "*", 150, style.TRUTH, "true source"),
            (0.215, "x", 78, style.MUTED, "initialization"),
            (0.410, "o", 52, style.MUTED, "final estimate"),
        )
        for x, marker, size, colour, label in marks:
            extra = (
                {"edgecolors": style.PAPER, "linewidths": 0.8}
                if marker == "o" else {"linewidths": 2.0}
            )
            axes.scatter(
                x, 0.55, marker=marker, s=size, color=colour, alpha=alpha,
                clip_on=False, **extra,
            )
            axes.text(
                x + 0.028, 0.55, label, va="center", ha="left", alpha=alpha,
                fontsize=style.FONT["small"], color=style.MUTED,
            )
        runs = (
            (0.600, style.UNINFORMED, "uninformed initialization"),
            (0.600, style.LEARNED, "learned initialization"),
        )
        for index, (x, colour, label) in enumerate(runs):
            y = 0.78 - index * 0.46
            axes.plot(
                [x, x + 0.032], [y, y], color=colour, linewidth=3.0, alpha=alpha,
                clip_on=False,
            )
            axes.text(
                x + 0.046, y, label, va="center", ha="left", alpha=alpha,
                fontsize=style.FONT["small"], color=colour,
            )

    # -- the component strip ----------------------------------------------

    def _draw_strip(self, seconds: float, position: float) -> None:
        """Three components, and a pulse crossing their boundaries each step.

        The cue the clip is otherwise missing: the refinement above is not one
        program. Every step leaves JAX, enters a container running PyTorch, then
        one running OpenMEEG's C++ solver, and comes back as a gradient.
        """
        axes = self.strip
        axes.clear()
        style.bare(axes)
        axes.set_xlim(0, 1)
        axes.set_ylim(0, 1)
        alpha = style.fade(STARTS + 0.9, STARTS + 1.9, seconds)
        if alpha < 0.02:
            return

        axes.text(
            0.015, 0.62, "Tesseract components", fontsize=style.FONT["small"],
            color=style.MUTED, va="center", alpha=alpha,
        )
        left, width, gap = 0.185, 0.175, 0.055
        chip_bottom, chip_height, lane = 0.40, 0.50, 0.17
        centres = []
        for index, (name, stack, colour) in enumerate(CHIPS):
            x = left + index * (width + gap)
            centres.append(x + width / 2)
            axes.add_patch(
                self.plt.Rectangle(
                    (x, chip_bottom), width, chip_height, facecolor=style.PAPER,
                    edgecolor=style.RULE, linewidth=1.1, alpha=alpha, zorder=2,
                )
            )
            axes.text(
                x + width / 2, chip_bottom + 0.32, name, ha="center", va="center",
                fontsize=style.FONT["small"], fontweight="bold", color=colour,
                alpha=alpha, zorder=3,
            )
            axes.text(
                x + width / 2, chip_bottom + 0.13, stack, ha="center", va="center",
                fontsize=style.FONT["small"], color=style.MUTED, alpha=alpha,
                zorder=3,
            )

        middle = chip_bottom + chip_height / 2
        for index in range(len(centres) - 1):
            axes.annotate(
                "", xy=(centres[index + 1] - width / 2 - 0.008, middle),
                xytext=(centres[index] + width / 2 + 0.008, middle),
                arrowprops=dict(arrowstyle="-|>", color=style.FAINT, linewidth=1.0,
                                mutation_scale=9, alpha=alpha), zorder=2,
            )

        # The pulse runs in its own lane under the chips, so it never sits on a
        # label: forward left to right, then the gradient back the other way.
        if REFINE <= seconds < ANSWER:
            first, last = centres[0] - width / 2, centres[-1] + width / 2
            axes.plot(
                [first, last], [lane, lane], color=style.RULE, linewidth=1.0,
                alpha=alpha, zorder=1,
            )
            span = last - first
            phase = (seconds * 0.8) % 2.0
            if phase < 1.0:
                axes.scatter(
                    first + span * phase, lane, s=62, color=style.INK, zorder=6,
                    alpha=alpha,
                )
            else:
                axes.scatter(
                    last - span * (phase - 1.0), lane, s=62, facecolors="none",
                    edgecolors=style.INK, linewidths=2.0, zorder=6, alpha=alpha,
                )
            axes.text(
                last + 0.020, lane, "forward  •      gradient  ○",
                fontsize=style.FONT["small"], color=style.MUTED, va="center",
                ha="left", alpha=alpha,
            )

    # -- one frame ---------------------------------------------------------

    def render(self, seconds: float) -> None:
        """Draw the whole figure at one moment."""
        span = max(END - REFINE - 5.4, 1e-6)
        progress = float(np.clip((seconds - REFINE) / span, 0.0, 1.0))
        eased = 0.5 - 0.5 * np.cos(np.pi * progress)
        position = eased * (self.n_records - 1)

        self.caption.set_text(caption_for(seconds))
        self.caption.set_color(
            style.INK if seconds < ANSWER else style.INK
        )
        self.caption.set_fontweight("bold" if seconds >= ANSWER else "normal")
        if REFINE <= seconds:
            index = int(np.clip(round(position), 0, self.n_records - 1))
            self.counter.set_text(f"step {int(self.data['steps'][index]):3d} / 300")
        else:
            self.counter.set_text("")
        self.punchline.set_text(
            "the two answers fit the measurement equally well"
            if seconds >= ANSWER + 1.2 else ""
        )

        self._draw_brain(seconds, position)
        self._draw_zoom(seconds, position)
        self._draw_eeg(seconds)
        self._draw_curves(seconds, position)
        self._draw_key(seconds)
        self._draw_strip(seconds, position)


def shrink_gif(path: Path, colours: int = 96) -> None:
    """Re-encode a GIF onto one small shared palette.

    Matplotlib's writer emits full-colour frames, and a flat two-accent design on
    white does not need them: quantizing every frame against one adaptive palette
    and letting Pillow diff the frames cuts the file by more than half with no
    visible change.
    """
    from PIL import Image

    with Image.open(path) as source:
        duration = source.info.get("duration", 80)
        frames = []
        for index in range(getattr(source, "n_frames", 1)):
            source.seek(index)
            frames.append(source.convert("RGB"))
    # The palette has to be built from frames that contain every colour. The
    # first frame is grey — the accents only exist once both runs have started —
    # so sampling it alone would quantize the whole clip to greyscale.
    step = max(1, len(frames) // 8)
    sample = frames[::step] + [frames[-1]]
    width, height = frames[0].size
    montage = Image.new("RGB", (width, height * len(sample)))
    for index, frame in enumerate(sample):
        montage.paste(frame, (0, index * height))
    palette = montage.quantize(colors=colours, method=Image.MEDIANCUT)
    quantized = [frame.quantize(palette=palette, dither=Image.Dither.NONE)
                 for frame in frames]
    quantized[0].save(
        path, save_all=True, append_images=quantized[1:], duration=duration,
        loop=0, optimize=True,
    )
    print(f"  {path.name}: {len(quantized)} frames, "
          f"{path.stat().st_size / 1e6:.1f} MB")


def ffmpeg_writer(fps: int):
    """An ``ffmpeg`` writer, from the PATH or from ``imageio-ffmpeg``."""
    import shutil

    import matplotlib
    from matplotlib.animation import FFMpegWriter

    if shutil.which("ffmpeg") is None:
        try:
            import imageio_ffmpeg
        except ImportError as error:  # pragma: no cover - environment dependent
            raise SystemExit(
                "no ffmpeg on the PATH and imageio-ffmpeg is not installed; "
                "install one of them, or pass --no-video"
            ) from error
        matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    return FFMpegWriter(
        fps=fps, bitrate=-1, codec="libx264",
        extra_args=["-crf", "21", "-preset", "slow", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart"],
    )


def main() -> int:
    """Render the clip, the loop and the still."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=REPO_ROOT / "results" / "hybrid_k2_visual.npz"
    )
    parser.add_argument(
        "--cortex", type=Path, default=REPO_ROOT / "results" / "cortex_ico5.npz"
    )
    parser.add_argument("--figure-dir", type=Path, default=REPO_ROOT / "docs" / "figures")
    parser.add_argument("--media-dir", type=Path, default=REPO_ROOT / "docs" / "media")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--no-gif", action="store_true")
    arguments = parser.parse_args()

    data = load(arguments.data)
    with np.load(arguments.cortex) as mesh:
        outline = style.cortex_outline(mesh["vertices"])

    plt = style.use_agg()
    with plt.rc_context(style.rc()):
        hero = Hero(data, outline)

        still = arguments.figure_dir / "hero.png"
        still.parent.mkdir(parents=True, exist_ok=True)
        hero.render(END - 0.1)
        hero.figure.savefig(still, dpi=150, facecolor=style.PAPER)
        print(f"wrote {still}")

        if arguments.no_video and arguments.no_gif:
            plt.close(hero.figure)
            return 0

        from matplotlib.animation import FuncAnimation

        total = int(round(FPS * END))

        def update(frame: int):
            hero.render(frame / FPS)
            return ()

        if not arguments.no_video:
            video = arguments.media_dir / "hero.mp4"
            video.parent.mkdir(parents=True, exist_ok=True)
            FuncAnimation(hero.figure, update, frames=total, interval=1000 // FPS).save(
                str(video), writer=ffmpeg_writer(FPS), dpi=150
            )
            print(f"wrote {video}")

        if not arguments.no_gif:
            gif = arguments.figure_dir / "hero.gif"
            # Every third frame at half the resolution, then re-encoded onto a
            # small palette: a README loop, not the deliverable. The tail holds,
            # and a GIF pays for every held frame, so it is sampled down to a few.
            moving = list(range(0, int(FPS * (ANSWER + 1.6)), 3))
            held = list(
                np.linspace(int(FPS * (ANSWER + 1.6)), total - 1, 8).astype(int)
            )
            FuncAnimation(
                hero.figure, update, frames=moving + held, interval=3000 // FPS
            ).save(str(gif), writer="pillow", dpi=72)
            shrink_gif(gif)
            print(f"wrote {gif}")

        plt.close(hero.figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
