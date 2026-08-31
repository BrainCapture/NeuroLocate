#!/usr/bin/env python
# Copyright 2026 NeuroLocate contributors
# SPDX-License-Identifier: Apache-2.0
r"""The K=2 figure and animation: one refinement, two initializations.

Draws one deterministic trial of ``h-k2-shared-close`` — two sources 15.2 mm
apart sharing a single time course at 20 dB — under two runs that differ in their
starting point and in nothing else. Same objective, same optimizer, same 300
steps through the same OpenMEEG BEM.

``uninformed initialization``
    The continuous OpenMEEG refinement from the uninformed starting point the
    frozen benchmark has always used. It recovers one source and ends 124.3 mm
    from the other.
``learned initialization``
    The same loop, started from the proposal network's output. Both estimates
    stay near the truth; the worse one ends 6.9 mm away.

The two runs end at a sensor residual of 0.0117 and 0.0120 — the anatomically
poor answer fits the measurement very slightly *better*. That is what the figure
shows: at K=2 on a shared time course, similar EEG fit corresponds to very
different source anatomy, so the answer is not identified by the data alone and
the starting point decides which one is returned.

Outputs
-------
``docs/figures/hybrid_k2_visual.png``
    The static comparison: cortex, both trajectories, a zoom on the true pair,
    the two curves against optimizer step, and the objective along the line
    joining the two converged answers.
``docs/media/hybrid_k2_visual.mp4``
    The same scene as an 18-second clip, fixed camera, no narration, ending on
    the finished comparison.
``docs/figures/hybrid_k2_visual.gif``
    A smaller, shorter loop for a README.

Every number comes from ``results/hybrid_k2_visual.npz``, which
:mod:`scripts.build_hybrid_k2_visual` writes only after checking each method's
reproduced error and residual against the frozen shard.

Usage::

    python scripts/build_hybrid_k2_visual.py     # once; replays the trial
    python scripts/plot_hybrid_k2_visual.py
    python scripts/plot_hybrid_k2_visual.py --no-video
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import brainstyle as style  # noqa: E402

#: The two arms, in drawing order, with the label each one carries everywhere.
ARMS = (
    ("gradient", "uninformed initialization", style.METHOD_COLORS["single"]),
    ("hybrid", "learned initialization", style.METHOD_COLORS["hybrid"]),
)

#: Left-lateral view. Both true sources are in the left hemisphere, so this is
#: the near side; the whole trajectory stays in front of the cortex.
VIEW = (12.0, -165.0)

#: Half-width of the zoom inset, millimetres, and the plane it is drawn in as
#: axis indices into ``(x, y, z)`` = ``(right, anterior, superior)``.
ZOOM_HALF_WIDTH_MM = 26.0
ZOOM_PLANE = (1, 2)


def load(path: Path) -> dict:
    """The replayed trial, with its frozen record parsed out."""
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files if key != "frozen"}
        arrays["frozen"] = json.loads(str(data["frozen"]))
    return arrays


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


#
# The cortex panel
#


def draw_scene(
    axis, data: dict, cortex: dict, *, upto: float | None = None,
    annotate: bool = True,
) -> None:
    """Both trajectories on the cortex, grown up to a fractional record index.

    Args:
        axis: A 3-D axis.
        data: The replayed trial.
        cortex: The mesh.
        upto: How far along the optimization to draw, as a fractional index into
            the recorded samples; ``None`` draws all of it.
        annotate: Label the true pair and the lost source once the run is
            complete.
    """
    n_records = len(data["steps"])
    position = float(n_records - 1) if upto is None else float(np.clip(upto, 0, n_records - 1))

    extra = np.concatenate(
        [data["gradient_path_m"].reshape(-1, 3), data["hybrid_path_m"].reshape(-1, 3)]
    )
    style.draw_cortex(axis, cortex, alpha=0.28)

    for name, _, color in ARMS:
        path = data[f"{name}_path_m"] * 1e3
        cut = int(np.floor(position))
        head = sample(data[f"{name}_path_m"], position) * 1e3
        for source in range(path.shape[1]):
            drawn = np.vstack([path[: cut + 1, source], head[source]])
            axis.plot(
                drawn[:, 0], drawn[:, 1], drawn[:, 2],
                color=color, linewidth=9.0, alpha=0.16, zorder=6,
                solid_capstyle="round",
            )
            axis.plot(
                drawn[:, 0], drawn[:, 1], drawn[:, 2],
                color=color, linewidth=3.0, alpha=1.0, zorder=7,
                solid_capstyle="round",
            )
            # The sampled optimizer positions themselves, growing with progress
            # so the direction of travel is readable without an arrow.
            if cut:
                sizes = 18.0 + 44.0 * np.linspace(0.0, 1.0, cut + 1) ** 2
                axis.scatter(
                    path[: cut + 1, source, 0], path[: cut + 1, source, 1],
                    path[: cut + 1, source, 2], s=sizes, color=color,
                    edgecolors="none", alpha=0.85, depthshade=False, zorder=8,
                )
            axis.scatter(
                *path[0, source], s=180, marker="X", color=color, alpha=0.95,
                edgecolors=style.BACKGROUND, linewidths=1.0, depthshade=False, zorder=9,
            )
            if position >= n_records - 1:
                style.glow(axis, head[source] * 1e-3, color, size=150)

    for truth in data["truth_m"]:
        style.glow(axis, truth, style.TRUE_COLOR, size=460, marker="*")

    style.frame_head(axis, cortex, view=VIEW, zoom=1.55, extra=extra)
    if annotate and position >= n_records - 1:
        lost = data["gradient_final_m"][
            int(np.argmax(data["gradient_error_mm"][-1]))
        ]
        _label_point(
            axis, data["truth_m"].mean(axis=0), "true sources",
            style.TRUE_COLOR, offset=(0.00, 0.075),
        )
        _label_point(
            axis, lost, f"{worst(data['gradient_error_mm'])[-1]:.0f} mm from truth",
            style.METHOD_COLORS["single"], offset=(0.02, -0.065),
        )


def _label_point(axis, point, text: str, color: str, *, offset) -> None:
    """A short label next to a 3-D point, stroked so it reads over the cortex."""
    from matplotlib import patheffects
    from mpl_toolkits.mplot3d import proj3d

    x, y, _ = proj3d.proj_transform(*(np.asarray(point) * 1e3), axis.get_proj())
    label = axis.text2D(
        *(np.asarray(axis.transLimits.transform((x, y))) + np.asarray(offset)),
        text, transform=axis.transAxes, color=color, fontsize=style.FONT["label"],
        ha="center", va="center", zorder=12, fontweight="bold",
    )
    label.set_path_effects(
        [patheffects.withStroke(linewidth=3.5, foreground=style.BACKGROUND)]
    )


#
# The zoom on the true pair
#


def draw_zoom(
    axis, data: dict, *, upto: float | None = None, caption: bool = True
) -> None:
    """A ±26 mm sagittal window on the two true sources.

    The cortex panel shows a 124 mm miss; it cannot show a 6.9 mm one. This does,
    at one fixed scale, with the same marks and the same colours.
    """
    horizontal, vertical = ZOOM_PLANE
    truth = data["truth_m"] * 1e3
    centre = truth.mean(axis=0)
    n_records = len(data["steps"])
    position = float(n_records - 1) if upto is None else float(np.clip(upto, 0, n_records - 1))

    for name, _, color in ARMS:
        path = data[f"{name}_path_m"] * 1e3
        cut = int(np.floor(position))
        head = sample(data[f"{name}_path_m"], position) * 1e3
        for source in range(path.shape[1]):
            drawn = np.vstack([path[: cut + 1, source], head[source]])
            axis.plot(
                drawn[:, horizontal], drawn[:, vertical], color=color,
                linewidth=1.8, alpha=0.9, zorder=4,
            )
            axis.scatter(
                path[0, source, horizontal], path[0, source, vertical], s=140,
                marker="X", color=color, alpha=0.95, edgecolors=style.BACKGROUND,
                linewidths=0.8, zorder=5,
            )
            axis.scatter(
                head[source, horizontal], head[source, vertical], s=130, color=color,
                edgecolors="white", linewidths=1.0, zorder=6,
            )
    axis.scatter(
        truth[:, horizontal], truth[:, vertical], s=420, marker="*",
        facecolors=style.TRUE_COLOR, edgecolors=style.BACKGROUND, linewidths=1.2,
        zorder=7,
    )

    axis.set_xlim(centre[horizontal] - ZOOM_HALF_WIDTH_MM, centre[horizontal] + ZOOM_HALF_WIDTH_MM)
    axis.set_ylim(centre[vertical] - ZOOM_HALF_WIDTH_MM, centre[vertical] + ZOOM_HALF_WIDTH_MM)
    axis.set_aspect("equal")
    axis.set_facecolor(style.BACKGROUND)
    for spine in axis.spines.values():
        spine.set_color(style.PANEL_EDGE)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_xlabel("anterior →", fontsize=style.FONT["small"], color=style.MUTED, labelpad=3)
    axis.set_ylabel("superior →", fontsize=style.FONT["small"], color=style.MUTED, labelpad=3)
    axis.set_title(
        "zoom on the true pair", color=style.TEXT, fontsize=style.FONT["panel"], pad=7
    )

    if caption:
        axis.text(
            0.5, -0.14,
            f"the uninformed run's second estimate is "
            f"{worst(data['gradient_error_mm'])[-1]:.0f} mm outside this window",
            transform=axis.transAxes, ha="center", va="top",
            color=style.MUTED, fontsize=style.FONT["small"],
        )

    bar = 10.0
    x0 = centre[horizontal] - ZOOM_HALF_WIDTH_MM * 0.84
    y0 = centre[vertical] - ZOOM_HALF_WIDTH_MM * 0.86
    axis.plot([x0, x0 + bar], [y0, y0], color="white", linewidth=2.4, zorder=8)
    axis.text(
        x0 + bar / 2, y0 + 1.4, "10 mm", color="white", fontsize=style.FONT["small"],
        ha="center", va="bottom", zorder=8,
    )


#
# The curves
#


def draw_curves(
    error_axis, residual_axis, data: dict, *, upto: float | None = None
) -> None:
    """Worst-source error and sensor residual against optimizer step."""
    steps = data["steps"]
    n_records = len(steps)
    position = float(n_records - 1) if upto is None else float(np.clip(upto, 0, n_records - 1))
    cut = int(np.floor(position)) + 1

    for name, label, color in ARMS:
        per_source = np.sort(data[f"{name}_error_mm"], axis=1)
        errors, residual = per_source[:, -1], data[f"{name}_residual"]
        # Dashed: the source each run got closest to. At K=2 on a shared time
        # course a run can find one source and lose the other, and the worst-case
        # line alone would hide that.
        error_axis.plot(
            steps[:cut], per_source[:cut, 0], color=color, linewidth=1.4,
            linestyle="--", alpha=0.8,
        )
        error_axis.plot(steps[:cut], errors[:cut], color=color, linewidth=2.6, label=label)
        residual_axis.plot(steps[:cut], residual[:cut], color=color, linewidth=2.6)
        if cut:
            for axis, values in ((error_axis, errors), (residual_axis, residual)):
                axis.scatter(
                    steps[cut - 1], values[cut - 1], s=52, color=color, zorder=6,
                    edgecolors="white", linewidths=0.9,
                )

    for axis, title, unit in (
        (error_axis, "localization error", "worst source, mm"),
        (residual_axis, "sensor residual", "relative"),
    ):
        axis.set_yscale("log")
        axis.set_xlim(-8, int(steps[-1]) + 8)
        style.style_axes(axis, title=title)
        axis.set_ylabel(unit, color=style.MUTED, fontsize=style.FONT["small"])
        axis.set_xlabel("optimizer step", color=style.MUTED, fontsize=style.FONT["small"])

    error_axis.set_ylim(0.75, 460.0)
    error_axis.set_xlabel(
        "optimizer step\nsolid: worse source · dashed: better source",
        color=style.MUTED, fontsize=style.FONT["small"],
    )
    residual_axis.set_ylim(6e-3, 1.4)
    if cut >= n_records:
        # One label for both, because the two curves land on top of each other:
        # that coincidence is the whole point of the panel.
        residual_axis.annotate(
            "  ".join(f"{data[f'{name}_residual'][-1]:.4f}" for name, _, _ in ARMS),
            xy=(steps[-1], data["hybrid_residual"][-1]), xytext=(-6, 16),
            textcoords="offset points", color=style.TEXT,
            fontsize=style.FONT["small"], ha="right", fontweight="bold",
        )


def draw_barrier(axis, data: dict) -> None:
    """The objective on the segment joining the two converged answers."""
    fraction = data["barrier_fraction"]
    residual = data["barrier_residual"]
    axis.plot(fraction, residual, color=style.MUTED, linewidth=2.2, zorder=3)
    for index, (_, _, color) in zip((0, -1), ARMS, strict=True):
        axis.scatter(
            fraction[index], residual[index], s=90, color=color, zorder=5,
            edgecolors="white", linewidths=1.0,
        )
        axis.annotate(
            f"{residual[index]:.4f}",
            xy=(fraction[index], residual[index]),
            xytext=(8 if index == 0 else -8, 15),
            textcoords="offset points", color=color, fontsize=style.FONT["small"],
            ha="left" if index == 0 else "right",
        )
    style.style_axes(axis, title="sensor residual between the two answers")
    axis.set_xlim(-0.04, 1.04)
    axis.set_ylim(
        float(residual.min()) - 0.0012, float(residual.max()) + 0.0016
    )
    axis.set_xticks([0.0, 1.0])
    axis.set_xticklabels(
        ["uninformed answer", "learned answer"], fontsize=style.FONT["small"]
    )
    axis.set_ylabel("relative residual", color=style.MUTED, fontsize=style.FONT["small"])


def draw_legend(axis) -> None:
    """What the marks mean, and which colour is which arm."""
    axis.set_axis_off()
    axis.set_facecolor(style.BACKGROUND)
    marks = [
        ("*", style.TRUE_COLOR, "true source", 300),
        ("X", style.MUTED, "initialization", 130),
        ("o", style.MUTED, "final estimate", 100),
    ]  # the four labels the figure uses, and no others
    for index, (marker, color, label, size) in enumerate(marks):
        y = 0.90 - index * 0.19
        axis.scatter(
            0.07, y, marker=marker, s=size, color=color, transform=axis.transAxes,
            edgecolors="white" if marker == "o" else style.BACKGROUND,
            linewidths=0.8, clip_on=False,
        )
        axis.text(
            0.17, y, label, transform=axis.transAxes, va="center",
            fontsize=style.FONT["label"], color=style.TEXT,
        )
    for index, (_, label, color) in enumerate(ARMS):
        y = 0.25 - index * 0.19
        axis.plot(
            [0.03, 0.12], [y, y], color=color, linewidth=3.2,
            transform=axis.transAxes, clip_on=False,
        )
        axis.text(
            0.17, y, label, transform=axis.transAxes, va="center",
            fontsize=style.FONT["label"], color=color,
        )


#
# The static hero
#


def figure_hero(data: dict, cortex: dict, out: Path) -> None:
    """The static K=2 hero."""
    plt = style.use_agg()
    frozen = data["frozen"]

    figure = plt.figure(figsize=(16.6, 9.0), facecolor=style.BACKGROUND)
    grid = figure.add_gridspec(
        3, 3, width_ratios=[1.08, 0.86, 0.98], height_ratios=[1.0, 1.0, 0.88],
        wspace=0.26, hspace=0.70, left=0.012, right=0.965, top=0.845, bottom=0.135,
    )

    axis = figure.add_subplot(grid[:, 0], projection="3d", facecolor=style.BACKGROUND)
    draw_scene(axis, data, cortex)

    draw_zoom(figure.add_subplot(grid[0:2, 1]), data)
    draw_legend(figure.add_subplot(grid[2, 1]))
    draw_curves(
        figure.add_subplot(grid[0, 2]), figure.add_subplot(grid[1, 2]), data
    )
    draw_barrier(figure.add_subplot(grid[2, 2]), data)

    figure.suptitle(
        "Same physical refinement, two initializations  (K = 2)",
        color=style.TEXT, fontsize=style.FONT["title"], y=0.975,
    )
    figure.text(
        0.5, 0.915,
        f"two sources {frozen['true_separation_mm']:.1f} mm apart · one shared "
        f"time course · 64 channels · {frozen['snr_db']:.0f} dB SNR · "
        "300 refinement steps",
        color=style.MUTED, fontsize=style.FONT["subtitle"], ha="center",
    )

    single = frozen["methods"]["gradient"]
    hybrid = frozen["methods"]["hybrid"]
    figure.text(
        0.5, 0.072,
        f"uninformed initialization {single['worst_mm']:.1f} mm    "
        f"·    learned initialization {hybrid['worst_mm']:.1f} mm",
        color=style.TEXT, fontsize=style.FONT["value"], ha="center", fontweight="bold",
    )
    figure.text(
        0.5, 0.028,
        f"Sensor residual {single['sensor_residual']:.4f} against "
        f"{hybrid['sensor_residual']:.4f}. Similar EEG fit, different source "
        "anatomy: the data alone does not pick one.",
        color=style.MUTED, fontsize=style.FONT["label"], ha="center",
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=160, facecolor=style.BACKGROUND)
    plt.close(figure)
    print(f"wrote {out}")


#
# The animation
#


def build_animation_figure(data: dict, cortex: dict):
    """The animated layout, and a callable that redraws it at one frame."""
    plt = style.use_agg()
    frozen = data["frozen"]
    n_records = len(data["steps"])

    figure = plt.figure(figsize=(16.0, 9.0), facecolor=style.BACKGROUND)
    brain = figure.add_axes((0.00, 0.09, 0.55, 0.79), projection="3d")
    zoom = figure.add_axes((0.045, 0.485, 0.165, 0.295))
    error_axis = figure.add_axes((0.655, 0.575, 0.315, 0.255))
    residual_axis = figure.add_axes((0.655, 0.185, 0.315, 0.255))
    for axis in (brain, zoom, error_axis, residual_axis):
        axis.set_facecolor(style.BACKGROUND)

    figure.suptitle(
        "Same physical refinement, two initializations  (K = 2)",
        color=style.TEXT, fontsize=style.FONT["title"], y=0.968,
    )
    figure.text(
        0.5, 0.905,
        f"two sources {frozen['true_separation_mm']:.1f} mm apart · one shared "
        f"time course · 64 channels · {frozen['snr_db']:.0f} dB SNR",
        color=style.MUTED, fontsize=style.FONT["subtitle"], ha="center",
    )
    figure.text(
        0.030, 0.078, "★  true source", color=style.TEXT,
        fontsize=style.FONT["label"], ha="left",
    )
    for index, (_, label, color) in enumerate(ARMS):
        figure.text(
            0.175 + index * 0.185, 0.078, f"—  {label}", color=color,
            fontsize=style.FONT["label"], ha="left",
        )
    figure.text(
        0.030, 0.032,
        "Position gradients cross the OpenMEEG C++ head model at every step.",
        color=style.MUTED, fontsize=style.FONT["label"], ha="left",
    )
    readout = {
        "step": figure.text(
            0.700, 0.078, "", color=style.TEXT, fontsize=style.FONT["value"],
            ha="left", fontweight="bold",
        ),
        "gradient": figure.text(
            0.860, 0.078, "", color=style.METHOD_COLORS["single"],
            fontsize=style.FONT["value"], ha="right", fontweight="bold",
        ),
        "hybrid": figure.text(
            0.970, 0.078, "", color=style.METHOD_COLORS["hybrid"],
            fontsize=style.FONT["value"], ha="right", fontweight="bold",
        ),
    }
    verdict = figure.text(
        0.970, 0.032, "", color=style.MUTED, fontsize=style.FONT["label"], ha="right",
    )
    residuals = tuple(frozen["methods"][name]["sensor_residual"] for name, _, _ in ARMS)

    def render(position: float, *, final: bool) -> None:
        for axis in (brain, zoom, error_axis, residual_axis):
            axis.clear()
        draw_scene(brain, data, cortex, upto=position, annotate=final)
        draw_zoom(zoom, data, upto=position, caption=False)
        draw_curves(error_axis, residual_axis, data, upto=position)
        zoom.set_title(
            "zoom on the true pair", color=style.TEXT,
            fontsize=style.FONT["small"], pad=4,
        )
        index = int(np.clip(round(position), 0, n_records - 1))
        errors = {name: worst(data[f"{name}_error_mm"])[index] for name, _, _ in ARMS}
        readout["step"].set_text(f"step {int(data['steps'][index]):3d}")
        for name, _, _ in ARMS:
            readout[name].set_text(f"{errors[name]:.1f} mm")
        verdict.set_text(
            f"Sensor residual {residuals[0]:.4f} against {residuals[1]:.4f}: "
            "similar EEG fit, different anatomy."
            if final
            else "worst-source error, by initialization"
        )

    return figure, render, n_records


def write_animation(
    data: dict,
    cortex: dict,
    video: Path | None,
    gif: Path | None,
    *,
    fps: int = 25,
    seconds: float = 18.0,
) -> None:
    """Render the loop once and write it to every requested format."""
    from matplotlib.animation import FuncAnimation

    figure, render, n_records = build_animation_figure(data, cortex)
    total = int(round(fps * seconds))
    hold_in, hold_out = int(fps * 1.2), int(fps * 6.5)
    growing = total - hold_in - hold_out

    def position(frame: int) -> tuple[float, bool]:
        """``(fractional record index, is the run complete)`` for one frame.

        The clip ends on the finished comparison and holds there, rather than
        walking back to the start: the last frame is the one a still preview
        should freeze on.
        """
        if frame < hold_in:
            return 0.0, False
        if frame < hold_in + growing:
            # Eased, so the fast early steps are not a blur and the long tail is
            # not dead time.
            fraction = (frame - hold_in) / max(growing - 1, 1)
            return (0.5 - 0.5 * np.cos(np.pi * fraction)) * (n_records - 1), False
        return float(n_records - 1), True

    def update(frame: int):
        where, final = position(frame)
        render(where, final=final)
        return ()

    animation = FuncAnimation(figure, update, frames=total, interval=1000 // fps)

    if video is not None:
        video.parent.mkdir(parents=True, exist_ok=True)
        animation.save(str(video), writer=_ffmpeg_writer(fps), dpi=120)
        print(f"wrote {video}")
    if gif is not None:
        gif.parent.mkdir(parents=True, exist_ok=True)
        # Every other frame, half the resolution, and a closing hold sampled down
        # to a handful of frames: a README preview, not the deliverable. The hold
        # is identical frame after identical frame, and a GIF pays full price for
        # each one.
        moving = list(range(0, hold_in + growing, 2))
        held = list(np.linspace(hold_in + growing, total - 1, 10).astype(int))
        preview = FuncAnimation(
            figure, update, frames=moving + held, interval=2000 // fps
        )
        preview.save(str(gif), writer="pillow", dpi=52)
        print(f"wrote {gif}")

    import matplotlib.pyplot as plt

    plt.close(figure)


def _ffmpeg_writer(fps: int):
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
    # Quality-targeted rather than bitrate-targeted: the scene is mostly flat
    # black, so a fixed bitrate would spend megabytes on nothing and the file has
    # to live in the repository.
    return FFMpegWriter(
        fps=fps, bitrate=-1, codec="libx264",
        extra_args=[
            "-crf", "23", "-preset", "slow", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
        ],
    )


def main() -> int:
    """Draw the hero, then the animation."""
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
    parser.add_argument("--seconds", type=float, default=18.0)
    arguments = parser.parse_args()

    data = load(arguments.data)
    cortex = style.load_cortex(arguments.cortex)

    figure_hero(data, cortex, arguments.figure_dir / "hybrid_k2_visual.png")
    if not (arguments.no_video and arguments.no_gif):
        write_animation(
            data,
            cortex,
            None if arguments.no_video else arguments.media_dir / "hybrid_k2_visual.mp4",
            None if arguments.no_gif else arguments.figure_dir / "hybrid_k2_visual.gif",
            seconds=arguments.seconds,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
