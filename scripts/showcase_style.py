# Copyright 2026 NeuroLocate contributors
# SPDX-License-Identifier: Apache-2.0
"""One visual language for the three figures the README shows.

Light paper, charcoal type, two accent colours and nothing else. The accents
carry the only distinction the figures are about — which initialization a
trajectory started from — so no other element is allowed to use them.

Everything here is 2-D. A translucent 3-D cortex is slow to render, hard to read
at README width, and by now the default look for this kind of figure; a flat
sagittal silhouette is faster, sharper and legible when GitHub scales it down.
"""

from __future__ import annotations

import numpy as np

#: Paper and type. The warm off-white is the one `docs/figures/architecture.png`
#: uses, so the README's figures share a ground.
PAPER = "#fbf9f7"
INK = "#14161a"
MUTED = "#767b85"
FAINT = "#a8adb6"
RULE = "#dcdfe4"

#: The cortex silhouette. Deliberately quiet: it is a reference frame, not data.
BRAIN_FILL = "#eceef1"
BRAIN_EDGE = "#c9ccd2"

#: The two accents, and the only place they are allowed. `UNINFORMED` is the
#: refinement started from an uninformed point; `LEARNED` is the same refinement
#: started from the proposal network's output.
UNINFORMED = "#d9482b"
LEARNED = "#127b86"

#: A true source is charcoal, never an accent: it belongs to neither run.
TRUTH = INK

#: Point sizes at dpi 100, scaled by the figure's own dpi where it matters.
FONT = {
    "hero": 34.0,
    "title": 20.0,
    "caption": 17.0,
    "label": 13.5,
    "small": 11.5,
}

#: Helvetica metrics, with the two most likely substitutes behind it.
FAMILY = ["TeX Gyre Heros", "Nimbus Sans", "Liberation Sans", "DejaVu Sans"]


def rc() -> dict:
    """Matplotlib settings every showcase figure opens with."""
    return {
        "font.family": FAMILY,
        "mathtext.fontset": "stixsans",
        "text.color": INK,
        "axes.edgecolor": RULE,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": FONT["small"],
        "ytick.labelsize": FONT["small"],
        "figure.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "axes.grid": False,
        "legend.frameon": False,
    }


def use_agg():
    """Import pyplot on the non-interactive backend and return it."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


#: The sagittal plane, as axis indices into head-frame ``(x, y, z)``.
#: ``y`` is anterior, ``z`` is superior — so this is a left-lateral view, which
#: is the side both true sources of the frozen trial are on.
SAGITTAL = (1, 2)


def cortex_outline(
    vertices_m: np.ndarray, *, cell_mm: float = 1.5, smooth: float = 2.0
) -> list[np.ndarray]:
    """Closed 2-D paths tracing the cortex in the sagittal plane, millimetres.

    Built by occupancy rather than by a hull: the cortical surface projected onto
    one plane is a filled region, so a smoothed 2-D histogram of its vertices has
    the shape of a brain, and its half-maximum contour is the silhouette. A
    convex hull would give an egg.

    Args:
        vertices_m: ``[V, 3]`` cortical vertices, head frame, metres.
        cell_mm: Grid pitch of the occupancy map.
        smooth: Gaussian smoothing width, in cells.

    Returns:
        One or more ``[N, 2]`` arrays of ``(anterior, superior)`` millimetres.
    """
    from scipy.ndimage import gaussian_filter

    horizontal, vertical = SAGITTAL
    points = np.asarray(vertices_m, dtype=np.float64)[:, [horizontal, vertical]] * 1e3
    low = points.min(axis=0) - 4.0
    high = points.max(axis=0) + 4.0
    bins = [
        np.arange(low[axis], high[axis] + cell_mm, cell_mm) for axis in (0, 1)
    ]
    density, x_edges, y_edges = np.histogram2d(points[:, 0], points[:, 1], bins=bins)
    density = gaussian_filter(density, smooth)

    plt = use_agg()
    figure = plt.figure()
    axes = figure.add_subplot()
    centres = [(edge[:-1] + edge[1:]) / 2 for edge in (x_edges, y_edges)]
    contour = axes.contour(
        centres[0], centres[1], density.T, levels=[0.10 * density.max()]
    )
    paths = [
        np.asarray(segment)
        for segment in contour.allsegs[0]
        if len(segment) > 40
    ]
    plt.close(figure)
    return paths


def draw_cortex(axes, paths: list[np.ndarray], *, alpha: float = 1.0) -> None:
    """Fill the silhouette on a 2-D axis, in millimetres."""
    for path in paths:
        axes.fill(
            path[:, 0], path[:, 1], facecolor=BRAIN_FILL, edgecolor=BRAIN_EDGE,
            linewidth=1.1, zorder=1, alpha=alpha,
        )


def bare(axes) -> None:
    """Strip an axis to nothing: no frame, no ticks, paper background."""
    axes.set_facecolor(PAPER)
    axes.set_xticks([])
    axes.set_yticks([])
    for spine in axes.spines.values():
        spine.set_visible(False)


def thin_axes(axes) -> None:
    """A quiet plot frame: two rules, small ticks, no grid."""
    axes.set_facecolor(PAPER)
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(RULE)
    axes.tick_params(length=3, width=0.8, colors=MUTED, labelsize=FONT["small"])


def fade(start: float, end: float, position: float) -> float:
    """Smooth 0 to 1 ramp for ``position`` between ``start`` and ``end``."""
    if position <= start:
        return 0.0
    if position >= end:
        return 1.0
    fraction = (position - start) / (end - start)
    return float(0.5 - 0.5 * np.cos(np.pi * fraction))
