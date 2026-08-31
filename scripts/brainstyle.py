# Copyright 2026 NeuroLocate contributors
# SPDX-License-Identifier: Apache-2.0
"""One visual language for every figure drawn on the cortex.

:mod:`scripts.plot_brain_views` and :mod:`scripts.plot_hybrid_k2_visual` draw
different results on the same anatomy, so the marks have to mean the same thing
in both: white star is a true source, a filled disc in the method's colour is an
estimate, a cross is an initialization, and the cortex is the same translucent
grey seen from the same side. Everything shared lives here rather than being
copied, so a change to the palette cannot reach one figure and miss another.

Type sizes are set once, in :data:`FONT`, against one requirement: every mark
carrying information stays legible when the figure is displayed at half size.
Nothing meaningful is smaller than :data:`FONT` ``["small"]``.
"""

from __future__ import annotations

import numpy as np

#: Page colours.
BACKGROUND = "#07070c"
CORTEX_FACE = "#39405e"
PANEL_EDGE = "#3a3f55"
GRIDLINE = "#22263a"
TEXT = "#e8ebf5"
MUTED = "#a3aac2"

#: The source grid, wherever candidate locations are drawn.
GRID_COLOR = "#4a5068"

#: A curve that belongs to no estimator — a residual, a loss, a landscape.
CURVE_COLOR = "#8fa0c8"

#: A true source is always white, in every figure.
TRUE_COLOR = "#ffffff"

#: Estimator colours, unchanged from the benchmark figures so a reader moving
#: between the two sets is not relearning the legend.
METHOD_COLORS = {
    "dspm": "#c86bd6",
    "irmxne": "#f0a202",
    "scan": "#9aa0b5",
    "neurolocate": "#4dd6c1",
    # The hybrid figures. `single` is the refinement from the uninformed
    # initialization and `hybrid` the proposal-initialized one; `proposal` is
    # the network's own output, before any physics.
    "single": "#ff6b6b",
    "hybrid": "#4dd6c1",
    "proposal": "#a78bfa",
}

#: The one type scale. Sizes are points at ``dpi=160``.
FONT = {
    "title": 21.0,
    "subtitle": 13.0,
    "panel": 14.0,
    "value": 17.0,
    "label": 12.0,
    "small": 11.0,
}

#: The sequential map used wherever a continuous per-location value is painted.
#: Only ever used with a colourbar.
BRAIN_COLORS = [
    (0.00, "#12142a"),
    (0.35, "#3b2f8f"),
    (0.60, "#3f7fd0"),
    (0.80, "#37c2b0"),
    (1.00, "#eafff6"),
]


def use_agg():
    """Return ``pyplot`` with a non-interactive backend and the shared defaults."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update(
        {
            "font.size": FONT["label"],
            "text.color": TEXT,
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "figure.facecolor": BACKGROUND,
            "savefig.facecolor": BACKGROUND,
        }
    )
    return plt


def brain_cmap():
    """The sequential colormap for painted per-location values."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("brain", BRAIN_COLORS)


def load_cortex(path) -> dict[str, np.ndarray]:
    """The ico5 cortical surface, in head coordinates, metres.

    The vertex order is the source space's own — the same order the OpenMEEG gain
    dictionary and every reported position index into — so a per-location value
    can be painted straight onto the mesh with no resampling.
    """
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def shade(vertices: np.ndarray, triangles: np.ndarray, light=(0.5, 0.3, 0.8)) -> np.ndarray:
    """Lambertian shading per face, so the folds read as folds."""
    corners = vertices[triangles]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.clip(lengths, 1e-30, None)
    direction = np.asarray(light, dtype=float)
    direction = direction / np.linalg.norm(direction)
    return 0.35 + 0.65 * np.clip(normals @ direction, 0.0, 1.0)


def draw_cortex(
    axis,
    cortex: dict[str, np.ndarray],
    *,
    values: np.ndarray | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    alpha: float = 0.30,
    hemi: str = "both",
):
    """Paint the cortical surface, optionally coloured by a per-location value.

    Args:
        axis: A 3-D axis.
        cortex: The mesh from :func:`load_cortex`.
        values: ``[20484]`` per-location values, or ``None`` for a neutral cortex.
            When given, ``vmin``/``vmax`` must be given too: an unexplained
            colour on a brain is worse than no colour, so a painted surface is
            only ever drawn together with the colourbar its limits belong to.
        vmin: Low end of the colour scale.
        vmax: High end.
        alpha: Face opacity. The surface is deliberately translucent: every source
            in this problem is *inside* the brain, and an opaque cortex would hide
            every marker the figure exists to show.
        hemi: ``"lh"``, ``"rh"`` or ``"both"``.

    Returns:
        The ``ScalarMappable`` behind the painted colours, or ``None``.
    """
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize, to_rgba
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    vertices, triangles = cortex["vertices"] * 1e3, cortex["triangles"]
    n_lh = int(cortex["n_lh"])
    if hemi == "lh":
        triangles = triangles[triangles[:, 0] < n_lh]
    elif hemi == "rh":
        triangles = triangles[triangles[:, 0] >= n_lh]

    lighting = shade(vertices, triangles)
    mappable = None
    if values is None:
        colors = np.tile(np.asarray(to_rgba(CORTEX_FACE)), (len(triangles), 1))
        colors[:, :3] *= lighting[:, None]
    else:
        if vmin is None or vmax is None:
            raise ValueError("a painted cortex needs explicit colour limits")
        face = np.asarray(values)[triangles].mean(axis=1)
        norm = Normalize(vmin=vmin, vmax=vmax)
        mappable = ScalarMappable(norm=norm, cmap=brain_cmap())
        colors = mappable.to_rgba(face)
        colors[:, :3] *= (0.55 + 0.45 * lighting)[:, None]
    colors[:, 3] = alpha

    axis.add_collection3d(
        Poly3DCollection(vertices[triangles], facecolors=colors, edgecolors="none")
    )
    return mappable


def frame_head(
    axis,
    cortex: dict[str, np.ndarray],
    *,
    view=(14.0, 18.0),
    zoom: float = 1.55,
    pad: float = 0.04,
    extra: np.ndarray | None = None,
) -> None:
    """Equal-aspect limits around the cortex, with the axes off.

    The default ``view`` is the right-lateral one the K=1 figures use; the hybrid
    figures look from the left, because their sources are in the left hemisphere
    and the near side is the one worth showing.

    Args:
        axis: A 3-D axis.
        cortex: The mesh from :func:`load_cortex`.
        view: ``(elevation, azimuth)`` in degrees.
        zoom: Fill factor inside the axes.
        pad: Fractional margin around the bounding box.
        extra: Further points, metres, that must stay inside the frame — an
            initialization outside the cortex, for instance.
    """
    vertices = cortex["vertices"] * 1e3
    if extra is not None and len(extra):
        vertices = np.concatenate([vertices, np.asarray(extra) * 1e3])
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    centre, span = 0.5 * (low + high), (high - low) * (1.0 + pad)
    axis.set_xlim(centre[0] - span[0] / 2, centre[0] + span[0] / 2)
    axis.set_ylim(centre[1] - span[1] / 2, centre[1] + span[1] / 2)
    axis.set_zlim(centre[2] - span[2] / 2, centre[2] + span[2] / 2)
    # The box aspect follows the data's own bounding box rather than a cube, so
    # the head fills the panel instead of floating in two bands of empty black,
    # and one millimetre is still one millimetre along every axis.
    axis.set_box_aspect(tuple(span / span.max()), zoom=zoom)
    axis.view_init(elev=view[0], azim=view[1])
    axis.set_axis_off()
    axis.set_facecolor(BACKGROUND)


def glow(axis, point: np.ndarray, color: str, *, size: float = 60.0, marker: str = "o"):
    """A marker with a halo, so a single dipole reads through a glass cortex."""
    point = np.asarray(point, dtype=float) * 1e3
    for scale, opacity in ((7.0, 0.10), (3.2, 0.18)):
        axis.scatter(
            *point, s=size * scale, marker=marker, color=color, alpha=opacity,
            edgecolors="none", depthshade=False, zorder=9,
        )
    axis.scatter(
        *point, s=size * 1.45, marker=marker, color=BACKGROUND, alpha=0.75,
        edgecolors="none", depthshade=False, zorder=9,
    )
    return axis.scatter(
        *point, s=size, marker=marker, color=color, edgecolors="white",
        linewidths=1.0, depthshade=False, zorder=10,
    )


def style_axes(axis, *, title: str | None = None) -> None:
    """The shared look for a 2-D panel: dark, thin, unobtrusive."""
    axis.set_facecolor(BACKGROUND)
    axis.tick_params(colors=MUTED, labelsize=FONT["small"])
    for spine in axis.spines.values():
        spine.set_color(PANEL_EDGE)
    axis.grid(True, color=GRIDLINE, linewidth=0.6)
    axis.set_axisbelow(True)
    if title is not None:
        axis.set_title(title, color=TEXT, fontsize=FONT["panel"], pad=7)


def topomap(axis, observed: np.ndarray, sensor_xyz: np.ndarray) -> None:
    """The measurement as a scalp map: 64 numbers at the peak sample.

    Azimuthal-equidistant projection about the vertex, MNE's own topomap layout.
    """
    from matplotlib.patches import Circle
    from scipy.interpolate import griddata

    observed = np.asarray(observed)
    peak = observed[:, int(np.argmax(np.abs(observed).max(axis=0)))]
    sensor_xyz = np.asarray(sensor_xyz)

    offset = sensor_xyz - sensor_xyz.mean(axis=0)
    radius = np.linalg.norm(offset, axis=1)
    theta = np.arccos(np.clip(offset[:, 2] / radius, -1.0, 1.0))
    phi = np.arctan2(offset[:, 1], offset[:, 0])
    x, y = theta * np.cos(phi), theta * np.sin(phi)

    extent = 1.05 * float(np.abs(np.concatenate([x, y])).max())
    grid_x, grid_y = np.mgrid[-extent:extent:200j, -extent:extent:200j]
    values = griddata((x, y), peak, (grid_x, grid_y), method="cubic")
    values[grid_x**2 + grid_y**2 > extent**2] = np.nan

    limit = float(np.nanmax(np.abs(values)))
    axis.imshow(
        values.T, origin="lower", extent=(-extent, extent, -extent, extent),
        cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="bilinear",
    )
    axis.scatter(x, y, s=9, color="white", edgecolors=BACKGROUND, linewidths=0.4, zorder=5)
    axis.add_patch(
        Circle((0, 0), extent, fill=False, edgecolor="white", linewidth=1.2, zorder=6)
    )
    axis.set_xlim(-extent * 1.10, extent * 1.10)
    axis.set_ylim(-extent * 1.10, extent * 1.10)
    axis.set_aspect("equal")
    axis.set_axis_off()
