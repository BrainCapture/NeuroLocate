# Copyright 2026 NeuroLocate contributors
# SPDX-License-Identifier: Apache-2.0
r"""OpenMEEG symmetric-BEM forward model on a cached template head.

This module is the mechanistic core of NeuroLocate. It owns

* :class:`HeadGeometry` — the *portable* description of a head model: three
  nested BEM surfaces, their conductivities, the canonical sensor array and the
  reference operator, all in one documented coordinate frame;
* :class:`OpenMEEGForward` — the thing that actually calls OpenMEEG.

Where the expense goes
----------------------
The symmetric BEM (Kybic et al. 2005) assembles a dense system matrix ``A`` over
all interface vertices and triangles. For the default geometry ``A`` is
:math:`4486 \times 4486`; assembling it costs seconds and factorizing it costs
tens of seconds. None of that depends on the source, so it is done **once,
offline** and collapsed into

.. math::  H = S A^{-1}

where ``S`` is OpenMEEG's sparse sensor-interpolation operator (``Head2EEGMat``).
``H`` is only ``[64, 4486]`` — 2.3 MB — so it ships with the package and the
Tesseract image never rebuilds a BEM system.

What is left per call is genuinely OpenMEEG: for a dipole at an arbitrary
position ``p`` with moment ``m``, ``DipSourceMat`` assembles the source term
``D(p, m)`` by analytic integration over the interface elements, and

.. math::  g(p, m) = R \, H \, D(p, m)

is the referenced 64-channel topography. That assembly is ~1 ms, and — the point
of the whole design — ``p`` is a genuine continuous variable: OpenMEEG places
the dipole exactly where it is asked to, with no nearest-vertex snapping and no
interpolation error.

Coordinate frame
----------------
Everything here is **MNE head coordinates, in metres**: origin at the midpoint
of the auricular points, ``+x`` toward the right ear, ``+y`` toward the nose,
``+z`` up. The fsaverage BEM surfaces are transformed into that frame by the
offline builder; the sensors are natively in it.

Triangle winding
----------------
OpenMEEG's assembly uses the winding convention *opposite* to the right-hand-rule
outward convention that MNE (and most mesh tooling) uses. Handing it
outward-wound triangles produces a solution that is exactly the negative of the
correct one. :func:`openmeeg_triangles` checks the incoming winding and reverses
it; ``tests/test_openmeeg_physics.py`` pins this down by reproducing the analytic
homogeneous-sphere potential from a three-layer BEM with equal conductivities.

OpenMEEG's SWIG bindings hand back views into C++ objects, and a temporary whose
last Python reference dies takes its buffer with it. Every conversion in this
module therefore goes through :func:`_dense`, which keeps the wrapper alive
until the copy is made.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

#: OpenMP threads OpenMEEG is allowed, unless the caller already chose.
#:
#: This is not a tuning knob, it is a bug workaround, and the numbers are worth
#: writing down. A ``DipSourceMat`` assembly for one source's finite-difference
#: probe set (18 dipoles over a 4486-element BEM) takes, on a 48-core host:
#:
#: ===============  =========
#: OMP_NUM_THREADS  time
#: ===============  =========
#: 1                60.7 ms
#: 2                31.8 ms
#: 4                17.0 ms
#: **8**            **9.5 ms**
#: 16               12.6 ms
#: 48 (default)     641.5 ms
#: ===============  =========
#:
#: The assembly is far too small to feed 48 threads, so at the default the run is
#: **67x slower** than at 8 — pure barrier and false-sharing overhead. Since an
#: optimizer step is dominated by exactly this call, leaving the default in place
#: would have made the whole benchmark 30x more expensive for no reason, and the
#: right way to use a 48-core host here is several processes at 8 threads each.
#:
#: Set before OpenMEEG's shared library is loaded, which is why it lives at module
#: import: every path into the solver — the Tesseract component, the offline
#: scripts, the tests — goes through this module first. ``setdefault`` means an
#: explicit choice by the caller always wins.
OPENMP_THREADS = "8"

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, OPENMP_THREADS)

__all__ = [
    "OPENMP_THREADS",
    "HEAD_MODEL_DIR_ENV",
    "SOLVER_ID",
    "SURFACE_NAMES",
    "DEFAULT_CONDUCTIVITIES",
    "HeadGeometry",
    "OpenMEEGForward",
    "openmeeg_triangles",
    "openmeeg_version",
    "signed_volume",
    "is_closed",
    "default_artifact_path",
    "resolve_head_model",
    "load_forward",
]

#: Environment variable naming a directory of extra head-model artifacts.
#:
#: The packaged fsaverage template ships inside the component image. A cached
#: operator for some other anatomy is tens of megabytes, so an alternative one is
#: staged outside the repository and addressed by name — the name travels over
#: the Tesseract boundary as an input field, the bytes never do.
HEAD_MODEL_DIR_ENV = "NEUROLOCATE_HEADMODEL_DIR"

#: Provenance tag written into every artifact. Bump when the physics changes.
SOLVER_ID = "openmeeg-symmetric-bem-v1"

#: BEM interfaces, ordered inner to outer. OpenMEEG's nested-geometry helper
#: names the corresponding domains Brain / Skull / Scalp.
SURFACE_NAMES: tuple[str, str, str] = ("inner_skull", "outer_skull", "outer_skin")

#: Conductivities in S/m for (brain, skull, scalp). MNE's defaults, and the
#: conventional literature values; the 50:1 brain-to-skull ratio is the number
#: that matters most for EEG.
DEFAULT_CONDUCTIVITIES: tuple[float, float, float] = (0.3, 0.006, 0.3)

#: Name of the packaged artifact built by ``scripts/build_openmeeg_headmodel.py``.
ARTIFACT_NAME = "fsaverage_ico3_1005_64.npz"


def openmeeg_version() -> str:
    """Version string of the OpenMEEG library actually loaded."""
    import openmeeg as om

    return str(getattr(om, "__version__", "unknown"))


def _dense(matrix: Any) -> np.ndarray:
    """Copy any OpenMEEG matrix into an owned float64 array.

    ``Matrix(x)`` is a temporary; ``.array()`` is a view into it. Binding the
    temporary to a local before taking the view is what keeps the buffer alive
    long enough for :func:`numpy.array` to copy it.
    """
    import openmeeg as om

    full = matrix if isinstance(matrix, om.Matrix) else om.Matrix(matrix)
    return np.array(full.array(), dtype=np.float64, copy=True)


def signed_volume(vertices: np.ndarray, triangles: np.ndarray) -> float:
    r"""Enclosed volume of a closed surface, signed by its winding.

    By the divergence theorem, :math:`V = \frac{1}{6}\sum (v_0 \times v_1)\cdot v_2`
    over the triangles. Positive means right-hand-rule normals point outward,
    negative means inward. Unlike a "does the normal point away from the
    centroid?" test this is exact for any closed surface, convex or not — and
    real BEM surfaces are not convex.

    Returns:
        Volume in the cube of the vertex units (m³ here).
    """
    corners = vertices[triangles]
    return float(
        np.sum(np.einsum("ij,ij->i", np.cross(corners[:, 0], corners[:, 1]), corners[:, 2]))
        / 6.0
    )


def is_closed(triangles: np.ndarray) -> bool:
    """Whether every edge is shared by exactly two oppositely-wound triangles."""
    edges = np.concatenate(
        [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]], axis=0
    )
    forward = {(int(a), int(b)) for a, b in edges}
    if len(forward) != len(edges):  # a directed edge used twice: inconsistent winding
        return False
    return all((b, a) in forward for a, b in forward)


def openmeeg_triangles(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Re-wind an outward-oriented surface into the winding OpenMEEG expects.

    Args:
        vertices: ``[V, 3]`` surface vertices.
        triangles: ``[F, 3]`` vertex indices, consistently wound so that the
            right-hand-rule normal points outward (MNE's convention).

    Returns:
        ``[F, 3]`` triangles with reversed winding.

    Raises:
        ValueError: If the surface is not closed, or is wound inward, either of
            which would produce a silently sign-flipped or garbage solve.
    """
    if not is_closed(triangles):
        raise ValueError("BEM surface is not a closed, consistently wound manifold")
    volume = signed_volume(vertices, triangles)
    if volume <= 0.0:
        raise ValueError(
            f"BEM surface encloses signed volume {volume:.3e}; triangles must be "
            "wound so right-hand-rule normals point outward"
        )
    return np.ascontiguousarray(triangles[:, ::-1])


@dataclass(frozen=True)
class HeadGeometry:
    """A portable, hashable description of a template head model.

    Everything needed to rebuild the OpenMEEG geometry from scratch, plus the
    canonical sensor array it is paired with. Deliberately free of MNE: the
    Tesseract image carries this and OpenMEEG, and nothing else.

    Attributes:
        vertices: Three ``[V, 3]`` arrays, inner skull → outer skull → scalp, in
            MNE head coordinates, metres, outward right-hand-rule winding.
        triangles: Three ``[F, 3]`` index arrays matching ``vertices``.
        conductivities: ``[3]`` S/m for brain, skull, scalp.
        sensor_xyz: ``[C, 3]`` electrode positions, same frame and units.
        channel_names: ``[C]`` channel names, in the canonical benchmark order.
        source_space: ``[S, 3]`` cortical source-space positions used for
            visualization and for drawing plausible test sources. It is *not*
            a constraint on the differentiable source position.
        source_normals: ``[S, 3]`` unit cortical normals at ``source_space``.
        metadata: Free-form provenance (subject, decimation, library versions).
    """

    vertices: tuple[np.ndarray, ...]
    triangles: tuple[np.ndarray, ...]
    conductivities: np.ndarray
    sensor_xyz: np.ndarray
    channel_names: tuple[str, ...]
    source_space: np.ndarray
    source_normals: np.ndarray
    metadata: dict[str, Any]

    coord_frame: str = "mne-head"
    units: str = "m"

    @property
    def n_channels(self) -> int:
        """Number of EEG channels ``C``."""
        return len(self.channel_names)

    @property
    def n_sources(self) -> int:
        """Size of the reference cortical source space ``S``."""
        return int(self.source_space.shape[0])

    def reference_operator(self) -> np.ndarray:
        """``R = I - 11ᵀ/C`` over this geometry's channel ordering."""
        from neurolayout_shared.source_model import average_reference_operator

        return average_reference_operator(self.n_channels)

    def fingerprint(self) -> str:
        """SHA-256 over everything that changes the BEM system matrix.

        The cached ``H`` operator is only valid for the geometry it was built
        from, so it is stored next to this digest and refused if they disagree.
        Deliberately excludes :attr:`metadata`, which is descriptive only.
        """
        digest = hashlib.sha256()
        digest.update(SOLVER_ID.encode())
        digest.update(self.coord_frame.encode())
        digest.update(self.units.encode())
        for array in (*self.vertices, *self.triangles):
            digest.update(np.ascontiguousarray(array).tobytes())
        digest.update(np.asarray(self.conductivities, dtype=np.float64).tobytes())
        digest.update(np.ascontiguousarray(self.sensor_xyz, dtype=np.float64).tobytes())
        digest.update("\x00".join(self.channel_names).encode())
        return digest.hexdigest()

    def brain_extent(self) -> tuple[np.ndarray, np.ndarray]:
        """Axis-aligned bounding box ``(low, high)`` of the inner-skull surface."""
        inner = self.vertices[0]
        return inner.min(axis=0), inner.max(axis=0)

    def save(self, path: str | Path, sensor_operator: np.ndarray | None = None) -> Path:
        """Write the geometry — and optionally the cached ``H`` — to one ``.npz``.

        ``np.load`` is lazy, so callers that only need the geometry never pay to
        read the operator.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, np.ndarray] = {
            "conductivities": np.asarray(self.conductivities, dtype=np.float64),
            "sensor_xyz": np.asarray(self.sensor_xyz, dtype=np.float64),
            "channel_names": np.array(self.channel_names, dtype="U16"),
            "source_space": np.asarray(self.source_space, dtype=np.float64),
            "source_normals": np.asarray(self.source_normals, dtype=np.float64),
            "surface_names": np.array(SURFACE_NAMES, dtype="U16"),
            "coord_frame": np.array(self.coord_frame),
            "units": np.array(self.units),
            "solver_id": np.array(SOLVER_ID),
            "fingerprint": np.array(self.fingerprint()),
            "metadata": np.array(json.dumps(self.metadata, sort_keys=True)),
        }
        for index, (verts, tris) in enumerate(zip(self.vertices, self.triangles, strict=True)):
            payload[f"vertices_{index}"] = np.asarray(verts, dtype=np.float64)
            payload[f"triangles_{index}"] = np.asarray(tris, dtype=np.int64)
        if sensor_operator is not None:
            payload["sensor_operator"] = np.asarray(sensor_operator, dtype=np.float64)
        np.savez_compressed(path, **payload)
        return path

    @classmethod
    def load(cls, path: str | Path) -> HeadGeometry:
        """Read a geometry written by :meth:`save` (ignoring any cached operator)."""
        with np.load(path, allow_pickle=False) as data:
            geometry = cls(
                vertices=tuple(data[f"vertices_{i}"] for i in range(3)),
                triangles=tuple(data[f"triangles_{i}"] for i in range(3)),
                conductivities=data["conductivities"],
                sensor_xyz=data["sensor_xyz"],
                channel_names=tuple(str(name) for name in data["channel_names"]),
                source_space=data["source_space"],
                source_normals=data["source_normals"],
                metadata=json.loads(str(data["metadata"])),
                coord_frame=str(data["coord_frame"]),
                units=str(data["units"]),
            )
            stored = str(data["fingerprint"])
        if geometry.fingerprint() != stored:
            raise ValueError(
                f"head-model artifact {path} is corrupt: stored fingerprint "
                f"{stored[:12]} != recomputed {geometry.fingerprint()[:12]}"
            )
        return geometry


def load_sensor_operator(path: str | Path, geometry: HeadGeometry) -> np.ndarray | None:
    """Read the cached ``H`` from an artifact, or ``None`` if it holds only geometry."""
    with np.load(path, allow_pickle=False) as data:
        if "sensor_operator" not in data:
            return None
        if str(data["fingerprint"]) != geometry.fingerprint():
            raise ValueError(f"cached sensor operator in {path} belongs to another geometry")
        return np.asarray(data["sensor_operator"], dtype=np.float64)


class OpenMEEGForward:
    """The OpenMEEG-backed forward operator for one :class:`HeadGeometry`.

    Construction is cheap (it only builds the OpenMEEG geometry object and the
    sensor operator, ~1 s). Building ``H`` from scratch is *not* cheap and only
    happens when no cached operator is supplied — see
    :meth:`build_sensor_operator`.

    Args:
        geometry: The head model.
        sensor_operator: Cached ``H = S A^{-1}``, shape ``[C, n]``. Built on
            first use when omitted.
        reference: Apply the average-reference operator to every returned gain.
            The stored epochs are average-referenced, so the forward has
            to be too; turning it off is for diagnostics only.
    """

    #: Entries kept in the gain memo. One optimizer step touches two distinct
    #: position sets (the current point and its finite-difference probes), so a
    #: handful is enough and the memory stays a few hundred kilobytes.
    _GAIN_MEMO_SIZE = 8

    def __init__(
        self,
        geometry: HeadGeometry,
        sensor_operator: np.ndarray | None = None,
        *,
        reference: bool = True,
    ) -> None:
        """Build the OpenMEEG geometry and sensor operator for ``geometry``."""
        import openmeeg as om

        self.geometry = geometry
        self.reference = reference
        self._reference_operator = geometry.reference_operator()

        meshes = [
            (
                np.ascontiguousarray(verts, dtype=np.float64),
                openmeeg_triangles(verts, tris),
            )
            for verts, tris in zip(geometry.vertices, geometry.triangles, strict=True)
        ]
        self._om = om
        self._geom = om.make_nested_geometry(meshes, list(np.asarray(geometry.conductivities)))
        if not self._geom.is_nested():
            raise ValueError("OpenMEEG rejected the BEM surfaces as non-nested")
        # Held as attributes, not locals: OpenMEEG objects that go out of scope
        # take their buffers with them.
        self._sensor_matrix = om.Matrix(np.asfortranarray(geometry.sensor_xyz))
        self._sensors = om.Sensors(self._sensor_matrix, self._geom)
        self._head2eeg = om.Head2EEGMat(self._geom, self._sensors)
        self._gain_memo: dict[tuple[Any, bytes], np.ndarray] = {}
        self._operator = None if sensor_operator is None else np.asarray(sensor_operator)
        if self._operator is not None and self._operator.shape[0] != geometry.n_channels:
            raise ValueError(
                f"cached sensor operator has {self._operator.shape[0]} rows but the "
                f"geometry has {geometry.n_channels} channels"
            )

    #
    # The expensive, source-independent half
    #

    def build_sensor_operator(self) -> np.ndarray:
        r"""Assemble and factorize the BEM system, returning ``H = S A^{-1}``.

        ``A`` is symmetric but indefinite, so this uses a general LU solve
        (LAPACK via NumPy) rather than OpenMEEG's own ``invert()``: it is the
        same answer an order of magnitude faster, and it solves only the ``C``
        right-hand sides that are actually needed instead of forming a full
        inverse.

        Returns:
            ``[C, n]`` operator, where ``n`` is the symmetric-BEM system size.
        """
        head_matrix = self._om.HeadMat(self._geom)
        system = _dense(head_matrix)
        sensor = _dense(self._head2eeg)
        # H = S A^{-1}  <=>  A^T H^T = S^T, and A is symmetric.
        self._operator = np.linalg.solve(system, sensor.T).T
        return self._operator

    @property
    def sensor_operator(self) -> np.ndarray:
        """``H``, building it on first access if it was not supplied."""
        if self._operator is None:
            self.build_sensor_operator()
        assert self._operator is not None
        return self._operator

    @property
    def system_size(self) -> int:
        """Symmetric-BEM system size ``n`` (interface vertices plus triangles)."""
        return int(self.sensor_operator.shape[1])

    #
    # The cheap, per-source half — this is what the optimizer calls
    #

    def source_term(self, dipoles: np.ndarray) -> np.ndarray:
        """OpenMEEG's source term for ``[P, 6]`` (position, moment) rows."""
        dipoles = np.ascontiguousarray(dipoles, dtype=np.float64)
        if dipoles.ndim != 2 or dipoles.shape[1] != 6:
            raise ValueError(f"dipoles must be [P, 6], got {dipoles.shape}")
        matrix = self._om.Matrix(np.asfortranarray(dipoles))
        return _dense(self._om.DipSourceMat(self._geom, matrix, "Brain"))

    def gain(self, positions: np.ndarray) -> np.ndarray:
        """Free-orientation gain at each source position.

        Repeated calls at the same positions are served from a small memo. This is
        not a micro-optimization: one Adam step evaluates the gain at the *current*
        positions three times over — once for the primal, once inside the VJP to
        build the time-course cotangent, and once more whenever the trajectory is
        recorded — and each evaluation is an OpenMEEG source assembly costing tens
        of milliseconds per dipole. The map from positions to gain is a pure
        function of the (immutable) geometry, so the memo cannot go stale.

        Args:
            positions: ``[P, 3]`` dipole positions, metres, head frame. Must lie
                inside the inner-skull surface; OpenMEEG assigns each to the
                ``Brain`` domain and will produce nonsense for points outside it.

        Returns:
            ``[C, P, 3]``: entry ``(c, p, j)`` is the potential at channel ``c``
            for a unit dipole at ``positions[p]`` oriented along axis ``j``, in
            volts per A·m, average-referenced unless ``reference=False``. The
            returned array is read-only, because it may be shared with the memo.
        """
        positions = np.asarray(positions, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(f"positions must be [P, 3], got {positions.shape}")
        key = (positions.shape, np.ascontiguousarray(positions).tobytes())
        cached = self._gain_memo.get(key)
        if cached is not None:
            return cached
        gain = self._gain_uncached(positions)
        gain.flags.writeable = False
        if len(self._gain_memo) >= self._GAIN_MEMO_SIZE:
            # Plain FIFO eviction: the access pattern is "the same few position
            # sets, then move on", so recency and insertion order agree.
            self._gain_memo.pop(next(iter(self._gain_memo)))
        self._gain_memo[key] = gain
        return gain

    def _gain_uncached(self, positions: np.ndarray) -> np.ndarray:
        """:meth:`gain` without the memo — one real OpenMEEG source assembly."""
        n_points = positions.shape[0]
        dipoles = np.empty((3 * n_points, 6), dtype=np.float64)
        dipoles[:, :3] = np.repeat(positions, 3, axis=0)
        dipoles[:, 3:] = np.tile(np.eye(3), (n_points, 1))
        gain = self.sensor_operator @ self.source_term(dipoles)  # [C, 3P]
        if self.reference:
            gain = self._reference_operator @ gain
        return gain.reshape(self.geometry.n_channels, n_points, 3)


def default_artifact_path() -> Path:
    """Location of the head-model artifact shipped inside this package."""
    return Path(__file__).resolve().parent / "artifacts" / ARTIFACT_NAME


def resolve_head_model(name: str) -> Path:
    """Locate a named head-model artifact outside the package.

    Args:
        name: Artifact stem or file name, e.g. ``"fsaverage_ico4_1005_64"``.
            A name is never a path: allowing one would let a served component be
            pointed at an arbitrary file on its host.

    Returns:
        The artifact path.

    Raises:
        ValueError: If ``name`` contains a path separator or ``..``.
        FileNotFoundError: If the directory is unset or the artifact is absent.
    """
    if "/" in name or "\\" in name or ".." in name:
        raise ValueError(f"head-model name {name!r} must be a bare name, not a path")
    directory = os.environ.get(HEAD_MODEL_DIR_ENV)
    if not directory:
        raise FileNotFoundError(
            f"head model {name!r} was requested but {HEAD_MODEL_DIR_ENV} is not set. "
            "Point it at the directory the head-model artifacts were built into."
        )
    path = Path(directory).expanduser() / (name if name.endswith(".npz") else f"{name}.npz")
    if not path.exists():
        raise FileNotFoundError(f"no head-model artifact {path}")
    return path


@lru_cache(maxsize=8)
def load_forward(
    path: str | None = None, *, reference: bool = True, name: str | None = None
) -> OpenMEEGForward:
    """Load the packaged, a named, or a given head model, memoized per process.

    The Tesseract endpoints call this on every request; the whole point of the
    cached ``H`` is that this is cheap after the first call. ``name`` resolves
    through :func:`resolve_head_model`; ``path`` is for offline tooling that
    already knows where the artifact is.
    """
    if name is not None and path is not None:
        raise ValueError("pass either a head-model name or a path, not both")
    if name is not None:
        resolved = resolve_head_model(name)
    else:
        resolved = Path(path) if path is not None else default_artifact_path()
    if not resolved.exists():
        raise FileNotFoundError(
            f"no OpenMEEG head-model artifact at {resolved}. Build one with "
            "`python scripts/build_openmeeg_headmodel.py` (needs mne + nibabel "
            "and downloads fsaverage)."
        )
    geometry = HeadGeometry.load(resolved)
    return OpenMEEGForward(
        geometry, load_sensor_operator(resolved, geometry), reference=reference
    )
