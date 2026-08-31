# Copyright 2026 NeuroLocate contributors
# SPDX-License-Identifier: Apache-2.0
"""Deterministic construction and caching of the dense scalp lead field.

The differentiable ``headfield`` Tesseract never solves the volume-conduction
problem itself. It interpolates a *dense* lead field that has been evaluated
once on a fixed scalp lattice. This module owns that artifact:

* :class:`HeadModelSpec` is a hashable description of the geometry.
* :class:`HeadModel` is the built artifact (scalp lattice, source basis, lead
  field, normalization scale).
* :func:`get_head_model` builds and memoizes it.

Right now the solver behind it is the analytic sphere in
:mod:`neurolayout_shared.sphere_model`. Phase 2 swaps that call for an OpenMEEG
BEM solve and writes the result to an ``.npz`` (see
:func:`save_head_model` / :func:`load_head_model`). Nothing downstream of this
module has to change, because everything downstream only ever sees
``scalp_directions`` and ``lead_field``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .geometry import fibonacci_directions, normalize
from .sphere_model import sphere_lead_field

__all__ = [
    "DEFAULT_SPEC",
    "HeadModel",
    "HeadModelSpec",
    "build_head_model",
    "get_head_model",
    "load_head_model",
    "save_head_model",
]

#: Solver tag recorded in metadata. Bump when the physics behind the lead field
#: changes so cached artifacts can never be silently mixed up.
SOLVER_ID = "sphere-homogeneous-legendre-v1"


@dataclass(frozen=True)
class HeadModelSpec:
    """Hashable description of the head geometry and source basis.

    Attributes:
        n_scalp: Number of scalp lattice vertices ``J`` used for interpolation.
        n_sources: Number of cortical dipoles ``S`` in the source basis.
        radius: Outer (scalp) sphere radius, in metres.
        source_radius_fraction: Source shell radius as a fraction of ``radius``.
        sigma: Uniform conductivity, in S/m.
        n_terms: Legendre truncation degree of the analytic solver.
    """

    n_scalp: int = 642
    n_sources: int = 64
    radius: float = 0.09
    source_radius_fraction: float = 0.75
    sigma: float = 0.33
    n_terms: int = 80


@dataclass(frozen=True)
class HeadModel:
    """A built dense lead field plus the geometry it was evaluated on.

    Attributes:
        spec: The spec this artifact was built from.
        scalp_directions: ``[J, 3]`` unit scalp directions.
        source_positions: ``[S, 3]`` dipole positions, metres.
        source_moments: ``[S, 3]`` dipole moments (radial unit vectors).
        lead_field: ``[J, S]`` RMS-normalized lead field.
        lead_field_scale: RMS of the raw physical lead field, in V per unit
            source amplitude. Multiply ``lead_field`` by this to recover
            physical units.
        solver_id: Provenance tag of the solver that produced the lead field.
    """

    spec: HeadModelSpec
    scalp_directions: np.ndarray
    source_positions: np.ndarray
    source_moments: np.ndarray
    lead_field: np.ndarray
    lead_field_scale: float
    solver_id: str

    @property
    def n_scalp(self) -> int:
        """Scalp lattice size ``J`` of the built lead field."""
        return self.lead_field.shape[0]

    @property
    def n_sources(self) -> int:
        """Source-basis size ``S`` of the built lead field."""
        return self.lead_field.shape[1]

    def metadata(self) -> dict:
        """JSON-serializable provenance record."""
        return {
            "solver_id": self.solver_id,
            "lead_field_scale": float(self.lead_field_scale),
            "n_scalp": self.n_scalp,
            "n_sources": self.n_sources,
            **asdict(self.spec),
        }


def build_head_model(spec: HeadModelSpec) -> HeadModel:
    """Evaluate the dense lead field for ``spec``.

    The scalp lattice and the source lattice both use spherical Fibonacci
    sampling, but the source lattice is rotated by half a golden angle so that
    no source sits exactly under a scalp vertex. That avoids a degenerate
    ``cos(gamma) = 1`` column that would dominate the interpolation weights.
    """
    scalp_directions = fibonacci_directions(spec.n_scalp)

    source_directions = fibonacci_directions(spec.n_sources)
    # Rotate the source lattice about z by half a golden angle (deterministic).
    half_golden = 0.5 * np.pi * (3.0 - np.sqrt(5.0))
    cos_a, sin_a = np.cos(half_golden), np.sin(half_golden)
    rotation = np.array(
        [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]]
    )
    source_directions = normalize(source_directions @ rotation.T)

    source_radius = spec.source_radius_fraction * spec.radius
    source_positions = source_radius * source_directions
    # Radially oriented dipoles: the standard first-order model of the
    # macroscopic pyramidal-cell current that surface EEG is sensitive to.
    source_moments = source_directions.copy()

    raw = sphere_lead_field(
        scalp_directions,
        source_positions,
        source_moments,
        radius=spec.radius,
        sigma=spec.sigma,
        n_terms=spec.n_terms,
    )
    scale = float(np.sqrt(np.mean(raw**2)))
    if not np.isfinite(scale) or scale == 0.0:
        raise RuntimeError(f"degenerate lead field (rms={scale}) for spec {spec}")

    return HeadModel(
        spec=spec,
        scalp_directions=scalp_directions,
        source_positions=source_positions,
        source_moments=source_moments,
        lead_field=raw / scale,
        lead_field_scale=scale,
        solver_id=SOLVER_ID,
    )


#: The geometry every component defaults to. A module-level singleton so it is
#: shared by reference and safe as a function default.
DEFAULT_SPEC = HeadModelSpec()


@lru_cache(maxsize=4)
def get_head_model(spec: HeadModelSpec = DEFAULT_SPEC) -> HeadModel:
    """Memoized :func:`build_head_model`.

    The Tesseract endpoints call this on every request, so the (cheap but not
    free) lead-field evaluation must happen exactly once per process.
    """
    return build_head_model(spec)


def save_head_model(model: HeadModel, path: str | Path) -> Path:
    """Write a built head model to a compressed ``.npz``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        scalp_directions=model.scalp_directions,
        source_positions=model.source_positions,
        source_moments=model.source_moments,
        lead_field=model.lead_field,
        lead_field_scale=np.array(model.lead_field_scale),
        solver_id=np.array(model.solver_id),
        spec_keys=np.array(list(asdict(model.spec).keys())),
        spec_values=np.array(list(asdict(model.spec).values()), dtype=np.float64),
    )
    return path


def load_head_model(path: str | Path) -> HeadModel:
    """Read a head model written by :func:`save_head_model`."""
    with np.load(path, allow_pickle=False) as data:
        raw_spec = dict(
            zip(
                (str(k) for k in data["spec_keys"]),
                (float(v) for v in data["spec_values"]),
                strict=True,
            )
        )
        # `from __future__ import annotations` makes field annotations strings,
        # so reconstruct the declared type from the default value instead.
        defaults = asdict(HeadModelSpec())
        spec = HeadModelSpec(
            **{k: type(defaults[k])(v) for k, v in raw_spec.items()}
        )
        return HeadModel(
            spec=spec,
            scalp_directions=data["scalp_directions"],
            source_positions=data["source_positions"],
            source_moments=data["source_moments"],
            lead_field=data["lead_field"],
            lead_field_scale=float(data["lead_field_scale"]),
            solver_id=str(data["solver_id"]),
        )
