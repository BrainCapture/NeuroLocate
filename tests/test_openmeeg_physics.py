# Copyright 2026 NeuroLocate contributors
# SPDX-License-Identifier: Apache-2.0
r"""Gate G, part two — OpenMEEG is solving the physics we think it is.

The decisive check is a case with a closed-form answer. A three-layer BEM whose
three conductivities are *equal* is a homogeneous sphere, and the surface
potential of a dipole in a homogeneous sphere is the classical Legendre series
already implemented (and independently validated against MNE's analytic sphere)
in :mod:`neurolayout_shared.sphere_model`.

That single comparison pins down three things at once:

* the sign convention, i.e. OpenMEEG's triangle winding — with outward
  right-hand-rule winding the solution comes back as exactly ``-V``;
* the units — volts per A·m with positions in metres and conductivity in S/m;
* the discretization error of the decimated mesh, as a number rather than a hope.

These tests assemble small BEM systems (162- and 642-vertex spheres) and so cost
seconds, not milliseconds. That is the price of checking the solver rather than
trusting it.
"""

from __future__ import annotations

import numpy as np
import pytest
from neurolayout_shared.geometry import icosphere
from neurolayout_shared.openmeeg_model import HeadGeometry, OpenMEEGForward
from neurolayout_shared.sphere_model import sphere_lead_field

pytest.importorskip("openmeeg")

RADIUS = 0.09
SIGMA = 0.33
#: Concentric interfaces, as fractions of the outer radius. Equal conductivities
#: make the layering physically irrelevant, but the geometry still has to be a
#: legal three-interface nested BEM.
SHELL_FRACTIONS = (0.85, 0.93, 1.0)


def _sphere_geometry(subdivisions: int, n_sensors: int = 60) -> HeadGeometry:
    """A concentric three-shell sphere with equal conductivities."""
    unit_vertices, triangles = icosphere(subdivisions)
    sensor_directions = _sensor_directions(n_sensors)
    return HeadGeometry(
        vertices=tuple(fraction * RADIUS * unit_vertices for fraction in SHELL_FRACTIONS),
        triangles=(triangles, triangles, triangles),
        conductivities=np.array([SIGMA, SIGMA, SIGMA]),
        sensor_xyz=RADIUS * sensor_directions,
        channel_names=tuple(f"S{i:02d}" for i in range(n_sensors)),
        source_space=np.zeros((1, 3)),
        source_normals=np.array([[0.0, 0.0, 1.0]]),
        metadata={"kind": "concentric-sphere-test"},
    )


def _sensor_directions(n: int) -> np.ndarray:
    from neurolayout_shared.geometry import fibonacci_directions

    return fibonacci_directions(n)


def _rdm(a: np.ndarray, b: np.ndarray) -> float:
    """Relative difference measure: topography shape error, magnitude removed."""
    return float(np.linalg.norm(a / np.linalg.norm(a) - b / np.linalg.norm(b)))


def _magnitude_ratio(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a) / np.linalg.norm(b))


@pytest.fixture(scope="module")
def sphere_forward() -> OpenMEEGForward:
    """A 642-vertex-per-shell BEM sphere, assembled once."""
    forward = OpenMEEGForward(_sphere_geometry(3))
    forward.build_sensor_operator()
    return forward


def _analytic(geometry: HeadGeometry, position: np.ndarray, moment: np.ndarray) -> np.ndarray:
    directions = geometry.sensor_xyz / np.linalg.norm(
        geometry.sensor_xyz, axis=1, keepdims=True
    )
    potential = sphere_lead_field(
        directions, position[None], moment[None], radius=RADIUS, sigma=SIGMA, n_terms=300
    ).ravel()
    return geometry.reference_operator() @ potential


@pytest.mark.parametrize("depth_fraction", [0.3, 0.5, 0.7])
@pytest.mark.parametrize("orientation", ["radial", "tangential"])
def test_bem_reproduces_the_analytic_sphere(
    sphere_forward: OpenMEEGForward, depth_fraction: float, orientation: str
) -> None:
    """The whole point: a known-answer case, solved by the real solver."""
    direction = np.array([0.3, -0.5, 0.8])
    direction /= np.linalg.norm(direction)
    position = depth_fraction * RADIUS * direction
    if orientation == "radial":
        moment = direction
    else:
        moment = np.cross(direction, [0.0, 0.0, 1.0])
        moment /= np.linalg.norm(moment)

    gain = sphere_forward.gain(position[None])[:, 0, :] @ moment
    reference = _analytic(sphere_forward.geometry, position, moment)

    rdm = _rdm(gain, reference)
    ratio = _magnitude_ratio(gain, reference)
    # Both the sign and the shape have to be right; a winding mistake shows up
    # here as RDM ~2.0 (perfectly anti-correlated), not as a small error.
    assert rdm < 0.05, f"RDM {rdm:.4f} against the analytic sphere"
    assert 0.9 < ratio < 1.1, f"magnitude ratio {ratio:.4f}"


def test_inward_wound_surfaces_are_rejected(sphere_forward: OpenMEEGForward) -> None:
    """A wrongly wound surface must fail loudly instead of flipping the sign.

    OpenMEEG's assembly uses the opposite winding convention from MNE's outward
    right-hand-rule surfaces, so ``openmeeg_triangles`` reverses what it is
    given. Reverse the input as well and the two cancel: the solve succeeds and
    returns the exact negative of the correct potential, which no downstream
    check would catch. Hence the signed-volume gate at construction time.
    """
    geometry = sphere_forward.geometry
    flipped = HeadGeometry(
        vertices=geometry.vertices,
        triangles=tuple(np.ascontiguousarray(t[:, ::-1]) for t in geometry.triangles),
        conductivities=geometry.conductivities,
        sensor_xyz=geometry.sensor_xyz,
        channel_names=geometry.channel_names,
        source_space=geometry.source_space,
        source_normals=geometry.source_normals,
        metadata=geometry.metadata,
    )
    with pytest.raises(ValueError, match="outward"):
        OpenMEEGForward(flipped)


def test_discretization_error_falls_with_resolution() -> None:
    """A coarser sphere must be worse, and the fine one must be good.

    This is the number quoted in the writeup, so it is asserted rather than
    described.
    """
    direction = np.array([0.3, -0.5, 0.8])
    direction /= np.linalg.norm(direction)
    position = 0.5 * RADIUS * direction

    errors = []
    for subdivisions in (2, 3):
        forward = OpenMEEGForward(_sphere_geometry(subdivisions))
        gain = forward.gain(position[None])[:, 0, :] @ direction
        errors.append(_rdm(gain, _analytic(forward.geometry, position, direction)))

    assert errors[1] < errors[0], f"RDM did not improve with resolution: {errors}"
    assert errors[1] < 0.02


def test_forward_is_linear_in_the_moment(sphere_forward: OpenMEEGForward) -> None:
    """Exact linearity is what makes the moment derivative analytic."""
    position = np.array([[0.01, -0.02, 0.03]])
    gain = sphere_forward.gain(position)[:, 0, :]
    first, second = np.array([1.0, -2.0, 0.5]), np.array([0.3, 0.4, -1.2])
    np.testing.assert_allclose(
        gain @ (2.0 * first - 3.0 * second),
        2.0 * (gain @ first) - 3.0 * (gain @ second),
        rtol=1e-12,
    )


def test_referenced_gain_has_zero_channel_mean(sphere_forward: OpenMEEGForward) -> None:
    gain = sphere_forward.gain(np.array([[0.01, -0.02, 0.03], [0.0, 0.0, 0.04]]))
    assert np.abs(gain.mean(axis=0)).max() < 1e-12 * np.abs(gain).max()


def test_unreferenced_backend_does_not_apply_the_reference() -> None:
    """``reference=False`` is a diagnostic escape hatch, and must really differ."""
    geometry = _sphere_geometry(2)
    referenced = OpenMEEGForward(geometry, reference=True)
    operator = referenced.build_sensor_operator()
    raw = OpenMEEGForward(geometry, operator, reference=False)
    position = np.array([[0.01, -0.02, 0.03]])
    difference = raw.gain(position) - referenced.gain(position)
    # The difference is exactly the common mode, i.e. constant across channels.
    assert np.ptp(difference, axis=0).max() < 1e-12 * np.abs(raw.gain(position)).max()
    assert np.abs(difference).max() > 0
