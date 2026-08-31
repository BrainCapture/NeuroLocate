# Copyright 2026 NeuroLocate contributors
# SPDX-License-Identifier: Apache-2.0
"""Gate B for the source-localization path of ``headfield``.

Two derivatives, checked differently because they are produced differently.

The **moment** derivative is exact algebra, so it is held to machine precision
against differences of the served endpoint.

The **position** derivative is central differences through OpenMEEG's compiled
source assembly. Checking it against finite differences of the same endpoint is
necessary but self-referential, so the interesting assertions here are the ones
that could actually fail:

* the error scales as O(h²) over decades of step size, which it only does if the
  underlying forward really is smooth — a lookup table or a nearest-vertex
  snap would produce a staircase and fail this;
* it agrees with an independent 4th-order stencil;
* on a concentric-sphere geometry it agrees with the gradient of a completely
  different forward implementation, the analytic Legendre series;
* it points downhill: a step along the negative gradient lowers the objective.

Everything runs against the served component, i.e. against what the optimizer
actually calls.
"""

from __future__ import annotations

import numpy as np
import pytest
from neurolayout.clients import open_component
from neurolayout_shared.geometry import fibonacci_directions, icosphere
from neurolayout_shared.openmeeg_model import (
    HeadGeometry,
    OpenMEEGForward,
    default_artifact_path,
)
from neurolayout_shared.source_model import position_jacobian
from neurolayout_shared.sphere_model import sphere_lead_field

pytest.importorskip("openmeeg")

#: Source positions spanning depth and laterality, metres, MNE head frame.
PROBES = (
    np.array([[-0.037, 0.023, 0.036]]),
    np.array([[0.041, -0.018, 0.052]]),
    np.array([[0.004, -0.055, 0.021]]),
)

BACKENDS = ("openmeeg", "sphere")


@pytest.fixture(scope="module")
def headfield():
    if not default_artifact_path().exists():
        pytest.skip("no OpenMEEG head-model artifact; run scripts/build_openmeeg_headmodel.py")
    with open_component("headfield", "local") as tesseract:
        yield tesseract


@pytest.fixture(scope="module")
def problem():
    rng = np.random.default_rng(23)
    return {
        "timecourses": 2e-8 * rng.standard_normal((1, 1, 3, 5)),
        "rng": np.random.default_rng(24),
    }


def _static(backend: str) -> dict:
    return {"mode": "localize", "backend": backend}


def _scalar_fn(tesseract, backend, timecourses, weights):
    def scalar(position: np.ndarray) -> float:
        out = tesseract.apply(
            {
                "source_positions": position,
                "source_timecourses": timecourses,
                **_static(backend),
            }
        )
        return float(np.sum(np.asarray(out["eeg"]) * weights))

    return scalar


def _fd_gradient(scalar, position: np.ndarray, step: float, order: int = 2) -> np.ndarray:
    stencil = {
        2: ((-1, -0.5), (1, 0.5)),
        4: ((-2, 1 / 12), (-1, -8 / 12), (1, 8 / 12), (2, -1 / 12)),
    }[order]
    gradient = np.zeros_like(position)
    for axis in range(3):
        gradient[0, axis] = (
            sum(
                weight * scalar(_shifted(position, axis, shift * step))
                for shift, weight in stencil
            )
            / step
        )
    return gradient


def _shifted(position: np.ndarray, axis: int, delta: float) -> np.ndarray:
    probe = position.copy()
    probe[0, axis] += delta
    return probe


@pytest.mark.parametrize("backend", BACKENDS)
def test_moment_derivative_is_exact(headfield, problem, backend) -> None:
    """The forward is linear in the moments, so its cotangent must be exact."""
    timecourses = problem["timecourses"]
    position = PROBES[0]
    base = headfield.apply(
        {
            "source_positions": position,
            "source_timecourses": timecourses,
            **_static(backend),
        }
    )
    weights = np.random.default_rng(1).standard_normal(np.asarray(base["eeg"]).shape)

    analytic = np.asarray(
        headfield.vector_jacobian_product(
            inputs={
                "source_positions": position,
                "source_timecourses": timecourses,
                **_static(backend),
            },
            vjp_inputs=["source_timecourses"],
            vjp_outputs=["eeg"],
            cotangent_vector={"eeg": weights},
        )["source_timecourses"]
    )

    step = 1e-10
    numeric = np.zeros_like(timecourses)
    for index in np.ndindex(*timecourses.shape):
        plus, minus = timecourses.copy(), timecourses.copy()
        plus[index] += step
        minus[index] -= step
        numeric[index] = (
            float(
                np.sum(
                    np.asarray(
                        headfield.apply(
                            {
                                "source_positions": position,
                                "source_timecourses": plus,
                                **_static(backend),
                            }
                        )["eeg"]
                    )
                    * weights
                )
            )
            - float(
                np.sum(
                    np.asarray(
                        headfield.apply(
                            {
                                "source_positions": position,
                                "source_timecourses": minus,
                                **_static(backend),
                            }
                        )["eeg"]
                    )
                    * weights
                )
            )
        ) / (2.0 * step)

    error = np.abs(analytic - numeric).max() / np.abs(numeric).max()
    assert error < 1e-10, f"moment derivative relative error {error:.2e}"


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("probe", range(len(PROBES)))
def test_position_vjp_matches_the_real_forward(headfield, problem, backend, probe) -> None:
    """The shipped position VJP against differences of the served endpoint."""
    position = PROBES[probe]
    timecourses = problem["timecourses"]
    base = headfield.apply(
        {
            "source_positions": position,
            "source_timecourses": timecourses,
            **_static(backend),
        }
    )
    weights = np.random.default_rng(100 + probe).standard_normal(
        np.asarray(base["eeg"]).shape
    )
    analytic = np.asarray(
        headfield.vector_jacobian_product(
            inputs={
                "source_positions": position,
                "source_timecourses": timecourses,
                **_static(backend),
            },
            vjp_inputs=["source_positions"],
            vjp_outputs=["eeg"],
            cotangent_vector={"eeg": weights},
        )["source_positions"]
    )
    scalar = _scalar_fn(headfield, backend, timecourses, weights)
    reference = _fd_gradient(scalar, position, 1e-4, order=4)
    error = np.abs(analytic - reference).max() / np.abs(reference).max()
    assert error < 1e-5, f"position VJP relative error {error:.2e} ({backend}, probe {probe})"


def test_position_error_scales_as_h_squared(headfield, problem) -> None:
    """The signature of a genuinely smooth forward, and the anti-lookup-table test.

    Between 3e-5 m and 1e-3 m the error must fall like the square of the step. A
    discretized or interpolated gain would flatten out at its own approximation
    error instead.
    """
    position = PROBES[0]
    timecourses = problem["timecourses"]
    base = headfield.apply(
        {
            "source_positions": position,
            "source_timecourses": timecourses,
            **_static("openmeeg"),
        }
    )
    weights = np.random.default_rng(5).standard_normal(np.asarray(base["eeg"]).shape)
    scalar = _scalar_fn(headfield, "openmeeg", timecourses, weights)
    reference = _fd_gradient(scalar, position, 1e-4, order=4)

    steps = np.array([3e-5, 1e-4, 3e-4, 1e-3])
    errors = np.array(
        [
            np.abs(_fd_gradient(scalar, position, float(step)) - reference).max()
            for step in steps
        ]
    )
    slope = np.polyfit(np.log(steps), np.log(errors), 1)[0]
    assert 1.8 < slope < 2.2, f"error scaled as h^{slope:.2f}, expected h^2"
    # And the whole range stays far inside the target.
    scale = np.abs(reference).max()
    assert (errors / scale).max() < 1e-3


def test_no_step_in_the_sweep_is_cherry_picked(headfield, problem) -> None:
    """Every step from 10 nm to 1 mm clears the target, not just a lucky one.

    Six decades of perturbation, spanning "absurdly small" to "a millimetre",
    all agree with the high-order reference to better than 1e-3 relative. There
    is no favourable step to find and no unfavourable one to avoid.
    """
    position = PROBES[0]
    timecourses = problem["timecourses"]
    base = headfield.apply(
        {
            "source_positions": position,
            "source_timecourses": timecourses,
            **_static("openmeeg"),
        }
    )
    weights = np.random.default_rng(6).standard_normal(np.asarray(base["eeg"]).shape)
    scalar = _scalar_fn(headfield, "openmeeg", timecourses, weights)
    reference = _fd_gradient(scalar, position, 1e-4, order=4)
    scale = np.abs(reference).max()

    for step in (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3):
        error = np.abs(_fd_gradient(scalar, position, step) - reference).max() / scale
        assert error < 1e-3, f"step {step:.0e} m gave relative error {error:.2e}"


def test_second_and_fourth_order_stencils_agree() -> None:
    """Two different stencils, one answer."""
    from neurolayout_shared.openmeeg_model import load_sensor_operator

    path = default_artifact_path()
    geometry = HeadGeometry.load(path)
    forward = OpenMEEGForward(geometry, load_sensor_operator(path, geometry))
    for position in PROBES:
        second = position_jacobian(forward.gain, position, step=1e-5, order=2)
        fourth = position_jacobian(forward.gain, position, step=1e-4, order=4)
        error = np.abs(second - fourth).max() / np.abs(fourth).max()
        assert error < 1e-5, f"stencil disagreement {error:.2e}"


def test_gradient_agrees_with_an_independent_forward() -> None:
    """Cross-solver check on a sphere: BEM gradient vs analytic-series gradient.

    The two forwards share no code. If the BEM's position sensitivity were an
    artefact of its own discretization rather than the physics, this is where it
    would show.
    """
    radius, sigma, n_sensors = 0.09, 0.33, 60
    unit_vertices, triangles = icosphere(3)
    directions = fibonacci_directions(n_sensors)
    geometry = HeadGeometry(
        vertices=tuple(f * radius * unit_vertices for f in (0.85, 0.93, 1.0)),
        triangles=(triangles, triangles, triangles),
        conductivities=np.array([sigma, sigma, sigma]),
        sensor_xyz=radius * directions,
        channel_names=tuple(f"S{i:02d}" for i in range(n_sensors)),
        source_space=np.zeros((1, 3)),
        source_normals=np.array([[0.0, 0.0, 1.0]]),
        metadata={"kind": "cross-solver"},
    )
    forward = OpenMEEGForward(geometry)
    forward.build_sensor_operator()
    operator = geometry.reference_operator()

    def analytic_gain(positions: np.ndarray) -> np.ndarray:
        columns = [
            sphere_lead_field(
                directions,
                positions,
                np.tile(axis, (positions.shape[0], 1)),
                radius=radius,
                sigma=sigma,
                n_terms=300,
            )
            for axis in np.eye(3)
        ]
        return np.einsum("cd,dpj->cpj", operator, np.stack(columns, axis=-1))

    rng = np.random.default_rng(9)
    direction = np.array([0.3, -0.5, 0.8])
    direction /= np.linalg.norm(direction)
    for fraction in (0.3, 0.5):
        position = (fraction * radius * direction)[None]
        moment = rng.standard_normal(3)
        weights = rng.standard_normal(n_sensors)
        bem, exact = (
            np.einsum(
                "j,ckjd,c->d",
                moment,
                position_jacobian(gain_fn, position, step=1e-5),
                weights,
            )
            for gain_fn in (forward.gain, analytic_gain)
        )
        cosine = bem @ exact / (np.linalg.norm(bem) * np.linalg.norm(exact))
        relative = np.linalg.norm(bem - exact) / np.linalg.norm(exact)
        assert cosine > 0.999, f"gradient directions disagree (cos {cosine:.5f})"
        assert relative < 0.05, f"gradient magnitudes disagree ({relative:.4f})"


def test_gradient_points_downhill(headfield, problem) -> None:
    """The only property the optimizer actually needs from the derivative."""
    truth = PROBES[0]
    timecourses = problem["timecourses"]
    target = np.asarray(
        headfield.apply(
            {
                "source_positions": truth,
                "source_timecourses": timecourses,
                **_static("openmeeg"),
            }
        )["eeg"]
    )
    start = truth + np.array([[0.008, -0.004, 0.006]])

    def loss_at(position: np.ndarray) -> float:
        predicted = np.asarray(
            headfield.apply(
                {
                    "source_positions": position,
                    "source_timecourses": timecourses,
                    **_static("openmeeg"),
                }
            )["eeg"]
        )
        return 0.5 * float(np.sum((predicted - target) ** 2))

    predicted = np.asarray(
        headfield.apply(
            {
                "source_positions": start,
                "source_timecourses": timecourses,
                **_static("openmeeg"),
            }
        )["eeg"]
    )
    gradient = np.asarray(
        headfield.vector_jacobian_product(
            inputs={
                "source_positions": start,
                "source_timecourses": timecourses,
                **_static("openmeeg"),
            },
            vjp_inputs=["source_positions"],
            vjp_outputs=["eeg"],
            cotangent_vector={"eeg": predicted - target},
        )["source_positions"]
    )
    direction = gradient / np.linalg.norm(gradient)
    before = loss_at(start)
    after = loss_at(start - 1e-3 * direction)
    assert after < before, "a step along the negative gradient did not lower the loss"


def test_electrode_xyz_cotangent_is_zero_in_localize_mode(headfield, problem) -> None:
    """``electrode_xyz`` is a constant of the head model here, and must say so."""
    grads = headfield.vector_jacobian_product(
        inputs={
            "source_positions": PROBES[0],
            "source_timecourses": problem["timecourses"],
            **_static("openmeeg"),
        },
        vjp_inputs=["source_positions", "source_timecourses"],
        vjp_outputs=["electrode_xyz"],
        cotangent_vector={"electrode_xyz": np.ones((64, 3))},
    )
    assert np.abs(np.asarray(grads["source_positions"])).max() == 0.0
    assert np.abs(np.asarray(grads["source_timecourses"])).max() == 0.0
