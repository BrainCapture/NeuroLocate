#!/usr/bin/env python
# Copyright 2026 NeuroLocate contributors
# SPDX-License-Identifier: Apache-2.0
"""Audit the source-parameter derivatives of the ``headfield`` Tesseract.

The position derivative is the one number a reader should be suspicious of: it
is produced by central differences through a compiled BEM solver, so "it agrees
with finite differences" is only meaningful if the finite differences are shown
to be trustworthy in the first place. This script therefore reports four
independent things, and writes them all out rather than picking the flattering
one.

1. **Moment derivative.** Exact analytic algebra versus central differences of
   the served endpoint. Should be at machine precision.
2. **Step-size sweep.** The position derivative recomputed across five decades
   of perturbation, from 10 nm to 3 mm, against a fixed high-order reference.
   What should appear is a clean O(h²) truncation line that only bottoms out
   where round-off takes over — if instead the error is flat and large, or rises
   at moderate steps, the underlying forward is not as smooth as claimed.
3. **Stencil order.** The 2nd-order rule the component ships against a 4th-order
   Richardson stencil at the same step.
4. **Cross-solver.** On a concentric-sphere geometry, the gradient taken through
   the OpenMEEG BEM against the gradient taken through the *independent*
   analytic Legendre-series forward. This is the check the BEM cannot pass by
   being self-consistently wrong.

Usage::

    python scripts/report_source_vjp.py --json results/source_vjp.json
    python scripts/report_source_vjp.py --transport image
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "app"))
sys.path.insert(0, str(REPO_ROOT / "components" / "shared_code"))

from neurolayout.clients import open_component  # noqa: E402
from neurolayout_shared.geometry import fibonacci_directions, icosphere  # noqa: E402
from neurolayout_shared.openmeeg_model import (  # noqa: E402
    HeadGeometry,
    OpenMEEGForward,
    default_artifact_path,
    openmeeg_version,
)
from neurolayout_shared.source_model import position_jacobian  # noqa: E402
from neurolayout_shared.sphere_model import sphere_lead_field  # noqa: E402

#: Steps to sweep, in metres: 10 nm to 3 mm. The small end is there to find
#: the round-off floor, not because anyone would use it.
SWEEP_STEPS = (1e-8, 1e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3)

#: Probe sources, in metres, MNE head frame. Chosen by hand to span depth and
#: laterality rather than drawn at random, so the table is stable run to run.
PROBES = (
    (-0.037, 0.023, 0.036),  # left, mid-depth
    (0.041, -0.018, 0.052),  # right, superficial-ish
    (0.004, -0.055, 0.021),  # posterior midline, deep
)


def _timecourses(rng: np.random.Generator, n_times: int = 6) -> np.ndarray:
    return 2e-8 * rng.standard_normal((1, 1, 3, n_times))


def _scalar(outputs: dict, weights: np.ndarray) -> float:
    return float(np.sum(np.asarray(outputs["eeg"]) * weights))


def audit_component(tesseract, backend: str) -> dict:
    """Compare the component's VJP against differences of its own ``apply``."""
    rng = np.random.default_rng(7)
    timecourses = _timecourses(rng)
    static = {"mode": "localize", "backend": backend}
    results = []

    for probe in PROBES:
        position = np.array([probe], dtype=np.float64)
        base = tesseract.apply(
            {"source_positions": position, "source_timecourses": timecourses, **static}
        )
        weights = rng.standard_normal(np.asarray(base["eeg"]).shape)

        vjp = tesseract.vector_jacobian_product(
            inputs={
                "source_positions": position,
                "source_timecourses": timecourses,
                **static,
            },
            vjp_inputs=["source_positions", "source_timecourses"],
            vjp_outputs=["eeg"],
            cotangent_vector={"eeg": weights},
        )
        analytic_position = np.asarray(vjp["source_positions"])
        analytic_moment = np.asarray(vjp["source_timecourses"])

        def scalar_at(pos: np.ndarray, weights: np.ndarray = weights) -> float:
            return _scalar(
                tesseract.apply(
                    {"source_positions": pos, "source_timecourses": timecourses, **static}
                ),
                weights,
            )

        # --- moment: exact algebra, so this should be at round-off ---
        moment_fd = np.zeros_like(timecourses)
        step = 1e-10
        for index in np.ndindex(*timecourses.shape):
            plus, minus = timecourses.copy(), timecourses.copy()
            plus[index] += step
            minus[index] -= step
            moment_fd[index] = (
                _scalar(
                    tesseract.apply(
                        {"source_positions": position, "source_timecourses": plus, **static}
                    ),
                    weights,
                )
                - _scalar(
                    tesseract.apply(
                        {"source_positions": position, "source_timecourses": minus, **static}
                    ),
                    weights,
                )
            ) / (2.0 * step)
        moment_error = float(
            np.abs(analytic_moment - moment_fd).max() / np.abs(moment_fd).max()
        )

        # --- position: the sweep ---
        reference = _fd_gradient(scalar_at, position, 1e-4, order=4)
        sweep = []
        for h in SWEEP_STEPS:
            numeric = _fd_gradient(scalar_at, position, h, order=2)
            sweep.append(
                {
                    "step_m": h,
                    "abs_error": float(np.abs(numeric - reference).max()),
                    "rel_error": float(
                        np.abs(numeric - reference).max() / np.abs(reference).max()
                    ),
                }
            )

        results.append(
            {
                "position_m": list(probe),
                "moment_rel_error": moment_error,
                "position_rel_error": float(
                    np.abs(analytic_position - reference).max() / np.abs(reference).max()
                ),
                "position_grad": analytic_position.tolist(),
                "reference_grad": reference.tolist(),
                "sweep": sweep,
            }
        )
    return {"backend": backend, "probes": results}


def _fd_gradient(scalar_at, position: np.ndarray, step: float, order: int) -> np.ndarray:
    """Finite-difference gradient of a scalar wrt a ``[1, 3]`` position."""
    offsets = {
        2: ((-1, -0.5), (1, 0.5)),
        4: ((-2, 1 / 12), (-1, -8 / 12), (1, 8 / 12), (2, -1 / 12)),
    }[order]
    gradient = np.zeros_like(position)
    for axis in range(3):
        total = 0.0
        for shift, weight in offsets:
            probe = position.copy()
            probe[0, axis] += shift * step
            total += weight * scalar_at(probe)
        gradient[0, axis] = total / step
    return gradient


def audit_stencils(forward: OpenMEEGForward) -> dict:
    """2nd- versus 4th-order stencils on the packaged head model."""
    out = []
    for probe in PROBES:
        position = np.array([probe])
        second = position_jacobian(forward.gain, position, step=1e-5, order=2)
        fourth = position_jacobian(forward.gain, position, step=1e-4, order=4)
        out.append(
            {
                "position_m": list(probe),
                "max_abs_diff": float(np.abs(second - fourth).max()),
                "rel_diff": float(np.abs(second - fourth).max() / np.abs(fourth).max()),
            }
        )
    return {"probes": out}


def audit_cross_solver(subdivisions: int = 3, n_sensors: int = 60) -> dict:
    """OpenMEEG's position gradient against the analytic sphere's, on a sphere.

    Two entirely different forward implementations — a discretized boundary
    element solve and a Legendre series — differentiated the same way. Agreement
    means the gradient is a property of the physics, not of the solver.
    """
    radius, sigma = 0.09, 0.33
    unit_vertices, triangles = icosphere(subdivisions)
    directions = fibonacci_directions(n_sensors)
    geometry = HeadGeometry(
        vertices=tuple(f * radius * unit_vertices for f in (0.85, 0.93, 1.0)),
        triangles=(triangles, triangles, triangles),
        conductivities=np.array([sigma, sigma, sigma]),
        sensor_xyz=radius * directions,
        channel_names=tuple(f"S{i:02d}" for i in range(n_sensors)),
        source_space=np.zeros((1, 3)),
        source_normals=np.array([[0.0, 0.0, 1.0]]),
        metadata={"kind": "concentric-sphere-cross-check"},
    )
    forward = OpenMEEGForward(geometry)
    forward.build_sensor_operator()
    reference_operator = geometry.reference_operator()

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
        return np.einsum("cd,dpj->cpj", reference_operator, np.stack(columns, axis=-1))

    rng = np.random.default_rng(5)
    out = []
    for fraction in (0.3, 0.5, 0.7):
        direction = np.array([0.3, -0.5, 0.8])
        direction /= np.linalg.norm(direction)
        position = (fraction * radius * direction)[None]
        moment = rng.standard_normal(3)
        moment /= np.linalg.norm(moment)
        weights = rng.standard_normal(n_sensors)

        gradients = {}
        for name, gain_fn in (("openmeeg", forward.gain), ("analytic", analytic_gain)):
            jacobian = position_jacobian(gain_fn, position, step=1e-5, order=2)
            gradients[name] = np.einsum("j,ckjd,c->d", moment, jacobian, weights)
        bem, exact = gradients["openmeeg"], gradients["analytic"]
        out.append(
            {
                "depth_fraction": fraction,
                "cosine": float(
                    bem @ exact / (np.linalg.norm(bem) * np.linalg.norm(exact))
                ),
                "rel_error": float(np.linalg.norm(bem - exact) / np.linalg.norm(exact)),
                "magnitude_ratio": float(np.linalg.norm(bem) / np.linalg.norm(exact)),
            }
        )
    return {"subdivisions": subdivisions, "probes": out}


def audit_timing(forward: OpenMEEGForward) -> dict:
    """Wall-clock cost of the pieces the optimizer pays for."""
    position = np.array([PROBES[0]])
    forward.gain(position)  # warm

    start = time.perf_counter()
    for _ in range(20):
        forward.gain(position)
    gain_ms = (time.perf_counter() - start) / 20 * 1e3

    start = time.perf_counter()
    for _ in range(5):
        position_jacobian(forward.gain, position, order=2)
    jacobian_ms = (time.perf_counter() - start) / 5 * 1e3

    cold = OpenMEEGForward(forward.geometry, forward.sensor_operator)
    start = time.perf_counter()
    cold.gain(position)
    first_ms = (time.perf_counter() - start) * 1e3

    return {
        "gain_ms": gain_ms,
        "position_jacobian_ms": jacobian_ms,
        "first_gain_after_load_ms": first_ms,
        "solver_calls_per_position_vjp": 1,
        "dipoles_per_position_vjp": 18,
        "system_size": forward.system_size,
    }


def main() -> int:
    """Run every audit and write the JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transport", default="local", choices=("local", "image"))
    parser.add_argument("--json", type=Path, default=REPO_ROOT / "results" / "source_vjp.json")
    parser.add_argument("--skip-cross-solver", action="store_true")
    args = parser.parse_args()

    report: dict = {
        "openmeeg_version": openmeeg_version(),
        "transport": args.transport,
        "sweep_steps_m": list(SWEEP_STEPS),
    }

    with open_component("headfield", args.transport) as tesseract:
        for backend in ("openmeeg", "sphere"):
            report[f"component_{backend}"] = audit_component(tesseract, backend)

    geometry = HeadGeometry.load(default_artifact_path())
    from neurolayout_shared.openmeeg_model import load_sensor_operator

    forward = OpenMEEGForward(geometry, load_sensor_operator(default_artifact_path(), geometry))
    report["stencils"] = audit_stencils(forward)
    report["timing"] = audit_timing(forward)
    if not args.skip_cross_solver:
        report["cross_solver"] = audit_cross_solver()

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"OpenMEEG {report['openmeeg_version']} · transport {args.transport}")
    for backend in ("openmeeg", "sphere"):
        section = report[f"component_{backend}"]
        moment = max(p["moment_rel_error"] for p in section["probes"])
        position = max(p["position_rel_error"] for p in section["probes"])
        print(f"  {backend:9s} moment rel err {moment:.2e} · position rel err {position:.2e}")
    print("  step sweep (openmeeg, worst probe):")
    worst = report["component_openmeeg"]["probes"]
    for index, step in enumerate(SWEEP_STEPS):
        rel = max(p["sweep"][index]["rel_error"] for p in worst)
        print(f"    h = {step:.0e} m   rel err {rel:.2e}")
    if "cross_solver" in report:
        for probe in report["cross_solver"]["probes"]:
            print(
                f"  cross-solver depth {probe['depth_fraction']}: "
                f"cos {probe['cosine']:.6f} rel {probe['rel_error']:.4f}"
            )
    timing = report["timing"]
    print(
        f"  timing: gain {timing['gain_ms']:.1f} ms · position jacobian "
        f"{timing['position_jacobian_ms']:.1f} ms · system {timing['system_size']}"
    )
    print(f"  wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
