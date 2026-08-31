#!/usr/bin/env python
# Copyright 2026 NeuroLocate contributors
# SPDX-License-Identifier: Apache-2.0
r"""What the batched headfield path costs, and that it changed no answer.

Three constructions of the same quantity, measured against each other:

``loop``
    ``B`` separate ``localize`` calls, one per source set. What the batched mode
    replaces, and the definition of the right answer.
``flat``
    One ``localize`` call over all ``B*K`` positions treated as a single source
    set, with the off-block time courses zeroed. Exact, and quadratic in the batch
    in the array it carries across the boundary.
``batch``
    ``localize_batch``. One call, linear in the batch.

Each is measured at **two levels**, and the two say different things.

*Solver level* calls the shared-code functions directly. Here the three are within
about 15% of each other, which is the finding: OpenMEEG's ``DipSourceMat`` cost is
essentially linear in the dipole count with little per-call overhead, so batching
buys nothing from the solver. The mathematics is unchanged and so is the work.

*Component level* goes through the ``headfield`` Tesseract, which is where the
optimizer actually meets it. Every call pays schema validation, an
``abstract_eval``, and a serialization round trip, and the loop pays all of it
``B`` times. That is where the batched path earns its keep, and it is why the
measurement is reported at both levels rather than at the convenient one.

A throughput change that alters an answer is a bug, so the numerical equality of
all three is measured every time the speed is.

Usage::

    python scripts/report_batched_headfield.py --out results/hybrid/batching.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "app"))
sys.path.insert(0, str(REPO_ROOT / "components" / "shared_code"))

from neurolayout_shared.openmeeg_model import (  # noqa: E402
    OPENMP_THREADS,
    HeadGeometry,
    default_artifact_path,
    load_forward,
    openmeeg_version,
)
from neurolayout_shared.source_model import (  # noqa: E402
    backward,
    backward_batched,
    forward,
    forward_batched,
)

DEFAULT_OUT = REPO_ROOT / "results" / "hybrid" / "batching.json"

#: (batch, sources) pairs to measure. The last is what the benchmark runs.
SHAPES = ((1, 4), (2, 4), (4, 4), (8, 2), (8, 4))

#: Epoch length. The forward is linear in it and the solver work is not, so it
#: only matters for the einsum, which is the part that is never the bottleneck.
N_TIMES = 32


def problem(geometry: HeadGeometry, batch: int, sources: int, seed: int):
    """Deterministic positions, moment time courses and an output cotangent."""
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(geometry.source_space), batch * sources, replace=False)
    positions = geometry.source_space[picks].reshape(batch, sources, 3) * 0.95
    timecourses = rng.normal(0.0, 1e-8, (batch, sources, 3, N_TIMES))
    cotangent = rng.normal(0.0, 1.0, (batch, geometry.n_channels, N_TIMES))
    return positions, timecourses, cotangent


def flattened(positions: np.ndarray, timecourses: np.ndarray):
    """The batch as one ``B*K``-source set with block-diagonal time courses."""
    batch, sources = positions.shape[0], positions.shape[1]
    flat = np.zeros((batch, batch * sources, 3, timecourses.shape[3]))
    for index in range(batch):
        flat[index, index * sources : (index + 1) * sources] = timecourses[index]
    return positions.reshape(-1, 3), flat


def timed(function, repeats: int) -> tuple[float, object]:
    """Median seconds over ``repeats`` calls, and the last result.

    ``function`` takes the repeat index, and is handed a *different* position set
    per repeat — and a set no other construction has seen. That is not cosmetic.
    The OpenMEEG forward memoizes its gain per position set, so timing the same
    positions three times measures one solve and two dictionary lookups, and
    timing a second construction on the *first* one's positions measures no solve
    at all. Both mistakes were made before this note was written, in opposite
    directions, and both produced two-hundred-fold speedups that were entirely the
    memo. An optimizer never revisits a point, so disjoint fresh positions are
    also the honest workload.
    """
    times, result = [], None
    for index in range(repeats):
        start = time.perf_counter()
        result = function(index)
        times.append(time.perf_counter() - start)
    return float(np.median(times)), result


def measure_shape(gain, geometry: HeadGeometry, batch: int, sources: int, repeats: int):
    """Time and compare the three constructions at one ``(batch, sources)`` shape.

    A function rather than a loop body so the closures below capture arguments
    rather than a loop variable, which is the difference between measuring what
    this iteration set up and measuring whatever the last one left behind.
    """
    def draws_for(construction: int):
        """Position sets for one construction, disjoint from every other's."""
        return [
            problem(
                geometry,
                batch,
                sources,
                seed=batch * 31 + sources + 977 * repeat + 104729 * construction,
            )
            for repeat in range(repeats)
        ]

    loop_draws, flat_draws, batch_draws = (draws_for(index) for index in range(3))
    loop_back_draws, batch_back_draws = draws_for(3), draws_for(4)

    def run_loop(repeat):
        positions, timecourses, _ = loop_draws[repeat]
        return np.concatenate(
            [
                forward(gain, positions[index], timecourses[index : index + 1])["eeg"]
                for index in range(batch)
            ],
            axis=0,
        )

    def run_flat(repeat):
        positions, timecourses, _ = flat_draws[repeat]
        flat_positions, flat_timecourses = flattened(positions, timecourses)
        return forward(gain, flat_positions, flat_timecourses)["eeg"]

    def run_batch(repeat):
        positions, timecourses, _ = batch_draws[repeat]
        return forward_batched(gain, positions, timecourses)["eeg"]

    def run_loop_backward(repeat):
        positions, timecourses, cotangent = loop_back_draws[repeat]
        return [
            backward(
                forward(gain, positions[index], timecourses[index : index + 1]),
                gain,
                positions[index],
                timecourses[index : index + 1],
                cotangent[index : index + 1],
            )["source_positions"]
            for index in range(batch)
        ]

    def run_batch_backward(repeat):
        positions, timecourses, cotangent = batch_back_draws[repeat]
        cache = forward_batched(gain, positions, timecourses)
        return backward_batched(cache, gain, positions, timecourses, cotangent)[
            "source_positions_batch"
        ]

    loop_seconds, _ = timed(run_loop, repeats)
    flat_seconds, _ = timed(run_flat, repeats)
    batch_seconds, _ = timed(run_batch, repeats)
    loop_backward_seconds, _ = timed(run_loop_backward, repeats)
    batch_backward_seconds, _ = timed(run_batch_backward, repeats)

    # The timed draws are deliberately disjoint, so equality is checked on one
    # shared draw instead: the speed and the agreement are different questions and
    # sharing a draw for the timing would corrupt the first to answer the second.
    shared_positions, shared_timecourses, shared_cotangent = problem(
        geometry, batch, sources, seed=7_000_003 + batch * 31 + sources
    )
    shared_flat = flattened(shared_positions, shared_timecourses)
    loop_eeg = np.concatenate(
        [
            forward(gain, shared_positions[index], shared_timecourses[index : index + 1])[
                "eeg"
            ]
            for index in range(batch)
        ],
        axis=0,
    )
    flat_eeg = forward(gain, shared_flat[0], shared_flat[1])["eeg"]
    batch_eeg = forward_batched(gain, shared_positions, shared_timecourses)["eeg"]
    loop_grad = [
        backward(
            forward(gain, shared_positions[index], shared_timecourses[index : index + 1]),
            gain,
            shared_positions[index],
            shared_timecourses[index : index + 1],
            shared_cotangent[index : index + 1],
        )["source_positions"]
        for index in range(batch)
    ]
    batch_grad = backward_batched(
        forward_batched(gain, shared_positions, shared_timecourses),
        gain,
        shared_positions,
        shared_timecourses,
        shared_cotangent,
    )["source_positions_batch"]
    scale = max(np.abs(np.asarray(loop_eeg)).max(), 1e-300)
    grad_scale = max(np.abs(np.stack(loop_grad)).max(), 1e-300)
    return {
        "batch": batch,
        "sources": sources,
        "dipoles_per_forward": batch * sources * 3,
        "forward": {
            "loop_s": loop_seconds,
            "flat_s": flat_seconds,
            "batch_s": batch_seconds,
            "speedup_vs_loop": loop_seconds / max(batch_seconds, 1e-12),
            "speedup_vs_flat": flat_seconds / max(batch_seconds, 1e-12),
            "max_relative_difference_vs_loop": float(
                np.abs(np.asarray(batch_eeg) - np.asarray(loop_eeg)).max() / scale
            ),
            "max_relative_difference_vs_flat": float(
                np.abs(np.asarray(batch_eeg) - np.asarray(flat_eeg)).max() / scale
            ),
        },
        "backward": {
            "loop_s": loop_backward_seconds,
            "batch_s": batch_backward_seconds,
            "speedup_vs_loop": loop_backward_seconds
            / max(batch_backward_seconds, 1e-12),
            "max_relative_difference_vs_loop": float(
                np.abs(np.asarray(batch_grad) - np.stack(loop_grad)).max() / grad_scale
            ),
        },
    }


def measure_transport(headfield, config, batch: int, sources: int, repeats: int):
    """The same three constructions, through the component boundary.

    The loss is a plain sum of squares rather than the projector residual: the
    quantity being timed is the boundary crossing and the solver behind it, and a
    more interesting objective would only add JAX arithmetic to both sides
    equally.
    """
    import jax
    import jax.numpy as jnp
    from tesseract_jax import apply_tesseract

    rng = np.random.default_rng(batch * 17 + sources)

    def draw_set():
        """One construction's draws, disjoint from the others' (see `timed`)."""
        return [
            (
                jnp.asarray(rng.uniform(-0.05, 0.05, (batch, sources, 3))),
                jnp.asarray(rng.normal(0.0, 1e-8, (batch, sources, 3, N_TIMES))),
            )
            for _ in range(repeats)
        ]

    draw_sets = {name: draw_set() for name in ("loop", "flat", "batch")}
    shared = draw_set()[0]
    static = dict(config.static_inputs())
    batch_static = {
        **{k: v for k, v in static.items() if k != "source_positions_batch"},
        "mode": "localize_batch",
        "source_positions": np.zeros((1, 3)),
    }

    def call_loop(pos, times):
        total = 0.0
        for index in range(batch):
            outputs = apply_tesseract(
                headfield,
                {
                    "source_positions": pos[index],
                    "source_timecourses": times[index : index + 1],
                    **static,
                },
            )
            total = total + jnp.sum(outputs["eeg"] ** 2)
        return total

    def call_flat(pos, times):
        selector = jnp.eye(batch)
        wide = times[:, None, :, :, :] * selector[:, :, None, None, None]
        outputs = apply_tesseract(
            headfield,
            {
                "source_positions": pos.reshape(-1, 3),
                "source_timecourses": wide.reshape(batch, batch * sources, 3, N_TIMES),
                **static,
            },
        )
        return jnp.sum(outputs["eeg"] ** 2)

    def call_batch(pos, times):
        outputs = apply_tesseract(
            headfield,
            {
                "source_positions_batch": pos,
                "source_timecourses": times,
                **batch_static,
            },
        )
        return jnp.sum(outputs["eeg"] ** 2)

    def measure(function, draws):
        """Time one construction, then evaluate it once on the shared draw."""
        value_and_grad = jax.value_and_grad(function)
        forward_seconds, _ = timed(
            lambda repeat: float(function(*draws[repeat])), repeats
        )
        grad_seconds, _ = timed(
            lambda repeat: value_and_grad(*draws[repeat]), repeats
        )
        value, gradient = value_and_grad(*shared)
        return forward_seconds, float(value), grad_seconds, gradient

    results = {}
    for name, function in (
        ("loop", call_loop),
        ("flat", call_flat),
        ("batch", call_batch),
    ):
        forward_seconds, value, grad_seconds, gradient = measure(
            function, draw_sets[name]
        )
        results[name] = {
            "forward_s": forward_seconds,
            "value_and_grad_s": grad_seconds,
            "value": float(value),
            "gradient": np.asarray(gradient),
        }

    reference = results["loop"]
    row = {
        "batch": batch,
        "sources": sources,
        "forward": {
            f"{name}_s": entry["forward_s"] for name, entry in results.items()
        },
        "value_and_grad": {
            f"{name}_s": entry["value_and_grad_s"] for name, entry in results.items()
        },
        "forward_speedup_vs_loop": reference["forward_s"]
        / max(results["batch"]["forward_s"], 1e-12),
        "value_and_grad_speedup_vs_loop": reference["value_and_grad_s"]
        / max(results["batch"]["value_and_grad_s"], 1e-12),
        "value_agreement": {
            name: abs(entry["value"] - reference["value"])
            / max(abs(reference["value"]), 1e-300)
            for name, entry in results.items()
        },
        "gradient_agreement": {
            name: float(
                np.abs(entry["gradient"] - reference["gradient"]).max()
                / max(np.abs(reference["gradient"]).max(), 1e-300)
            )
            for name, entry in results.items()
        },
    }
    return row


def main() -> int:
    """Measure the three constructions and write the record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--no-transport", action="store_true",
                        help="solver-level measurements only")
    arguments = parser.parse_args()

    geometry = HeadGeometry.load(default_artifact_path())
    solver = load_forward()

    def gain(points: np.ndarray) -> np.ndarray:
        return solver.gain(points)

    rows = []
    for batch, sources in SHAPES:
        row = measure_shape(gain, geometry, batch, sources, arguments.repeats)
        rows.append(row)
        print(
            f"B={batch} K={sources}:  forward {row['forward']['loop_s'] * 1e3:7.1f} -> "
            f"{row['forward']['batch_s'] * 1e3:7.1f} ms "
            f"({row['forward']['speedup_vs_loop']:.2f}x)   "
            f"backward {row['backward']['loop_s'] * 1e3:8.1f} -> "
            f"{row['backward']['batch_s'] * 1e3:8.1f} ms "
            f"({row['backward']['speedup_vs_loop']:.2f}x)   "
            f"agreement {row['forward']['max_relative_difference_vs_loop']:.2e} / "
            f"{row['backward']['max_relative_difference_vs_loop']:.2e}",
            flush=True,
        )

    transport_rows = []
    if not arguments.no_transport:
        from neurolayout.clients import open_component
        from neurolayout.localize import LocalizeConfig

        config = LocalizeConfig(backend="openmeeg", n_times=N_TIMES, sfreq=160.0)
        with open_component("headfield", "local") as headfield:
            for batch, sources in SHAPES:
                row = measure_transport(
                    headfield, config, batch, sources, arguments.repeats
                )
                transport_rows.append(row)
                print(
                    f"through the component  B={batch} K={sources}:  "
                    f"forward {row['forward']['loop_s'] * 1e3:8.1f} -> "
                    f"{row['forward']['batch_s'] * 1e3:8.1f} ms "
                    f"({row['forward_speedup_vs_loop']:.2f}x)   "
                    f"value_and_grad {row['value_and_grad']['loop_s'] * 1e3:9.1f} -> "
                    f"{row['value_and_grad']['batch_s'] * 1e3:9.1f} ms "
                    f"({row['value_and_grad_speedup_vs_loop']:.2f}x)   "
                    f"gradient agreement {row['gradient_agreement']['batch']:.2e}",
                    flush=True,
                )

    record = {
        "shapes": rows,
        "through_the_component": transport_rows,
        "n_times": N_TIMES,
        "omp_num_threads": OPENMP_THREADS,
        "head_model_fingerprint": geometry.fingerprint(),
        "system_size": solver.system_size,
        "openmeeg_version": openmeeg_version(),
        "numpy_version": np.__version__,
        "python": platform.python_version(),
        "note": (
            "`loop` is B separate localize calls; `flat` is one localize call over "
            "B*K shared sources with block-diagonal time courses; `batch` is "
            "localize_batch. All three are the same mathematics. `shapes` measures "
            "them at the solver, `through_the_component` at the Tesseract boundary."
        ),
    }
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    print(f"wrote {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
