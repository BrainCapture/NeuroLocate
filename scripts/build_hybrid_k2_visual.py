#!/usr/bin/env python
# Copyright 2026 NeuroLocate contributors
# SPDX-License-Identifier: Apache-2.0
r"""Record the optimizer trajectories for the K=2 hero figure and animation.

The frozen hybrid shards store answers, not paths: ``results/hybrid/shards`` has
each method's final source set, its sensor residual and its error, and nothing in
between. The hero visualization is *about* what happens in between, so the two
refinements of one deterministic trial are re-run here to record it.

This is a replay, not a benchmark run:

* it runs exactly one condition, ``h-k2-shared-close``, from the committed
  observation artifact, with the frozen ``RefineConfig`` and the frozen
  checkpoint;
* the refinements are batched over the condition's eight trials exactly as
  :mod:`scripts.run_hybrid_benchmark` batches them, because a differently sized
  batch is a different sequence of floating-point reductions;
* every reproduced number — each method's per-source errors, its worst error and
  its sensor residual — is checked against the frozen shard before anything is
  written, and the script exits non-zero if any of them has moved;
* it writes one ``.npz`` under ``results/`` and touches no shard, no summary and
  no document.

It also records a **1-D profile of the objective** along the straight line in
``(position, moment)`` between the two converged answers, which is what the
static figure uses to show that they are two separated basins rather than two
points on one slope. That is the real objective — the same
:func:`neurolayout.hybrid.physics.projector_residual` the optimizer descends,
evaluated through the same OpenMEEG component — restricted to a line, and it
involves no profiling and no re-optimization.

A cortical *conditional slice* (source 1 pinned at truth, source 2 swept over the
grid) was built first and is deliberately not shipped: profiling the two dipole
orientations in closed form at every grid point leaves three free columns that
absorb sensor noise anywhere on the cortex, so the resulting surface spans only
0.0117-0.0132 and its global minimum sits 96 mm from the true second source. That
is a property of the profiled criterion, not a picture of the landscape the
optimizer traverses, and drawing it would have been a dramatic figure of the
wrong thing.

Usage::

    make k2-data
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "app"))
sys.path.insert(0, str(REPO_ROOT / "components" / "shared_code"))

import neurolayout  # noqa: F401,E402  (enables float64 in JAX)

#: The trial the hero figure draws, named here rather than searched for. It is
#: the ``h-k2-shared-close`` trial the report already singles out: the one where
#: the single-start gradient ends in the wrong basin at 124.3 mm while the
#: proposal-initialized refinement finishes at 6.9 mm, at the *same* sensor
#: residual. Everything drawn is that trial's own frozen record.
CONDITION = "h-k2-shared-close"
TRIAL = 4

#: The name the frozen shards record for the checkpoint they were produced with.
#:
#: ``None`` here means "use the one packaged inside the proposal component",
#: which is byte-identical to that checkpoint — the packaged file *is* it, copied
#: into the component by ``make hybrid-finetune`` in the repository this matrix
#: was run in. The replay's own reproduction check is what actually enforces
#: that: it refuses to write unless every error and residual matches the shard.
SHARD_CHECKPOINT = "proposal_physics"
CHECKPOINT = None

#: Largest reproduction disagreement accepted, in millimetres and in relative
#: residual. A replay of the same batch on the same host should be bit-identical;
#: this is a tolerance for a different BLAS, not a licence to redraw a different
#: result.
POSITION_TOLERANCE_MM = 0.05
RESIDUAL_TOLERANCE = 1e-4


def frozen_record(shards: Path) -> dict:
    """The frozen trial record, and the shard context it came from."""
    shard = json.loads((shards / f"{CONDITION}.json").read_text())
    key = f"{CONDITION}/t{TRIAL:02d}"
    for record in shard["trials"]:
        if record["key"] == key:
            return {"trial": record, "shard": shard}
    raise SystemExit(f"no trial {key} in {CONDITION}.json")


def check(name: str, reproduced: dict, frozen: dict) -> None:
    """Refuse to draw a figure on numbers that are not the frozen ones."""
    errors = np.asarray(reproduced["errors_mm"], dtype=float)
    frozen_errors = np.asarray(frozen["errors_mm"], dtype=float)
    drift = float(np.abs(np.sort(errors) - np.sort(frozen_errors)).max())
    residual_drift = abs(reproduced["sensor_residual"] - frozen["sensor_residual"])
    if drift > POSITION_TOLERANCE_MM or residual_drift > RESIDUAL_TOLERANCE:
        raise SystemExit(
            f"{name}: the replay does not reproduce the frozen result "
            f"(errors moved {drift:.4f} mm, residual moved {residual_drift:.2e}). "
            "Refusing to write a visualization of a different run."
        )
    print(
        f"  {name}: worst {reproduced['worst_mm']:.1f} mm vs frozen "
        f"{frozen['worst_mm']:.1f} mm, residual {reproduced['sensor_residual']:.4f} "
        f"vs {frozen['sensor_residual']:.4f}  (max drift {drift:.2e} mm)"
    )


def barrier_profile(
    headfield,
    observed: np.ndarray,
    ends: dict[str, dict[str, np.ndarray]],
    config,
    samples: int = 61,
) -> dict[str, np.ndarray]:
    r"""The objective along the straight line between the two converged answers.

    Both refinements minimize the same function of ``(p, m)``; they simply start
    in different places. Evaluating that function on the segment joining their two
    fixed points answers the only question the two results raise — are these two
    basins, or two points on one slope — and answers it with the objective itself
    rather than with a proxy.

    Positions are interpolated linearly and moments are interpolated and
    renormalized, source by source under the truth-matched correspondence, so
    ``t = 0`` and ``t = 1`` reproduce the two recorded residuals exactly. Nothing
    is profiled and nothing is re-optimized: every intermediate point is one more
    evaluation of :func:`neurolayout.hybrid.physics.projector_residual` through
    the same OpenMEEG component.
    """
    import jax.numpy as jnp
    from neurolayout.hybrid.physics import columns, projector_residual

    fractions = np.linspace(0.0, 1.0, samples)
    start, finish = ends["gradient"], ends["hybrid"]
    positions = (
        start["positions_m"][None] * (1.0 - fractions[:, None, None])
        + finish["positions_m"][None] * fractions[:, None, None]
    )
    moments = (
        start["moments"][None] * (1.0 - fractions[:, None, None])
        + finish["moments"][None] * fractions[:, None, None]
    )
    moments = moments / np.maximum(
        np.linalg.norm(moments, axis=-1, keepdims=True), 1e-30
    )
    built = columns(
        headfield, jnp.asarray(positions), jnp.asarray(moments * 1e-8), config
    )
    residual = np.asarray(
        projector_residual(
            built, jnp.broadcast_to(jnp.asarray(observed)[None], (samples,) + observed.shape)
        )
    )
    return {
        "barrier_fraction": fractions,
        "barrier_residual": residual,
        "barrier_positions_m": positions,
    }


def main() -> int:
    """Replay the trial, verify it, and write the trajectory artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--observations", type=Path,
        default=REPO_ROOT / "results" / "hybrid" / "observations.npz",
    )
    parser.add_argument(
        "--shards", type=Path, default=REPO_ROOT / "results" / "hybrid" / "shards"
    )
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "results" / "hybrid_k2_visual.npz"
    )
    parser.add_argument(
        "--checkpoint", default=CHECKPOINT,
        help="a named checkpoint under NEUROLOCATE_PROPOSAL_DIR; the default is "
             "the one packaged inside the component",
    )
    parser.add_argument("--transport", default="local", choices=("local", "image"))
    parser.add_argument(
        "--no-barrier", action="store_true",
        help="skip the 1-D objective profile between the two converged answers",
    )
    arguments = parser.parse_args()

    from neurolayout.clients import open_components
    from neurolayout.hybrid.benchmark import conditions_by_name, conditions_fingerprint
    from neurolayout.hybrid.refine import refine
    from neurolayout.localize import Containment, LocalizeConfig
    from neurolayout.matching import match_sources
    from neurolayout_shared.openmeeg_model import HeadGeometry, default_artifact_path
    from run_hybrid_benchmark import (  # noqa: E402  - the runner is the reference
        REFINE,
        load_observations,
        observe_condition,
        run_proposal,
        sensor_error,
        uninformed_start,
    )

    frozen = frozen_record(arguments.shards)
    record, shard = frozen["trial"], frozen["shard"]
    if arguments.checkpoint is None and shard["checkpoint"] != SHARD_CHECKPOINT:
        raise SystemExit(
            f"the shard was produced with checkpoint {shard['checkpoint']!r}, "
            f"which is not the one packaged in the component"
        )
    if arguments.checkpoint is not None and shard["checkpoint"] != arguments.checkpoint:
        raise SystemExit(
            f"the shard was produced with checkpoint {shard['checkpoint']!r}, "
            f"not {arguments.checkpoint!r}"
        )
    if shard["refine"] != REFINE.to_dict():
        raise SystemExit("the runner's refinement settings differ from the shard's")

    manifest, stored = load_observations(arguments.observations)
    if manifest.get("conditions_fingerprint") != conditions_fingerprint():
        raise SystemExit("the observation artifact is bound to a different matrix")

    condition = conditions_by_name()[CONDITION]
    path = default_artifact_path()
    geometry = HeadGeometry.load(path)
    if geometry.fingerprint() != shard["head_model_fingerprint"]:
        raise SystemExit("the cached head model is not the one the shard was run on")
    containment = Containment.from_points(geometry.vertices[0])
    config = LocalizeConfig(
        backend="openmeeg",
        n_times=int(manifest["n_times"]),
        sfreq=float(manifest["sfreq"]),
    )

    trials = observe_condition(condition, stored, manifest)
    index = next(i for i, trial in enumerate(trials) if trial["trial"] == TRIAL)
    observed = np.stack([trial["eeg"] for trial in trials])
    truth = np.stack([trial["truth_positions_m"] for trial in trials])

    payload: dict[str, np.ndarray] = {}
    print(f"replaying {CONDITION}/t{TRIAL:02d} in its own batch of {len(trials)}")

    with open_components(["headfield", "proposal"], arguments.transport) as opened:
        headfield, proposal = opened["headfield"], opened["proposal"]

        proposed, proposed_moments, _, _ = run_proposal(
            proposal, trials, condition, arguments.checkpoint
        )
        match = match_sources(proposed[index], truth[index])
        check(
            "proposal",
            {
                "errors_mm": match.errors_mm,
                "worst_mm": float(match.max_error_mm),
                "sensor_residual": sensor_error(
                    headfield, proposed[index], proposed_moments[index],
                    observed[index], config,
                ),
            },
            record["methods"]["proposal"],
        )

        ends: dict[str, dict[str, np.ndarray]] = {}
        starts = {
            "gradient": uninformed_start(condition, trials, containment),
            "hybrid": proposed,
        }
        for name, start in starts.items():
            print(f"  refining: {name}", flush=True)
            result = refine(
                headfield, observed, start, None, config,
                containment, REFINE, truth_m=truth,
            )
            match = match_sources(result["positions_m"][index], truth[index])
            check(
                name,
                {
                    "errors_mm": match.errors_mm,
                    "worst_mm": float(match.max_error_mm),
                    "sensor_residual": float(result["data_loss"][index]),
                },
                record["methods"][name],
            )
            history = result["history"]
            payload[f"{name}_path_m"] = (
                np.asarray(history["position_cm"], dtype=float)[:, index] * 1e-2
            )
            payload[f"{name}_error_mm"] = np.asarray(
                history["error_mm"], dtype=float
            )[:, index]
            payload[f"{name}_residual"] = np.asarray(history["data"], dtype=float)[:, index]
            payload[f"{name}_start_m"] = np.asarray(start[index], dtype=float)
            payload[f"{name}_final_m"] = np.asarray(result["positions_m"][index])
            payload["steps"] = np.asarray(history["step"], dtype=int)
            # Truth-matched order, so source k of one method is compared with
            # source k of the other and with true source k.
            order = np.empty(len(match.assignment), dtype=int)
            order[np.asarray(match.assignment)] = np.arange(len(match.assignment))
            ends[name] = {
                "positions_m": np.asarray(result["positions_m"][index])[order],
                "moments": np.asarray(result["moments"][index])[order],
            }

        if not arguments.no_barrier:
            payload.update(barrier_profile(headfield, observed[index], ends, config))

    payload["truth_m"] = truth[index]
    payload["proposal_m"] = proposed[index]
    payload["observed"] = observed[index]
    payload["sensor_xyz"] = stored[f"{CONDITION}/t{TRIAL:02d}"]["sensor_xyz"]
    payload["frozen"] = np.asarray(
        json.dumps(
            {
                "condition": CONDITION,
                "trial": TRIAL,
                "n_sources": record["n_sources"],
                "correlation": record["correlation"],
                "separation": record["separation"],
                "snr_db": record["snr_db"],
                "true_separation_mm": record["true_separation_mm"],
                "fingerprint": shard["fingerprint"],
                "checkpoint": shard["checkpoint"],
                "refine": shard["refine"],
                "methods": {
                    name: {
                        key: record["methods"][name][key]
                        for key in ("errors_mm", "worst_mm", "median_mm",
                                    "sensor_residual")
                    }
                    for name in ("proposal", "gradient", "hybrid")
                },
            },
            sort_keys=True,
        )
    )

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arguments.out, **payload)
    print(f"wrote {arguments.out}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
