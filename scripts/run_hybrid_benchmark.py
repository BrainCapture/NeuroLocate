#!/usr/bin/env python
# Copyright 2026 NeuroLocate contributors
# SPDX-License-Identifier: Apache-2.0
r"""Run the hard-case matrix: six methods, one observation set, identical physics.

Consumes the artifact ``scripts/build_hybrid_observations.py`` wrote — whose EEG
came from a forward model none of these estimators has — adds noise at each
condition's SNR, hides the truth, and inverts.

The methods, and what separates them:

``rapmusic`` / ``scan``
    Classical, on the substituted OpenMEEG gain. See
    :mod:`neurolayout.hybrid.baselines`.
``proposal``
    The network's output, with no physics applied to it.
``gradient``
    The continuous OpenMEEG refinement from the uninformed initialization the
    frozen benchmark has always used.
``hybrid``
    The **same** refinement loop, same objective, same optimizer, same number of
    steps, from the network's proposal instead. So ``hybrid`` minus ``gradient``
    is the initialization and nothing else, and ``hybrid`` minus ``proposal`` is
    the physics and nothing else.
``hybrid_stopgrad``
    ``hybrid`` with a network trained on the same budget without the
    through-solver gradient.

Every refinement runs the whole condition's trials in **one** batched call into
the ``headfield`` component, which is what makes 300 steps of real BEM assemblies
per cell affordable.

Usage::

    python scripts/run_hybrid_benchmark.py --out results/hybrid/shards
    python scripts/run_hybrid_benchmark.py --conditions h-k2-shared --methods hybrid gradient
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

import neurolayout  # noqa: F401,E402  (enables float64 in JAX)
from neurolayout.baselines import DipoleDictionary, MneInverse  # noqa: E402
from neurolayout.benchmark import random_initialization  # noqa: E402
from neurolayout.hybrid.baselines import rapmusic, scan  # noqa: E402
from neurolayout.hybrid.benchmark import (  # noqa: E402
    CONDITIONS,
    METHODS,
    N_RESTARTS,
    HybridCondition,
    conditions_by_name,
    conditions_fingerprint,
    fingerprint,
)
from neurolayout.hybrid.refine import RefineConfig, refine  # noqa: E402
from neurolayout.localize import Containment, LocalizeConfig  # noqa: E402
from neurolayout.matching import match_sources  # noqa: E402
from neurolayout.montage import CANONICAL_CHANNELS  # noqa: E402
from neurolayout.noise import add_sensor_noise  # noqa: E402
from neurolayout_shared.openmeeg_model import (  # noqa: E402
    HeadGeometry,
    OpenMEEGForward,
    default_artifact_path,
    load_sensor_operator,
    openmeeg_version,
)

DEFAULT_OBSERVATIONS = REPO_ROOT / "results" / "hybrid" / "observations.npz"
DEFAULT_SHARDS = REPO_ROOT / "results" / "hybrid" / "shards"

#: Refinement budget. Identical for ``gradient``, ``hybrid`` and
#: ``hybrid_stopgrad``, because the comparison between them is the
#: initialization and the training, never the budget.
REFINE = RefineConfig(steps=300, record_every=15)

#: Suppression radius for reading a source set out of the heatmap, metres. A
#: little over one 8 mm lattice pitch. Fixed here, before any benchmark result,
#: and never re-chosen from one.
NMS_RADIUS_M = 0.010


def load_observations(path: Path) -> tuple[dict, dict[str, dict[str, np.ndarray]]]:
    """Read the artifact, returning ``(manifest, {trial_key: arrays})``."""
    if not path.exists():
        raise SystemExit(
            f"no observations at {path}. Build them with "
            "`python scripts/build_hybrid_observations.py`."
        )
    with np.load(path, allow_pickle=False) as data:
        manifest = json.loads(str(data["manifest"]))
        trials: dict[str, dict[str, np.ndarray]] = {}
        for key in data.files:
            if key == "manifest":
                continue
            trial_key, field = key.rsplit("/", 1)
            trials.setdefault(trial_key, {})[field] = np.asarray(data[key])
    return manifest, trials


def observe_condition(
    condition: HybridCondition, stored: dict, manifest: dict
) -> list[dict]:
    """Every trial of one condition, with its noise applied.

    Noise is a reproducible function of the stored clean signal and the
    condition's own :class:`~neurolayout.noise.NoiseSpec`, so the artifact holds
    no noise and this is exactly what every method sees.
    """
    from neurolayout.localize import reference_operator

    trials = []
    for trial in range(condition.n_trials):
        key = f"{condition.name}/t{trial:02d}"
        if key not in stored:
            continue
        arrays = stored[key]
        clean = arrays["clean"]
        noisy, report = add_sensor_noise(
            clean,
            arrays["sensor_xyz"],
            condition.noise(trial),
            reference_operator=reference_operator(clean.shape[-2]),
        )
        trials.append(
            {
                "key": key,
                "trial": trial,
                "eeg": noisy[0],
                "clean": clean[0],
                "noise": report,
                "truth_positions_m": arrays["truth_positions_m"],
                "truth_moments_am": arrays["truth_moments_am"],
                "record": manifest["trials"][key],
            }
        )
    return trials


def score(positions_m: np.ndarray, truth_m: np.ndarray) -> dict:
    """Assignment-matched error and the multi-source diagnostics for one trial."""
    if not np.isfinite(positions_m).all():
        return {
            "errors_mm": None,
            "median_mm": None,
            "worst_mm": None,
            "failed": True,
        }
    match = match_sources(positions_m, truth_m)
    return {
        "errors_mm": [float(value) for value in match.errors_mm],
        "median_mm": float(np.median(match.errors_mm)),
        "worst_mm": float(match.max_error_mm),
        "gross_failure": bool(match.max_error_mm > 20.0),
        "min_separation_mm": (
            None if np.isinf(match.min_separation_mm) else float(match.min_separation_mm)
        ),
        "assignment": [int(value) for value in match.assignment],
        "failed": False,
    }


def sensor_error(
    headfield, positions_m: np.ndarray, moments: np.ndarray, observed: np.ndarray, config
) -> float:
    """Relative sensor-space residual after profiling the time courses out.

    Reported next to the localization error because they are different questions:
    a method can fit the sensors perfectly at the wrong place, and on correlated
    sources it routinely does. Seeing both is what makes that legible.
    """
    import jax.numpy as jnp
    from neurolayout.hybrid.physics import columns, projector_residual

    built = columns(
        headfield,
        jnp.asarray(positions_m[None]),
        jnp.asarray(moments[None]),
        config,
    )
    return float(projector_residual(built, jnp.asarray(observed[None]))[0])


def run_proposal(proposal, trials: list[dict], condition: HybridCondition, checkpoint):
    """The network's source set for every trial of one condition, in one call."""
    eeg = np.stack([trial["eeg"] for trial in trials])
    mask = np.ones(eeg.shape[:2])
    start = time.perf_counter()
    outputs = proposal.apply(
        {
            "eeg": eeg,
            "channel_mask": mask,
            "checkpoint": checkpoint,
            "n_sources": condition.n_sources,
            "nms_radius_m": NMS_RADIUS_M,
        }
    )
    seconds = time.perf_counter() - start
    return (
        np.asarray(outputs["positions_m"]),
        np.asarray(outputs["moments"]),
        np.asarray(outputs["count_logits"]),
        seconds / max(len(trials), 1),
    )


def uninformed_start(
    condition: HybridCondition,
    trials: list[dict],
    containment: Containment,
    restart: int = 0,
) -> np.ndarray:
    """The initialization the frozen benchmark's ``far`` runs have always used.

    Drawn inside the containment ellipsoid and rejected if it lands within 30 mm
    of any truth. That rejection is what makes "far" far, and reusing the rule
    rather than inventing one is what keeps ``gradient`` a fair control.

    Only positions. Every refinement in this matrix warm-starts its moments in
    closed form (:func:`neurolayout.hybrid.refine.warm_start_moments`), so the
    difference between ``gradient`` and ``hybrid`` is the starting **position** and
    nothing else -- the network's moment head is deliberately not used at
    inference, which makes the ablation conservative.

    Args:
        condition: The cell.
        trials: Its trials.
        containment: The ellipsoid to draw inside.
        restart: Which independent start this is, for ``gradient_restarts``.
    """
    positions = []
    for trial in trials:
        rng = np.random.default_rng(
            condition.initialization_seed(trial["trial"]) + 7919 * restart
        )
        positions.append(
            random_initialization(containment, trial["truth_positions_m"], rng)
        )
    return np.stack(positions)


def run_condition(
    condition: HybridCondition,
    trials: list[dict],
    methods: list[str],
    *,
    headfield,
    proposal,
    dictionary,
    mne_inverse,
    config: LocalizeConfig,
    containment: Containment,
    checkpoint: str | None,
    stopgrad_checkpoint: str | None,
) -> list[dict]:
    """Run every requested method on every trial of one condition.

    The classical methods run per trial, because MNE's do. Every refinement is
    **batched**: a whole condition's trials go into one call, so 300 optimizer
    steps cost 300 BEM assemblies rather than 300 times the trial count. With the
    four-restart arm that is seven batched refinements per cell instead of seven
    times the trial count.
    """
    observed = np.stack([trial["eeg"] for trial in trials])
    truth = np.stack([trial["truth_positions_m"] for trial in trials])
    records: list[dict] = [
        {
            "key": trial["key"],
            "trial": trial["trial"],
            "condition": condition.name,
            "n_sources": condition.n_sources,
            "correlation": condition.correlation,
            "separation": condition.separation,
            "snr_db": condition.snr_db,
            "realized_correlation": trial["record"].get("realized_correlation"),
            "true_separation_mm": trial["record"].get("true_separation_mm"),
            "depth_mm": trial["record"].get("depth_mm"),
            "noise": trial["noise"],
            "methods": {},
        }
        for trial in trials
    ]

    proposed = proposed_moments = None
    if {"proposal", "hybrid"} & set(methods):
        proposed, proposed_moments, counts, seconds = run_proposal(
            proposal, trials, condition, checkpoint
        )
        if "proposal" in methods:
            for index, record in enumerate(records):
                record["methods"]["proposal"] = {
                    **score(proposed[index], truth[index]),
                    "seconds": seconds,
                    "positions_m": proposed[index].tolist(),
                    "predicted_k": int(np.argmax(counts[index]) + 1),
                    "sensor_residual": sensor_error(
                        headfield, proposed[index], proposed_moments[index],
                        observed[index], config,
                    ),
                }

    def record_refinement(name: str, result: dict, per_trial_seconds: float) -> None:
        for index, record in enumerate(records):
            positions = result["positions_m"][index]
            record["methods"][name] = {
                **score(positions, truth[index]),
                "seconds": per_trial_seconds,
                "positions_m": positions.tolist(),
                "sensor_residual": float(result["data_loss"][index]),
                "containment": float(result["containment"][index]),
                "separation_penalty": float(result["separation"][index]),
                "converged": (
                    None
                    if result.get("converged") is None
                    else bool(result["converged"][index])
                ),
            }

    if "gradient" in methods:
        start_positions = uninformed_start(condition, trials, containment)
        print("  gradient: refining from the uninformed start", flush=True)
        result = refine(
            headfield, observed, start_positions, None, config,
            containment, REFINE, truth_m=truth,
        )
        record_refinement("gradient", result, result["seconds"] / len(trials))
        for index, record in enumerate(records):
            record["methods"]["gradient"]["initial_error_mm"] = float(
                match_sources(start_positions[index], truth[index]).max_error_mm
            )

    if "gradient_restarts" in methods:
        # Selection is by data fit alone. A restart chosen by its distance to the
        # truth would be an oracle, and would answer a different question.
        attempts = []
        for restart in range(N_RESTARTS):
            starts = uninformed_start(condition, trials, containment, restart)
            print(f"  gradient_restarts: start {restart + 1}/{N_RESTARTS}", flush=True)
            attempts.append(
                refine(
                    headfield, observed, starts, None, config,
                    containment, REFINE, truth_m=truth,
                )
            )
        seconds = sum(attempt["seconds"] for attempt in attempts) / len(trials)
        best = np.argmin(
            np.stack([np.asarray(attempt["data_loss"]) for attempt in attempts]), axis=0
        )
        merged = {
            "converged": [
                attempts[best[index]]["converged"][index] for index in range(len(trials))
            ],
            "positions_m": np.stack(
                [attempts[best[index]]["positions_m"][index] for index in range(len(trials))]
            ),
            "data_loss": [
                attempts[best[index]]["data_loss"][index] for index in range(len(trials))
            ],
            "containment": [
                attempts[best[index]]["containment"][index] for index in range(len(trials))
            ],
            "separation": [
                attempts[best[index]]["separation"][index] for index in range(len(trials))
            ],
        }
        record_refinement("gradient_restarts", merged, seconds)
        for index, record in enumerate(records):
            record["methods"]["gradient_restarts"]["chosen_restart"] = int(best[index])
            record["methods"]["gradient_restarts"]["n_restarts"] = N_RESTARTS

    for name, source_checkpoint in (
        ("hybrid", checkpoint),
        ("hybrid_stopgrad", stopgrad_checkpoint),
    ):
        if name not in methods:
            continue
        if name == "hybrid":
            starts = proposed
        else:
            try:
                starts, _, _, _ = run_proposal(
                    proposal, trials, condition, source_checkpoint
                )
            except (FileNotFoundError, RuntimeError) as error:
                print(f"  {name}: skipped ({error})", flush=True)
                continue
        print(f"  {name}: refining from the proposal", flush=True)
        result = refine(
            headfield, observed, starts, None, config,
            containment, REFINE, truth_m=truth,
        )
        record_refinement(name, result, result["seconds"] / len(trials))
        for index, record in enumerate(records):
            record["methods"][name]["initial_error_mm"] = float(
                match_sources(starts[index], truth[index]).max_error_mm
            )

    if "rapmusic" in methods:
        print("  rapmusic", flush=True)
        for index, trial in enumerate(trials):
            result = rapmusic(
                mne_inverse,
                trial["eeg"],
                condition.n_sources,
                noise_rms_v=float(trial["noise"].get("noise_rms_v") or 0.0),
                signal_rms_v=float(np.sqrt(np.mean(trial["clean"] ** 2))),
            )
            records[index]["methods"]["rapmusic"] = {
                **score(result.positions_m, truth[index]),
                "seconds": result.seconds,
                "positions_m": np.asarray(result.positions_m).tolist(),
                "sensor_residual": result.residual_fraction,
                "n_found": result.n_found,
            }

    if "scan" in methods:
        print("  scan", flush=True)
        for index, trial in enumerate(trials):
            result = scan(dictionary, trial["eeg"], condition.n_sources)
            records[index]["methods"]["scan"] = {
                **score(result.positions_m, truth[index]),
                "seconds": result.seconds,
                "positions_m": np.asarray(result.positions_m).tolist(),
                "sensor_residual": result.residual_fraction,
                "n_found": result.n_found,
            }

    for record in records:
        summary = "  ".join(
            f"{name}={values['worst_mm']:.1f}"
            for name, values in sorted(record["methods"].items())
            if values.get("worst_mm") is not None
        )
        print(f"    {record['key']}: {summary}", flush=True)
    return records


def main() -> int:  # noqa: C901 - a benchmark runner is a sequence of stages
    """Run the selected conditions and methods, and write one shard per condition."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, default=DEFAULT_OBSERVATIONS)
    parser.add_argument("--out", type=Path, default=DEFAULT_SHARDS)
    parser.add_argument("--conditions", nargs="*", default=None)
    parser.add_argument("--methods", nargs="*", default=list(METHODS), choices=list(METHODS))
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--transport", default="local", choices=("local", "image"))
    parser.add_argument("--checkpoint", default=None,
                        help="proposal checkpoint name for `proposal` and `hybrid`")
    parser.add_argument("--stopgrad-checkpoint", default="proposal_stopgrad",
                        help="checkpoint name for `hybrid_stopgrad`")
    parser.add_argument("--dictionary-cache", type=Path,
                        default=REPO_ROOT / "results" / "dipole_dictionary.npz")
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()

    from neurolayout.clients import open_components

    manifest, stored = load_observations(arguments.observations)
    # The observations are bound to the *conditions*, which is all the generator
    # sees. The shards this run writes are bound to the full matrix, method list
    # included, which is what a result depends on.
    if manifest.get("conditions_fingerprint") != conditions_fingerprint():
        raise SystemExit(
            "the observations were built for conditions "
            f"{manifest.get('conditions_fingerprint')}, but this source declares "
            f"{conditions_fingerprint()}. Rebuild them with "
            "`python scripts/build_hybrid_observations.py`, or check out the commit "
            "the results belong to."
        )

    table = conditions_by_name()
    selected = (
        list(CONDITIONS)
        if arguments.conditions is None
        else [table[name] for name in arguments.conditions]
    )
    methods = [name for name in METHODS if name in set(arguments.methods)]

    path = default_artifact_path()
    geometry = HeadGeometry.load(path)
    containment = Containment.from_points(geometry.vertices[0])
    config = LocalizeConfig(
        backend="openmeeg",
        n_times=int(manifest["n_times"]),
        sfreq=float(manifest["sfreq"]),
    )

    dictionary = mne_inverse = None
    if {"scan", "rapmusic"} & set(methods):
        start = time.perf_counter()
        dictionary = DipoleDictionary(
            OpenMEEGForward(geometry, load_sensor_operator(path, geometry), reference=False),
            cache=arguments.dictionary_cache,
        )
        print(
            f"dipole dictionary: {dictionary.n_locations} locations "
            f"({time.perf_counter() - start:.0f} s"
            f"{', from cache' if dictionary.from_cache else ''})",
            flush=True,
        )
    if "rapmusic" in methods:
        start = time.perf_counter()
        mne_inverse = MneInverse(
            dictionary, tuple(CANONICAL_CHANNELS), sfreq=config.sfreq
        )
        print(f"mne forward: {time.perf_counter() - start:.0f} s", flush=True)

    needs_component = {"proposal", "hybrid", "hybrid_stopgrad"} & set(methods)
    needs_headfield = {
        "gradient", "gradient_restarts", "hybrid", "hybrid_stopgrad", "proposal"
    } & set(methods)
    components = ["headfield"] if needs_headfield or needs_component else []
    if needs_component:
        components.append("proposal")

    arguments.out.mkdir(parents=True, exist_ok=True)
    context = {
        "benchmark": "hybrid",
        "fingerprint": fingerprint(),
        "observations": str(arguments.observations),
        "observations_manifest": {
            key: value for key, value in manifest.items() if key != "trials"
        },
        "methods": methods,
        "refine": REFINE.to_dict(),
        "nms_radius_m": NMS_RADIUS_M,
        "checkpoint": arguments.checkpoint,
        "stopgrad_checkpoint": arguments.stopgrad_checkpoint,
        "transport": arguments.transport,
        "openmeeg_version": openmeeg_version(),
        "head_model_fingerprint": geometry.fingerprint(),
        "numpy_version": np.__version__,
        "python": platform.python_version(),
    }

    with open_components(components, arguments.transport) as opened:
        headfield = opened.get("headfield")
        proposal = opened.get("proposal")

        for condition in selected:
            shard = arguments.out / f"{condition.name}.json"
            if shard.exists() and not arguments.force:
                print(f"{shard.name}: already done, skipping", flush=True)
                continue
            trials = observe_condition(condition, stored, manifest)
            if arguments.trials is not None:
                trials = trials[: arguments.trials]
            if not trials:
                print(f"{condition.name}: no trials in the artifact", flush=True)
                continue
            print(
                f"\n=== {condition.name}  K={condition.n_sources} "
                f"{condition.correlation}  sep={condition.separation} "
                f"snr={condition.snr_db}  {len(trials)} trials",
                flush=True,
            )
            results = run_condition(
                condition,
                trials,
                methods,
                headfield=headfield,
                proposal=proposal,
                dictionary=dictionary,
                mne_inverse=mne_inverse,
                config=config,
                containment=containment,
                checkpoint=arguments.checkpoint,
                stopgrad_checkpoint=arguments.stopgrad_checkpoint,
            )
            payload = {
                **context,
                "condition": condition.to_dict(),
                "trials": results,
            }
            temporary = shard.with_suffix(".json.partial")
            temporary.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
            temporary.replace(shard)
            print(f"  -> {shard}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
