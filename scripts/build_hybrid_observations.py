#!/usr/bin/env python
# Copyright 2026 NeuroLocate contributors
# SPDX-License-Identifier: Apache-2.0
r"""Generate the hard-case benchmark's observations, with a forward model no estimator has.

The same discipline as ``scripts/build_observations.py`` and the same generator:
MNE-Python's linear-collocation BEM on ico4 surfaces, at a wrong skull
conductivity, with displaced electrodes. None of that appears in the OpenMEEG ico3
operator the estimators invert with, and none of it appears in the gain bank the
proposal network was trained on. There is no ``matched`` cell in this matrix at
all.

What differs from the older builder is the temporal structure. Its correlated
conditions were defined for two sources; this matrix needs a prescribed mutual
cosine across up to four, which is
:func:`neurolayout.hybrid.synth.correlated_waveforms`.

Usage::

    python scripts/build_hybrid_observations.py --out results/hybrid_observations.npz
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "app"))
sys.path.insert(0, str(REPO_ROOT / "components" / "shared_code"))

from neurolayout.benchmark import depth_mm, draw_truth, source_spacing_mm  # noqa: E402
from neurolayout.hybrid.benchmark import (  # noqa: E402
    CONDITIONS,
    HybridCondition,
    conditions_by_name,
    conditions_fingerprint,
)
from neurolayout.hybrid.synth import correlated_waveforms  # noqa: E402
from neurolayout.localize import LocalizeConfig  # noqa: E402
from neurolayout.matching import min_separation_mm  # noqa: E402
from neurolayout.mismatch import MISMATCH_LEVELS, IndependentForward  # noqa: E402
from neurolayout.montage import CANONICAL_CHANNELS  # noqa: E402
from neurolayout.waveforms import sample_waveform  # noqa: E402
from neurolayout_shared.openmeeg_model import (  # noqa: E402
    HeadGeometry,
    default_artifact_path,
)

DEFAULT_OUT = REPO_ROOT / "results" / "hybrid" / "observations.npz"


def trial_id(condition: str, trial: int) -> str:
    """The key one trial's arrays are stored under."""
    return f"{condition}/t{trial:02d}"


def trial_waveforms(
    condition: HybridCondition, trial: int, config: LocalizeConfig
) -> np.ndarray:
    """``[K, T]`` source time courses for one trial.

    ``distinct`` draws independently — which leaves a small *random* mutual
    correlation, as real independent sources would, rather than the exactly
    orthogonal set a prescribed cosine of zero would give. The realized
    correlation is measured and recorded either way.
    """
    rng = np.random.default_rng(condition.waveform_seed(trial))
    target = condition.correlation_value
    if target is None:
        return np.stack(
            [
                sample_waveform(rng, n_times=config.n_times, sfreq=config.sfreq)
                for _ in range(condition.n_sources)
            ]
        )
    return correlated_waveforms(
        rng,
        condition.n_sources,
        target,
        n_times=config.n_times,
        sfreq=config.sfreq,
    )


def mean_cosine(waveforms: np.ndarray) -> float | None:
    """Mean off-diagonal cosine of a set of time courses, or ``None`` for one."""
    if len(waveforms) < 2:
        return None
    unit = waveforms / np.maximum(np.linalg.norm(waveforms, axis=1, keepdims=True), 1e-30)
    gram = unit @ unit.T
    return float(np.abs(gram[~np.eye(len(waveforms), dtype=bool)]).mean())


def main() -> int:
    """Draw every trial, generate its clean EEG, and write the artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", nargs="*", default=None)
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--n-times", type=int, default=32)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    arguments = parser.parse_args()

    table = conditions_by_name()
    selected = (
        list(CONDITIONS)
        if arguments.conditions is None
        else [table[name] for name in arguments.conditions]
    )
    config = LocalizeConfig(n_times=arguments.n_times)
    geometry = HeadGeometry.load(default_artifact_path())

    by_generator: dict[str, list] = defaultdict(list)
    for condition in selected:
        n_trials = condition.n_trials if arguments.trials is None else arguments.trials
        for trial in range(n_trials):
            truth = draw_truth(
                geometry.source_space,
                geometry.source_normals,
                condition.n_sources,
                separation=condition.separation,
                seed=condition.truth_seed(trial),
            )
            waveforms = trial_waveforms(condition, trial, config)
            if waveforms.shape != (condition.n_sources, config.n_times):
                raise RuntimeError(
                    f"{condition.name}: waveforms {waveforms.shape} != "
                    f"{(condition.n_sources, config.n_times)}"
                )
            by_generator[condition.mismatch].append((condition, trial, truth, waveforms))

    payload: dict[str, np.ndarray] = {}
    records: dict[str, dict] = {}
    generators: dict[str, dict] = {}
    started = time.perf_counter()

    for mismatch, entries in by_generator.items():
        spec = MISMATCH_LEVELS[mismatch]
        if spec.is_matched:
            raise SystemExit(
                "this matrix has no matched cell by design; an inverse crime here "
                "would answer a different question"
            )
        print(f"generator {mismatch!r}: {spec.description}", flush=True)
        start = time.perf_counter()
        forward = IndependentForward(
            spec,
            tuple(CANONICAL_CHANNELS),
            geometry.sensor_xyz,
            geometry.vertices[2],
            sfreq=config.sfreq,
        )
        provenance = forward.provenance()
        build_seconds = time.perf_counter() - start
        print(
            f"  built in {build_seconds:.0f} s "
            f"(electrode RMS {provenance['realized_electrode_rms_mm']:.2f} mm)",
            flush=True,
        )

        positions = np.concatenate([truth.positions_m for _, _, truth, _ in entries])
        start = time.perf_counter()
        gain = forward.gain(positions)  # [C, sum K, 3]
        forward_seconds = time.perf_counter() - start
        print(
            f"  {len(entries)} trials, {positions.shape[0]} sources, "
            f"forward in {forward_seconds:.1f} s",
            flush=True,
        )

        offset = 0
        for condition, trial, truth, waveforms in entries:
            block = slice(offset, offset + truth.n_sources)
            offset += truth.n_sources
            clean = np.einsum(
                "ckj,kj,kt->ct", gain[:, block, :], truth.moments_am, waveforms
            )[None]
            if not np.isfinite(clean).all():
                raise RuntimeError(f"{condition.name}/t{trial}: non-finite clean signal")
            key = trial_id(condition.name, trial)
            payload[f"{key}/clean"] = clean
            payload[f"{key}/waveforms"] = waveforms
            payload[f"{key}/truth_positions_m"] = truth.positions_m
            payload[f"{key}/truth_moments_am"] = truth.moments_am
            payload[f"{key}/sensor_xyz"] = forward.sensor_xyz
            records[key] = {
                "condition": condition.name,
                "trial": trial,
                "truth": truth.to_dict(),
                "depth_mm": [depth_mm(geometry.vertices[0], p) for p in truth.positions_m],
                "true_separation_mm": (
                    None if truth.n_sources < 2 else min_separation_mm(truth.positions_m)
                ),
                "clean_rms_v": float(np.sqrt(np.mean(clean**2))),
                "realized_correlation": mean_cosine(waveforms),
                "generator": mismatch,
            }
        generators[mismatch] = {
            **provenance,
            "seconds_build": build_seconds,
            "seconds_forward": forward_seconds,
            "n_trials": len(entries),
        }

    manifest = {
        "benchmark": "hybrid",
        "conditions_fingerprint": conditions_fingerprint(),
        "n_times": config.n_times,
        "sfreq": config.sfreq,
        "channel_names": list(CANONICAL_CHANNELS),
        "coord_frame": "mne-head",
        "units": "m; eeg in volts",
        "head_model_fingerprint": geometry.fingerprint(),
        "source_spacing_mm": source_spacing_mm(geometry.source_space),
        "generators": generators,
        "conditions": {c.name: c.to_dict() for c in selected},
        "trials": records,
        "seconds_total": time.perf_counter() - started,
        "numpy_version": np.__version__,
        "python": platform.python_version(),
        "note": (
            "Noise is NOT included: apply the condition's NoiseSpec to `clean` to "
            "reproduce the observation the estimators saw."
        ),
    }
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        arguments.out, manifest=np.array(json.dumps(manifest, sort_keys=True)), **payload
    )
    print(
        f"wrote {arguments.out} ({arguments.out.stat().st_size / 1e6:.1f} MB, "
        f"{len(records)} trials) in {manifest['seconds_total']:.0f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
