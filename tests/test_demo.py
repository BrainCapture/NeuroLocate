# Copyright 2026 NeuroLocate contributors
# SPDX-License-Identifier: Apache-2.0
"""The judge-facing demo must reproduce the frozen trial.

``scripts/demo.py`` is the first thing a reader runs and the source of the three
numbers at the top of the README. It is run here as a subprocess, exactly as
``make demo`` runs it, and every error and residual it reports is checked against
the committed shard record for the same trial.

The tolerances are the ones ``scripts/build_hybrid_k2_visual.py`` uses for the
same replay: a re-run on the same host should be bit-identical, and this is room
for a different BLAS, not licence to publish a different result.

About a minute: two 300-step refinements through the real OpenMEEG BEM.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SHARD = REPO_ROOT / "results" / "hybrid" / "shards" / "h-k2-shared-close.json"
TRIAL_KEY = "h-k2-shared-close/t04"

#: Largest reproduction disagreement accepted, in millimetres and in relative
#: sensor residual.
POSITION_TOLERANCE_MM = 0.05
RESIDUAL_TOLERANCE = 1e-4


@pytest.fixture(scope="module")
def frozen() -> dict:
    """The committed record for the trial the demo runs."""
    shard = json.loads(SHARD.read_text())
    for trial in shard["trials"]:
        if trial["key"] == TRIAL_KEY:
            return trial
    raise AssertionError(f"no trial {TRIAL_KEY} in {SHARD.name}")


@pytest.fixture(scope="module")
def demo_output(tmp_path_factory) -> dict:
    """Run the demo once, as `make demo` does, and read back what it wrote."""
    out = tmp_path_factory.mktemp("demo") / "demo.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "demo.py"),
            "--json",
            str(out),
            "--no-gradcheck",
        ],
        capture_output=True,
        text=True,
        env={"OMP_NUM_THREADS": "8", "JAX_PLATFORMS": "cpu", "PATH": "/usr/bin:/bin"},
        cwd=REPO_ROOT,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert out.exists(), completed.stdout + completed.stderr
    return json.loads(out.read_text())


def test_the_demo_runs_the_trial_it_documents(demo_output) -> None:
    """A demo that quietly moved to an easier trial would be a different claim."""
    assert demo_output["condition"] == "h-k2-shared-close"
    assert demo_output["trial"] == 4
    assert demo_output["steps"] == 300


@pytest.mark.parametrize("method", ["proposal", "gradient", "hybrid"])
def test_the_demo_reproduces_the_frozen_error(demo_output, frozen, method) -> None:
    """Each arm's worst-source error must be the shard's."""
    measured = demo_output["methods"][method]["worst_mm"]
    expected = frozen["methods"][method]["worst_mm"]
    assert measured == pytest.approx(expected, abs=POSITION_TOLERANCE_MM), method


@pytest.mark.parametrize("method", ["gradient", "hybrid"])
def test_the_demo_reproduces_the_frozen_residual(demo_output, frozen, method) -> None:
    """And so must the sensor residual, which is half the point of the figure."""
    measured = demo_output["methods"][method]["sensor_residual"]
    expected = frozen["methods"][method]["sensor_residual"]
    assert measured == pytest.approx(expected, abs=RESIDUAL_TOLERANCE), method


def test_the_refinement_helps_and_the_two_answers_fit_alike(demo_output) -> None:
    """The ordering the README states, asserted rather than assumed."""
    methods = demo_output["methods"]
    assert methods["hybrid"]["worst_mm"] < methods["proposal"]["worst_mm"]
    assert methods["hybrid"]["worst_mm"] < methods["gradient"]["worst_mm"] / 10.0
    # The anatomically poor answer fits the measurement very slightly better.
    assert methods["gradient"]["sensor_residual"] < methods["hybrid"]["sensor_residual"]
    assert (
        abs(methods["gradient"]["sensor_residual"] - methods["hybrid"]["sensor_residual"])
        < 0.001
    )
