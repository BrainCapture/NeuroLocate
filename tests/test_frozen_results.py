# Copyright 2026 NeuroLocate contributors
# SPDX-License-Identifier: Apache-2.0
"""The committed artifacts must still say what the documents quote.

Three separate things are checked here, none of which re-runs any physics:

* the frozen shards are the ones this source declares, are complete, and
  aggregate to the recovery rates, paired differences and runtimes the README
  and ``docs/BENCHMARK.md`` print;
* the deterministic demo trial's frozen record holds the three errors and two
  sensor residuals the README shows, so a documentation number and an artifact
  cannot drift apart silently;
* every number the README prints appears in the artifact it came from.

``tests/test_demo.py`` runs the demo itself; this file only checks that the
published numbers match what is committed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from neurolayout.hybrid.benchmark import CONDITIONS, fingerprint
from neurolayout.hybrid.report import (
    RECOVERED_MM,
    detection_summary,
    load_shards,
    method_summary,
    paired_comparison,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SHARDS = REPO_ROOT / "results" / "hybrid" / "shards"
README = REPO_ROOT / "README.md"

#: The demo's condition and trial, as ``scripts/demo.py`` names them.
DEMO_CONDITION = "h-k2-shared-close"
DEMO_TRIAL = 4

#: Per-source recovery at K=2, as the README's table prints it. Percent.
RECOVERY_K2 = {
    "gradient": 41.25,
    "rapmusic": 57.5,
    "gradient_restarts": 71.25,
    "proposal": 80.0,
    "hybrid": 82.5,
}

#: Paired median difference, hybrid minus the named method, on the correlated
#: and shared-dynamics cells. Millimetres, negative meaning hybrid is closer.
PAIRED_HARD_MM = {"proposal": -4.1, "gradient": -6.7, "rapmusic": -16.7}

#: Median seconds per trial over the whole matrix.
RUNTIME_S = {"gradient_restarts": 69.0, "hybrid": 16.8}

#: The K=4 result the README does not hide: RAP-MUSIC is this much closer.
K4_VS_RAPMUSIC_MM = 29.9


@pytest.fixture(scope="module")
def records() -> list[dict]:
    """Every trial record in the committed shards."""
    loaded = load_shards(SHARDS)
    assert loaded, f"no shards in {SHARDS}"
    return loaded


def test_the_shards_are_the_frozen_matrix(records) -> None:
    """A shard produced for a different matrix would invalidate every table."""
    stored = {record["_shard"]["fingerprint"] for record in records}
    assert stored == {fingerprint()}
    assert len(records) == sum(condition.n_trials for condition in CONDITIONS)


def test_the_k2_recovery_table_is_what_the_readme_prints(records) -> None:
    """The recovery rates are recomputed, not transcribed."""
    subset = [record for record in records if record["n_sources"] == 2]
    for method, expected in RECOVERY_K2.items():
        measured = detection_summary(subset, method)["recall"] * 100.0
        assert measured == pytest.approx(expected, abs=0.05), method


def test_the_paired_hard_case_differences_are_what_the_readme_prints(records) -> None:
    """The three paired medians quoted for the correlated and shared cells."""
    hard = [
        record for record in records if record["correlation"] in ("correlated", "shared")
    ]
    assert len(hard) == 56
    for method, expected in PAIRED_HARD_MM.items():
        paired = paired_comparison(hard, "hybrid", method)
        assert paired["median_difference_mm"] == pytest.approx(expected, abs=0.05), method
        assert paired["excludes_zero"], method


def test_the_k4_result_is_still_a_loss(records) -> None:
    """RAP-MUSIC is clearly ahead at K=4, and the repository must keep saying so."""
    subset = [record for record in records if record["n_sources"] == 4]
    paired = paired_comparison(subset, "hybrid", "rapmusic")
    assert paired["median_difference_mm"] == pytest.approx(K4_VS_RAPMUSIC_MM, abs=0.05)
    assert paired["median_difference_mm"] > 0.0


def test_the_runtime_comparison_is_what_the_readme_prints(records) -> None:
    """Four restarts cost about four times what the composition costs."""
    for method, expected in RUNTIME_S.items():
        measured = method_summary(records, method)["seconds_median"]
        assert measured == pytest.approx(expected, abs=0.05), method


def test_recovery_uses_the_gross_failure_threshold() -> None:
    """"Recovered" and "not a gross failure" must stay one statement, not two."""
    assert RECOVERED_MM == 20.0


@pytest.fixture(scope="module")
def demo_record() -> dict:
    """The frozen shard record for the trial the demo runs."""
    shard = json.loads((SHARDS / f"{DEMO_CONDITION}.json").read_text())
    key = f"{DEMO_CONDITION}/t{DEMO_TRIAL:02d}"
    for trial in shard["trials"]:
        if trial["key"] == key:
            return trial
    raise AssertionError(f"no trial {key} in {DEMO_CONDITION}.json")


@pytest.mark.parametrize(
    ("method", "worst_mm"),
    [("gradient", 124.3), ("proposal", 8.7), ("hybrid", 6.9)],
)
def test_the_demo_trial_holds_the_three_published_errors(
    demo_record, method, worst_mm
) -> None:
    """The three numbers the README's quickstart shows."""
    assert demo_record["methods"][method]["worst_mm"] == pytest.approx(worst_mm, abs=0.05)


def test_the_two_sensor_residuals_are_nearly_equal(demo_record) -> None:
    """The point of the hero visual: similar sensor fit, different anatomy.

    The anatomically poor answer fits the measurement very slightly *better*, so
    the ordering is asserted as well as the values. If a change ever flipped it,
    the figure's caption would be wrong.
    """
    gradient = demo_record["methods"]["gradient"]["sensor_residual"]
    hybrid = demo_record["methods"]["hybrid"]["sensor_residual"]
    assert gradient == pytest.approx(0.0117, abs=5e-5)
    assert hybrid == pytest.approx(0.0120, abs=5e-5)
    assert gradient < hybrid
    assert abs(gradient - hybrid) < 0.001


def test_the_trial_is_close_and_shares_one_time_course(demo_record) -> None:
    """The regime the demo claims to be in."""
    assert demo_record["n_sources"] == 2
    assert demo_record["correlation"] == "shared"
    assert demo_record["snr_db"] == pytest.approx(20.0)
    assert demo_record["true_separation_mm"] == pytest.approx(15.2, abs=0.05)


#
# The README's own numbers
#


def _readme() -> str:
    return README.read_text()


@pytest.mark.parametrize(
    "text",
    [
        "124.3 mm",
        "8.7 mm",
        "6.9 mm",
        "41.3%",
        "57.5%",
        "71.3%",
        "80.0%",
        "82.5%",
        "29.9 mm",
    ],
)
def test_the_readme_still_states_the_number(text) -> None:
    """A number removed from the README is a claim quietly changed."""
    assert text in _readme()


def test_the_readme_does_not_overclaim() -> None:
    """Phrases this repository has decided not to use about its own result."""
    banned = (
        "state of the art",
        "state-of-the-art",
        "global optimum",
        "solves the EEG inverse problem",
        "solves EEG source localization",
        "clinically",
        "well posed",
        "well-posed",
        "proves that",
    )
    text = _readme().lower()
    found = [phrase for phrase in banned if phrase in text]
    assert not found, found


def test_the_readme_gives_k_to_every_method() -> None:
    """K is an assumption handed to every estimator, and must be said so."""
    assert re.search(r"every method.{0,40}K|K.{0,60}every method", _readme())
