# Copyright 2026 NeuroLocate contributors
# SPDX-License-Identifier: Apache-2.0
r"""Reproducibility and provenance: a checkpoint, a shard and a report have to agree.

None of this is about physics. It is about the two ways a benchmark quietly stops
meaning anything: a result that cannot be traced to the configuration that
produced it, and a configuration that drifted after the result was recorded.

* A checkpoint has to carry everything needed to rebuild the module *and* the
  configuration that trained it, or "which run produced this number" has no
  answer.
* The component's architecture copy has to match the orchestrator's, or the
  served network is not the trained one.
* The reporter has to refuse shards from a different matrix, and refuse a partial
  matrix unless asked.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

import torch  # noqa: E402
from neurolayout.hybrid.benchmark import fingerprint  # noqa: E402
from neurolayout.hybrid.model import (  # noqa: E402
    ProposalConfig,
    ProposalNet,
    load_checkpoint,
    save_checkpoint,
)
from neurolayout.hybrid.report import (  # noqa: E402
    method_summary,
    paired_comparison,
    summarize,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _tiny_model(seed: int = 0) -> ProposalNet:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    return ProposalNet(
        ProposalConfig(n_channels=8, width=16, depth=1, heads=2, voxel_dim=4),
        rng.uniform(-0.03, 0.03, (12, 3)),
        rng.standard_normal((8, 3)) * 0.09,
    ).double()


#
# Checkpoints
#


def test_a_checkpoint_round_trips_bit_for_bit_in_float32(tmp_path) -> None:
    """Weights are stored as float32, so a reload has to be exactly that."""
    model = _tiny_model()
    path = save_checkpoint(tmp_path / "c.pt", model, {"note": "test"})
    reloaded, metadata = load_checkpoint(path)
    assert metadata["note"] == "test"
    for (name, original), (other_name, restored) in zip(
        sorted(model.state_dict().items()),
        sorted(reloaded.state_dict().items()),
        strict=True,
    ):
        assert name == other_name
        expected = original.float().double() if name not in (
            "voxel_centres", "sensor_xyz"
        ) else original
        torch.testing.assert_close(restored, expected, rtol=0, atol=0)


def test_a_checkpoint_carries_its_own_lattice_and_sensors(tmp_path) -> None:
    """The served component has only PyTorch, so it cannot rebuild either."""
    model = _tiny_model(3)
    path = save_checkpoint(tmp_path / "c.pt", model, {})
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["voxel_centres_m"].shape == (12, 3)
    assert payload["sensor_xyz_m"].shape == (8, 3)
    assert payload["config"]["n_channels"] == 8
    reloaded, _ = load_checkpoint(path)
    torch.testing.assert_close(reloaded.voxel_centres, model.voxel_centres)


def test_a_reloaded_checkpoint_gives_the_same_proposal(tmp_path) -> None:
    """The end of the chain: same weights in, same source set out."""
    model = _tiny_model(5).eval()
    eeg = torch.randn(2, 8, 16, dtype=torch.float64)
    mask = torch.ones(2, 8, dtype=torch.float64)
    reloaded, _ = load_checkpoint(save_checkpoint(tmp_path / "c.pt", model, {}))
    with torch.no_grad():
        original = model(eeg, mask)["logits"]
        restored = reloaded(eeg, mask)["logits"]
    torch.testing.assert_close(restored, original, rtol=1e-6, atol=1e-6)


def test_the_checkpoint_records_the_configuration_that_trained_it(tmp_path) -> None:
    """A number with no traceable configuration is not a result."""
    from neurolayout.hybrid.train import TrainConfig

    config = TrainConfig(steps=7, batch_size=3)
    path = save_checkpoint(tmp_path / "c.pt", _tiny_model(), {"config": config.to_dict()})
    _, metadata = load_checkpoint(path)
    assert metadata["config"]["steps"] == 7
    assert metadata["config"]["batch_size"] == 3
    assert "loss" in metadata["config"] and "spec" in metadata["config"]


#
# The component's copy of the architecture
#


def test_the_component_architecture_matches_the_orchestrator_s() -> None:
    """A served network that is not the trained one would be undetectable.

    The component image carries PyTorch alone and cannot import the orchestrator
    package, so the definition is duplicated. This is the check that keeps the
    duplicate honest.
    """
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "sync_proposal_component.py"), "--check"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_component_module_name_is_unique() -> None:
    """Every opened component's directory lands on the same sys.path.

    Two components sharing a top-level module name means whichever loads second
    silently gets the other's module. That is not hypothetical: it happened once
    in this project, between two components that both shipped a `model.py`.
    """
    root = REPO_ROOT / "components" / "tesseracts"
    modules: dict[str, str] = {}
    for component in sorted(path.name for path in root.iterdir() if path.is_dir()):
        for module in (root / component).glob("*.py"):
            if module.name == "tesseract_api.py":
                continue
            assert module.stem not in modules, (
                f"{component}/{module.name} collides with "
                f"{modules[module.stem]}/{module.name}"
            )
            modules[module.stem] = component


#
# The report
#


def _record(condition: str, method_errors: dict[str, float], **extra) -> dict:
    return {
        "condition": condition,
        "trial": 0,
        "n_sources": 2,
        "correlation": extra.get("correlation", "shared"),
        "true_separation_mm": extra.get("separation", 30.0),
        "methods": {
            name: {
                "worst_mm": value,
                "errors_mm": [value * 0.5, value],
                "sensor_residual": 0.1,
                "seconds": 1.0,
                "failed": False,
            }
            for name, value in method_errors.items()
        },
        "_shard": {"fingerprint": fingerprint()},
    }


def test_a_method_summary_reports_the_worst_source_not_the_mean() -> None:
    """A trial that misses one of two sources is a failed trial.

    Two trials, per-source errors [20, 40] and [1, 2]. The headline median is over
    the *worst* of each trial — median(40, 2) = 21 mm — while the per-source
    median pools all four and gives 11 mm. Both are reported, and the difference
    between them is the whole reason both are: 11 mm would describe this pair as
    a mild result when one of its two trials missed a source by four centimetres.
    """
    records = [_record("a", {"hybrid": 40.0}), _record("a", {"hybrid": 2.0})]
    summary = method_summary(records, "hybrid")
    assert summary["median_mm"] == pytest.approx(21.0)
    assert summary["per_source_median_mm"] == pytest.approx(11.0)
    assert summary["gross_failure_rate"] == pytest.approx(0.5)


def test_a_paired_comparison_is_signed_the_way_the_document_reads_it() -> None:
    """Negative means the first method is better; a sign error would invert every row."""
    records = [
        _record("a", {"hybrid": 5.0, "gradient": 30.0}),
        _record("a", {"hybrid": 8.0, "gradient": 20.0}),
    ]
    paired = paired_comparison(records, "hybrid", "gradient")
    assert paired["median_difference_mm"] < 0
    assert paired["better_fraction"] == pytest.approx(1.0)
    assert paired["n"] == 2


def test_a_paired_comparison_drops_and_counts_an_unpaired_trial() -> None:
    """A paired test needs a pair; a missing arm must not become a zero."""
    records = [
        _record("a", {"hybrid": 5.0, "gradient": 30.0}),
        _record("a", {"hybrid": 8.0}),
    ]
    paired = paired_comparison(records, "hybrid", "gradient")
    assert paired["n"] == 1
    assert paired["dropped"] == 1


def test_a_method_that_produced_nothing_is_counted_not_ignored() -> None:
    """Silently dropping a failed method would report it as never having run."""
    records = [
        _record("a", {"hybrid": 5.0}),
        {
            **_record("a", {"hybrid": 0.0}),
            "methods": {"hybrid": {"worst_mm": None, "failed": True}},
        },
    ]
    summary = method_summary(records, "hybrid")
    assert summary["attempted"] == 2
    assert summary["n"] == 1
    assert summary["failed"] == 1


def test_the_summary_splits_by_every_axis_the_document_reports() -> None:
    """The document's tables are read straight off this structure."""
    records = [
        _record("h-k2-shared", {"hybrid": 5.0, "gradient": 30.0}, correlation="shared"),
        _record(
            "h-k2-distinct", {"hybrid": 2.0, "gradient": 3.0}, correlation="distinct"
        ),
    ]
    report = summarize(records, ("hybrid", "gradient"))
    assert set(report["by_correlation"]) == {"shared", "distinct"}
    assert set(report["by_condition"]) == {"h-k2-shared", "h-k2-distinct"}
    assert "hybrid_vs_gradient" in report["paired"]
    assert report["paired_hard_cases"]["hybrid_vs_gradient"]["n"] == 1


def test_the_shard_fingerprint_is_the_one_this_source_declares() -> None:
    """The freeze binds the shards to the matrix that produced them."""
    shard = _record("a", {"hybrid": 1.0})["_shard"]
    assert shard["fingerprint"] == fingerprint()
    assert len(fingerprint()) == 64


def test_the_committed_freeze_matches_the_source(tmp_path) -> None:
    """If a freeze file is committed, it has to still describe this matrix."""
    path = REPO_ROOT / "results" / "hybrid_freeze.json"
    if not path.exists():
        pytest.skip("the matrix has not been frozen to a file yet")
    frozen = json.loads(path.read_text())
    assert frozen["fingerprint"] == fingerprint(), (
        "results/hybrid_freeze.json was committed for a different matrix; "
        "the results it guards belong to that one, not to this source"
    )


#
# Detection: the failures that do not show up as millimetres
#


def _detection_record(errors: list[float], separation: float | None = None) -> dict:
    """One trial whose matched per-source errors are given directly."""
    return {
        "condition": "a",
        "trial": 0,
        "n_sources": len(errors),
        "correlation": "shared",
        "true_separation_mm": 30.0,
        "methods": {
            "hybrid": {
                "worst_mm": max(errors),
                "errors_mm": errors,
                "min_separation_mm": separation,
                "failed": False,
            }
        },
        "_shard": {"fingerprint": fingerprint()},
    }


def test_recall_counts_sources_and_not_trials() -> None:
    """Two trials of two sources with one miss each is 50%, not 0%.

    The distinction matters at `K = 4`, where a per-trial rate says 97% failed and
    a per-source rate says half the sources were found — and only the second
    distinguishes "slightly wrong about everything" from "exactly right about
    most of it".
    """
    from neurolayout.hybrid.report import detection_summary

    records = [_detection_record([3.0, 40.0]), _detection_record([5.0, 60.0])]
    summary = detection_summary(records, "hybrid")
    assert summary["n_sources"] == 4
    assert summary["recall"] == pytest.approx(0.5)
    assert summary["missed_rate"] == pytest.approx(0.5)


def test_the_recovery_threshold_is_the_gross_failure_threshold() -> None:
    """One statement, not two thresholds a reader has to reconcile."""
    from neurolayout.hybrid.report import RECOVERED_MM, detection_summary

    assert RECOVERED_MM == 20.0
    just_inside = detection_summary([_detection_record([19.9])], "hybrid")
    just_outside = detection_summary([_detection_record([20.1])], "hybrid")
    assert just_inside["recall"] == pytest.approx(1.0)
    assert just_outside["recall"] == pytest.approx(0.0)


def test_a_merged_pair_is_counted_even_when_it_scores_well() -> None:
    """The whole reason merging needs its own column.

    Two estimates 4 mm apart, one of them 2 mm from a true source: the error
    column reports a good result and a true source has gone unexplained.
    """
    from neurolayout.hybrid.report import detection_summary

    collapsed = _detection_record([2.0, 3.0], separation=4.0)
    summary = detection_summary([collapsed], "hybrid")
    assert summary["merged_rate"] == pytest.approx(1.0)
    assert summary["recall"] == pytest.approx(1.0)


def test_well_separated_estimates_are_never_counted_as_merged() -> None:
    """A rate that fires on ordinary configurations would report nothing."""
    from neurolayout.hybrid.report import detection_summary

    summary = detection_summary(
        [_detection_record([2.0, 3.0], separation=45.0)], "hybrid"
    )
    assert summary["merged_rate"] == pytest.approx(0.0)


def test_one_source_is_never_merged_with_itself() -> None:
    """`K = 1` reports no separation at all, and that is not a collapse."""
    from neurolayout.hybrid.report import detection_summary

    summary = detection_summary([_detection_record([4.0], separation=None)], "hybrid")
    assert summary["merged_rate"] == pytest.approx(0.0)
    assert summary["recall"] == pytest.approx(1.0)


def test_a_method_that_produced_nothing_has_no_detection_summary() -> None:
    """An absent method must not be reported as having recovered nothing."""
    from neurolayout.hybrid.report import detection_summary

    assert detection_summary([_detection_record([1.0])], "rapmusic")["n_trials"] == 0
