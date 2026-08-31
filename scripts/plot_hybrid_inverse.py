#!/usr/bin/env python
# Copyright 2026 NeuroLocate contributors
# SPDX-License-Identifier: Apache-2.0
r"""The four figures for the hybrid inverse.

1. ``hybrid_architecture.png`` — the two Tesseract boundaries, what is inside
   each, and the path one ``jax.grad`` takes back through them.
2. ``hybrid_ablation.png`` — the hard-case ablation: every method on every
   correlated and shared-dynamics trial, paired.
3. ``hybrid_k4_example.png`` — one difficult ``K = 4`` trial, in the head.
4. ``hybrid_refinement.png`` — the proposal before and after physics refinement,
   per trial and as a trajectory.
5. ``hybrid_k2_example.png`` — the same view of one difficult ``K = 2`` trial.

Figures 3 and 5 select their trial by the **gradient-only** method's error, so the
example is a case that was hard and not a case the composition happened to win.

Every panel plots individual trials rather than only summaries. At ``K = 4`` on
shared dynamics the tail *is* the result, and a median hides it.

Usage::

    python scripts/plot_hybrid_inverse.py
    python scripts/plot_hybrid_inverse.py --only 2 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "app"))
sys.path.insert(0, str(REPO_ROOT / "components" / "shared_code"))

from neurolayout.hybrid.report import load_shards  # noqa: E402
from neurolayout_shared.openmeeg_model import (  # noqa: E402
    HeadGeometry,
    default_artifact_path,
)

METHOD_COLORS = {
    "rapmusic": "#6b6b6b",
    "scan": "#9a9a9a",
    "gradient": "#c1121f",
    "gradient_restarts": "#e07a5f",
    "proposal": "#b06d00",
    "hybrid": "#2b4d8f",
    "hybrid_stopgrad": "#00707a",
}
METHOD_LABELS = {
    "rapmusic": "RAP-MUSIC",
    "scan": "scan",
    "gradient": "gradient only",
    "gradient_restarts": "gradient ×4 restarts",
    "proposal": "proposal only",
    "hybrid": "proposal + OpenMEEG",
    "hybrid_stopgrad": "stop-gradient control",
}
#: Two-line labels for the crowded categorical axis.
SHORT_LABELS = {
    "rapmusic": "RAP-MUSIC",
    "scan": "scan",
    "gradient": "gradient",
    "gradient_restarts": "gradient ×4",
    "proposal": "proposal",
    "hybrid": "proposal\n+ OpenMEEG",
    "hybrid_stopgrad": "stop-grad\ncontrol",
}
TRUE_COLOR = "#1b7f3b"
GROSS_MM = 20.0
VIEWS = (
    ((0, 2), "x (right) vs z (up)"),
    ((1, 2), "y (front) vs z (up)"),
)


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25})
    return plt


def _worst(record: dict, method: str) -> float | None:
    entry = record["methods"].get(method)
    if entry is None or entry.get("worst_mm") is None:
        return None
    return float(entry["worst_mm"])


def _hard(records: list[dict]) -> list[dict]:
    """The correlated and shared-dynamics trials — what the matrix exists for."""
    return [record for record in records if record["correlation"] != "distinct"]


def _present(records: list[dict]) -> list[str]:
    order = list(METHOD_COLORS)
    return [
        method
        for method in order
        if any(method in record["methods"] for record in records)
    ]


#
# 1. The architecture and the gradient path
#


def figure_architecture(path: Path, gradcheck: dict | None) -> None:
    """Draw the composition, and label each boundary with its own derivative rule.

    Hand-drawn boxes rather than a graph library: the point of the figure is which
    framework sits where and which arrow is which mechanism, and that is four
    boxes and six arrows.
    """
    plt = _plt()
    figure, axes = plt.subplots(figsize=(11.5, 5.6))
    axes.set_xlim(0, 11.5)
    axes.set_ylim(0, 5.6)
    axes.axis("off")
    axes.grid(False)

    boxes = [
        (0.15, 3.75, 2.0, 1.0, "observed EEG\n$y\\;[C, T]$", "#e8e8e8", "#444444"),
        (2.75, 3.75, 3.6, 1.0,
         "Tesseract  $\\bf{proposal}$\nPyTorch  ·  1.15 M parameters\n"
         "covariance $\\to$ heatmap $\\to$ $K$ sources", "#dfe8f5", "#2b4d8f"),
        (7.0, 3.75, 4.3, 1.0,
         "Tesseract  $\\bf{headfield}$\nOpenMEEG C++ symmetric BEM\n"
         "$G(p)\\,m$ — no AD framework inside", "#f5e6d5", "#b06d00"),
        (0.15, 0.75, 11.15, 0.85,
         "JAX / Optax   —   $L=\\|(I-P_{C(\\theta)})\\,y\\|^2/\\|y\\|^2$\n"
         "imports neither PyTorch nor OpenMEEG", "#e3f0e6", "#1b7f3b"),
    ]
    for x, y, width, height, text, fill, edge in boxes:
        axes.add_patch(
            plt.Rectangle(
                (x, y), width, height, facecolor=fill, edgecolor=edge, linewidth=1.6,
                zorder=2,
            )
        )
        axes.text(
            x + width / 2, y + height / 2, text, ha="center", va="center",
            fontsize=8.8, zorder=3,
        )

    forward = dict(arrowstyle="-|>", color="#444444", linewidth=1.7, mutation_scale=15)
    backward = dict(arrowstyle="-|>", color="#c1121f", linewidth=1.9, mutation_scale=15,
                    linestyle="--")
    axes.annotate("", xy=(2.72, 4.45), xytext=(2.18, 4.45), arrowprops=forward, zorder=4)
    axes.annotate("", xy=(6.97, 4.45), xytext=(6.38, 4.45), arrowprops=forward, zorder=4)
    axes.text(6.67, 4.62, "$p, m$", fontsize=8, ha="center", color="#444444")
    axes.annotate("", xy=(10.4, 1.64), xytext=(10.4, 3.73), arrowprops=forward, zorder=4)
    axes.text(10.5, 2.7, "predicted EEG", fontsize=8, color="#444444", va="center")

    axes.annotate("", xy=(8.1, 3.73), xytext=(8.1, 1.64), arrowprops=backward, zorder=4)
    axes.annotate("", xy=(6.41, 4.05), xytext=(6.94, 4.05), arrowprops=backward, zorder=4)
    axes.annotate("", xy=(2.21, 4.05), xytext=(2.72, 4.05), arrowprops=backward, zorder=4)
    axes.text(
        8.25, 2.7, "$\\nabla_\\theta L$", fontsize=10, color="#c1121f", va="center"
    )

    axes.text(
        0.15, 3.5,
        "One $\\nabla_\\theta L$, four derivative mechanisms with no framework in common:",
        fontsize=9.4, color="#c1121f", va="top",
    )
    labels = (
        ("① outer objective", "JAX", "#1b7f3b"),
        ("② source position", "central differences through the C++ BEM", "#b06d00"),
        ("③ source moment", "hand-written analytic algebra", "#b06d00"),
        ("④ network parameters", "torch.autograd", "#2b4d8f"),
    )
    for index, (stage, mechanism, color) in enumerate(labels):
        y = 3.16 - 0.32 * index
        axes.text(0.25, y, stage, fontsize=8.6, color=color, va="top", weight="bold")
        axes.text(2.35, y, f"— {mechanism}", fontsize=8.6, color="#333333", va="top")

    axes.text(
        0.15, 5.42,
        "The proposal network's weights are a differentiable $\\it{input}$ to its "
        "Tesseract, so $dL/d\\theta$ is an ordinary cotangent —\nthe same gradient "
        "that localizes a source is the one that trains the network.",
        fontsize=9, va="top",
    )
    if gradcheck:
        axes.text(
            11.35, 0.6,
            "A composed directional finite-difference check verifies this derivative "
            "path. Smallest recorded relative\ndiscrepancy in the configured check: "
            f"{gradcheck.get('best_relative_error', float('nan')):.2e}, over "
            f"{gradcheck.get('n_parameters', 0):,} network parameters on the real "
            "OpenMEEG BEM.\nThis tests derivative composition, not localization "
            "accuracy.",
            fontsize=8.3, ha="right", va="top", color="#1b7f3b",
        )
    figure.tight_layout()
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {path}")


#
# 2. The hard-case ablation
#


def figure_ablation(path: Path, records: list[dict]) -> None:
    """Every method on every correlated and shared trial, paired and per trial."""
    plt = _plt()
    hard = _hard(records)
    methods = _present(records)
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))

    # --- per-trial distributions ------------------------------------------
    for index, method in enumerate(methods):
        values = np.array(
            [_worst(record, method) for record in hard if _worst(record, method) is not None]
        )
        if values.size == 0:
            continue
        jitter = (np.random.default_rng(index).uniform(-0.16, 0.16, values.size))
        axes[0].scatter(
            index + jitter, values, s=16, alpha=0.55,
            color=METHOD_COLORS[method], edgecolors="none",
        )
        axes[0].plot(
            [index - 0.3, index + 0.3], [np.median(values)] * 2,
            color=METHOD_COLORS[method], linewidth=2.6,
        )
    axes[0].axhline(GROSS_MM, color="#888888", linestyle=":", linewidth=1.2)
    axes[0].text(
        len(methods) - 0.5, GROSS_MM * 1.06, "20 mm", fontsize=7.5,
        color="#888888", ha="right",
    )
    axes[0].set_xticks(range(len(methods)))
    axes[0].set_xticklabels(
        [SHORT_LABELS[m] for m in methods], fontsize=7.4, rotation=30, ha="right"
    )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("worst-source error, mm")
    axes[0].set_title(
        f"correlated and shared dynamics ({len(hard)} trials)\n"
        "one point per trial, bar is the median",
        fontsize=9,
    )

    # --- hybrid against each other method, paired -------------------------
    for method in methods:
        if method == "hybrid":
            continue
        pairs = [
            (_worst(record, method), _worst(record, "hybrid"))
            for record in hard
            if _worst(record, method) is not None and _worst(record, "hybrid") is not None
        ]
        if not pairs:
            continue
        other, ours = np.array(pairs).T
        axes[1].scatter(
            other, ours, s=18, alpha=0.6, color=METHOD_COLORS[method],
            edgecolors="none", label=METHOD_LABELS[method],
        )
    limits = (0.4, 200.0)
    axes[1].plot(limits, limits, color="#444444", linewidth=1.0, linestyle="--")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlim(limits)
    axes[1].set_ylim(limits)
    axes[1].set_xlabel("the other method, mm")
    axes[1].set_ylabel("proposal + OpenMEEG, mm")
    axes[1].set_title("below the line: the composition wins that trial", fontsize=9)
    axes[1].legend(fontsize=7, loc="upper left", framealpha=0.9)

    # --- by correlation regime --------------------------------------------
    regimes = ["distinct", "correlated", "shared"]
    width = 0.8 / max(len(methods), 1)
    for index, method in enumerate(methods):
        medians, positions = [], []
        for slot, regime in enumerate(regimes):
            values = [
                _worst(record, method)
                for record in records
                if record["correlation"] == regime and _worst(record, method) is not None
            ]
            if not values:
                continue
            medians.append(float(np.median(values)))
            positions.append(slot - 0.4 + width * (index + 0.5))
        axes[2].bar(
            positions, medians, width=width * 0.92,
            color=METHOD_COLORS[method], label=METHOD_LABELS[method],
        )
    axes[2].set_xticks(range(len(regimes)))
    axes[2].set_xticklabels(regimes)
    axes[2].margins(y=0.12)
    axes[2].set_ylabel("median worst-source error, mm")
    axes[2].set_title("the axis the matrix exists for", fontsize=9)
    axes[2].legend(fontsize=6.6, ncols=2)

    figure.suptitle(
        "The hard-case ablation: does the composition beat either half alone?",
        fontsize=11,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {path}")


#
# 3. One difficult K = 4 trial
#


def _hardest(records: list[dict], n_sources: int) -> dict | None:
    """The shared-dynamics trial at ``n_sources`` the gradient-only method failed worst on.

    Chosen by `gradient`'s error rather than by the hybrid's, so the example is
    "a case that was hard" and not "a case the method under test happened to
    win".
    """
    candidates = [
        record
        for record in records
        if record["n_sources"] == n_sources
        and record["correlation"] == "shared"
        and _worst(record, "gradient") is not None
        and _worst(record, "hybrid") is not None
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda record: _worst(record, "gradient"))


def truths_from_observations(path: Path) -> dict[str, np.ndarray]:
    """``{trial key: [K, 3] true positions}`` from the observations artifact.

    The shards do not carry the truth, and deliberately: a result file that holds
    the answer invites a reader to wonder whether the answer was consulted. The
    figures need it, so they read it from the artifact the *generator* wrote, on
    the other side of the estimator.
    """
    with np.load(path, allow_pickle=False) as data:
        return {
            key.rsplit("/", 1)[0]: np.asarray(data[key])
            for key in data.files
            if key.endswith("/truth_positions_m")
        }


def figure_hard_example(
    path: Path,
    records: list[dict],
    geometry: HeadGeometry,
    truths: dict[str, np.ndarray],
    n_sources: int,
) -> None:
    """One trial, two projections, every method's answer against the truth."""
    plt = _plt()
    record = _hardest(records, n_sources)
    if record is None:
        print(f"no K={n_sources} shared trial with both gradient and hybrid; skipping")
        return
    truth = truths.get(record["key"])
    if truth is None:
        print(f"no stored truth for {record['key']}; skipping")
        return
    methods = [m for m in _present(records) if m in record["methods"]]

    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))
    inner = np.asarray(geometry.vertices[0])
    for panel, ((first, second), label) in enumerate(VIEWS):
        axis = axes[panel]
        axis.scatter(
            inner[:, first] * 1e3, inner[:, second] * 1e3, s=1.5, color="#dddddd",
            edgecolors="none", zorder=1,
        )
        axis.scatter(
            truth[:, first] * 1e3, truth[:, second] * 1e3, s=150, marker="*",
            color=TRUE_COLOR, zorder=6, label="true sources", edgecolors="white",
            linewidths=0.6,
        )
        for method in methods:
            entry = record["methods"][method]
            positions = np.asarray(entry["positions_m"], dtype=float)
            if not np.isfinite(positions).all():
                continue
            axis.scatter(
                positions[:, first] * 1e3, positions[:, second] * 1e3, s=42,
                color=METHOD_COLORS[method], zorder=5, alpha=0.9,
                label=METHOD_LABELS[method] if panel == 0 else None,
                edgecolors="white", linewidths=0.5,
            )
        axis.set_xlabel(f"{label.split(' vs ')[0]}, mm")
        axis.set_ylabel(f"{label.split(' vs ')[1]}, mm")
        axis.set_aspect("equal")
        axis.set_title(label, fontsize=9)
    axes[0].legend(fontsize=7, loc="lower left", framealpha=0.9)

    # --- the error bar chart ----------------------------------------------
    values = [_worst(record, method) for method in methods]
    axes[2].barh(
        range(len(methods)), values,
        color=[METHOD_COLORS[method] for method in methods],
    )
    axes[2].set_yticks(range(len(methods)))
    axes[2].set_yticklabels([METHOD_LABELS[m] for m in methods], fontsize=8)
    axes[2].invert_yaxis()
    axes[2].axvline(GROSS_MM, color="#888888", linestyle=":", linewidth=1.2)
    axes[2].set_xlabel("worst-source error, mm")
    for index, value in enumerate(values):
        axes[2].text(value, index, f"  {value:.0f}", va="center", fontsize=8)
    axes[2].set_title(f"this trial, worst of {n_sources} sources", fontsize=9)

    separation = record.get("true_separation_mm")
    figure.suptitle(
        f"A difficult K = {n_sources} trial — {record['condition']}, trial "
        f"{record['trial']}, shared dynamics"
        + (f", closest pair {separation:.0f} mm apart" if separation else ""),
        fontsize=11,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {path}")


#
# 4. Before and after the physics refinement
#


def figure_refinement(path: Path, records: list[dict]) -> None:
    """What the OpenMEEG refinement did to the proposal, trial by trial."""
    plt = _plt()
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))

    pairs = [
        (_worst(record, "proposal"), _worst(record, "hybrid"), record)
        for record in records
        if _worst(record, "proposal") is not None and _worst(record, "hybrid") is not None
    ]
    if not pairs:
        print("no trial with both proposal and hybrid; skipping figure 4")
        plt.close(figure)
        return
    before = np.array([entry[0] for entry in pairs])
    after = np.array([entry[1] for entry in pairs])
    regimes = [entry[2]["correlation"] for entry in pairs]

    # --- before against after ---------------------------------------------
    styles = {"distinct": "#9a9a9a", "correlated": "#b06d00", "shared": "#2b4d8f"}
    for regime, color in styles.items():
        keep = [index for index, name in enumerate(regimes) if name == regime]
        if not keep:
            continue
        axes[0].scatter(
            before[keep], after[keep], s=22, alpha=0.7, color=color,
            edgecolors="none", label=regime,
        )
    limits = (0.4, max(200.0, float(max(before.max(), after.max())) * 1.2))
    axes[0].plot(limits, limits, color="#444444", linestyle="--", linewidth=1.0)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlim(limits)
    axes[0].set_ylim(limits)
    axes[0].set_xlabel("proposal alone, mm")
    axes[0].set_ylabel("after OpenMEEG refinement, mm")
    axes[0].set_title("below the line: the physics helped", fontsize=9)
    axes[0].legend(fontsize=7.5)

    # --- the paired change -------------------------------------------------
    for index in np.argsort(before):
        axes[1].plot(
            [0, 1], [before[index], after[index]],
            color=styles.get(regimes[index], "#888888"), alpha=0.5, linewidth=1.0,
        )
    axes[1].scatter([0] * len(before), before, s=14, color="#b06d00", zorder=3)
    axes[1].scatter([1] * len(after), after, s=14, color="#2b4d8f", zorder=3)
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["proposal", "+ OpenMEEG"])
    axes[1].set_xlim(-0.25, 1.25)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("worst-source error, mm")
    improved = float((after < before).mean())
    axes[1].set_title(
        f"per trial: {improved:.0%} improved, "
        f"median {np.median(before):.1f} → {np.median(after):.1f} mm",
        fontsize=9,
    )

    # --- the sensor residual, which is the thing being minimized -----------
    residual_before = np.array(
        [
            record["methods"]["proposal"].get("sensor_residual", np.nan)
            for _, _, record in pairs
        ],
        dtype=float,
    )
    residual_after = np.array(
        [
            record["methods"]["hybrid"].get("sensor_residual", np.nan)
            for _, _, record in pairs
        ],
        dtype=float,
    )
    finite = np.isfinite(residual_before) & np.isfinite(residual_after)
    if finite.any():
        axes[2].scatter(
            residual_before[finite], residual_after[finite], s=22, alpha=0.7,
            color="#1b7f3b", edgecolors="none",
        )
        span = (
            min(residual_before[finite].min(), residual_after[finite].min()) * 0.8,
            max(residual_before[finite].max(), residual_after[finite].max()) * 1.2,
        )
        axes[2].plot(span, span, color="#444444", linestyle="--", linewidth=1.0)
        axes[2].set_xscale("log")
        axes[2].set_yscale("log")
    axes[2].set_xlabel("proposal, relative sensor residual")
    axes[2].set_ylabel("after refinement")
    axes[2].set_title(
        "the objective the refinement actually minimizes\n"
        "(a fit can improve while the location does not)",
        fontsize=9,
    )

    figure.suptitle(
        "The proposal, before and after refinement through the C++ solver",
        fontsize=11,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {path}")


def main() -> int:
    """Draw the requested figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=Path,
                        default=REPO_ROOT / "results" / "hybrid" / "shards")
    parser.add_argument("--observations", type=Path,
                        default=REPO_ROOT / "results" / "hybrid" / "observations.npz")
    parser.add_argument("--gradcheck", type=Path,
                        default=REPO_ROOT / "results" / "hybrid" / "gradcheck.json")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "docs" / "figures")
    parser.add_argument("--only", nargs="*", type=int, default=None,
                        help="figure numbers to draw (default: all)")
    arguments = parser.parse_args()

    wanted = set(arguments.only) if arguments.only else {1, 2, 3, 4, 5}
    arguments.out_dir.mkdir(parents=True, exist_ok=True)
    gradcheck = (
        json.loads(arguments.gradcheck.read_text())
        if arguments.gradcheck.exists()
        else None
    )

    if 1 in wanted:
        figure_architecture(arguments.out_dir / "hybrid_architecture.png", gradcheck)
    if wanted - {1}:
        records = load_shards(arguments.shards)
        if not records:
            raise SystemExit(f"no shards in {arguments.shards}")
        geometry = HeadGeometry.load(default_artifact_path())
        if 2 in wanted:
            figure_ablation(arguments.out_dir / "hybrid_ablation.png", records)
        if wanted & {3, 5}:
            truths = truths_from_observations(arguments.observations)
        if 3 in wanted:
            figure_hard_example(
                arguments.out_dir / "hybrid_k4_example.png",
                records, geometry, truths, 4,
            )
        if 4 in wanted:
            figure_refinement(arguments.out_dir / "hybrid_refinement.png", records)
        if 5 in wanted:
            figure_hard_example(
                arguments.out_dir / "hybrid_k2_example.png",
                records, geometry, truths, 2,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
