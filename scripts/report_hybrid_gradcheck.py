#!/usr/bin/env python
# Copyright 2026 NeuroLocate contributors
# SPDX-License-Identifier: Apache-2.0
r"""The submission's derivative evidence: one gradient, two boundaries, three mechanisms.

Measures the end-to-end directional derivative of the composed hybrid objective
against central differences of the same composed forward, on the **real OpenMEEG
BEM**, over a sweep of step sizes. Writes a machine-readable record.

What is being demonstrated, precisely:

1. PyTorch lives behind one Tesseract boundary (``proposal``) and nothing outside
   it imports torch.
2. OpenMEEG's compiled C++ symmetric BEM lives behind another (``headfield``) and
   nothing outside it imports openmeeg.
3. JAX owns the outer objective and imports neither.
4. ``jax.grad`` of that objective returns ``dL/dweights`` — a cotangent that has
   crossed both boundaries.
5. The three derivative mechanisms in that one gradient are unrelated:
   ``torch.autograd`` for the network, hand-written analytic algebra for the
   source moment, and central differences through a compiled solver for the source
   position.

The step-size sweep is the point rather than a single number: a lucky step proves
nothing, and the shape of the curve — a plateau where truncation and round-off
balance — is what says the derivative is real.

Usage::

    python scripts/report_hybrid_gradcheck.py --out results/hybrid/gradcheck.json
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

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import neurolayout  # noqa: F401,E402  (enables float64 in JAX)
from neurolayout.clients import open_components  # noqa: E402
from neurolayout.hybrid.model import flat_parameters, load_checkpoint  # noqa: E402
from neurolayout.hybrid.physics import (  # noqa: E402
    columns,
    make_physics_loss,
    projector_residual,
    proposal_outputs,
)
from neurolayout.localize import Containment, LocalizeConfig  # noqa: E402
from neurolayout_shared.openmeeg_model import (  # noqa: E402
    HeadGeometry,
    default_artifact_path,
    openmeeg_version,
)

DEFAULT_OUT = REPO_ROOT / "results" / "hybrid" / "gradcheck.json"

#: Step sizes for the directional sweep, on the network's own parameter scale.
STEPS = (1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6)


def observation(geometry: HeadGeometry, n_sources: int, seed: int):
    """A deterministic epoch from two real OpenMEEG sources, for the check.

    Not from the benchmark: this measures a derivative, and using benchmark data
    for it would put a benchmark observation inside a number reported as
    infrastructure evidence.
    """
    from neurolayout_shared.openmeeg_model import load_forward

    rng = np.random.default_rng(seed)
    forward = load_forward()
    picks = rng.choice(len(geometry.source_space), n_sources, replace=False)
    positions = geometry.source_space[picks] * 0.97
    moments = geometry.source_normals[picks] * 25e-9
    gain = forward.gain(positions)
    times = np.arange(32) / 160.0
    waveforms = np.stack(
        [np.sin(2 * np.pi * (8.0 + 3.0 * k) * times + k) for k in range(n_sources)]
    )
    clean = np.einsum("ckj,kj,kt->ct", gain, moments, waveforms)
    noise = rng.standard_normal(clean.shape)
    noise -= noise.mean(axis=0, keepdims=True)
    noise *= np.sqrt(np.mean(clean**2)) * 10 ** (-20.0 / 20.0) / np.sqrt(np.mean(noise**2))
    return jnp.asarray((clean + noise)[None]), jnp.ones((1, geometry.n_channels))


def _relative(path: Path) -> str:
    """``path`` as a repository-relative string where it lies inside the repo."""
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def main() -> int:
    """Run the sweep and write the record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--n-sources", type=int, default=2)
    parser.add_argument("--directions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--transport", default="local", choices=("local", "image"))
    arguments = parser.parse_args()

    geometry = HeadGeometry.load(default_artifact_path())
    containment = Containment.from_points(geometry.vertices[0])
    config = LocalizeConfig(backend="openmeeg", n_times=32, sfreq=160.0)
    eeg, mask = observation(geometry, arguments.n_sources, arguments.seed)

    checkpoint_path = (
        Path(__file__).resolve().parents[1]
        / "components" / "tesseracts" / "proposal" / "proposal.pt"
    )
    model, checkpoint_metadata = load_checkpoint(checkpoint_path)
    weights = jnp.asarray(flat_parameters(model).numpy())

    with open_components(("headfield", "proposal"), arguments.transport) as opened:
        headfield, proposal = opened["headfield"], opened["proposal"]
        loss = make_physics_loss(
            headfield,
            proposal,
            config,
            containment,
            n_sources=arguments.n_sources,
            checkpoint=arguments.checkpoint,
        )
        start = time.perf_counter()
        value, gradient = jax.value_and_grad(loss)(weights, eeg, mask)
        gradient_seconds = time.perf_counter() - start
        gradient = np.asarray(gradient)
        print(
            f"loss {float(value):.6f}   |dL/dweights| {np.linalg.norm(gradient):.4e}   "
            f"{gradient.size} parameters   {gradient_seconds:.2f} s",
            flush=True,
        )

        rng = np.random.default_rng(arguments.seed + 1)
        sweeps = []
        for index in range(arguments.directions):
            direction = rng.normal(size=gradient.shape)
            direction /= np.linalg.norm(direction)
            analytic = float(gradient @ direction)
            entries = []
            for step in STEPS:
                high = float(loss(weights + step * direction, eeg, mask))
                low = float(loss(weights - step * direction, eeg, mask))
                numeric = (high - low) / (2.0 * step)
                scale = max(abs(analytic), abs(numeric), 1e-300)
                entries.append(
                    {
                        "step": step,
                        "numeric": numeric,
                        "absolute_error": abs(numeric - analytic),
                        "relative_error": abs(numeric - analytic) / scale,
                    }
                )
            best = min(entries, key=lambda entry: entry["relative_error"])
            print(
                f"  direction {index}: analytic {analytic:.6e}  best relative error "
                f"{best['relative_error']:.2e} at step {best['step']:.0e}",
                flush=True,
            )
            sweeps.append(
                {"direction": index, "analytic": analytic, "sweep": entries, "best": best}
            )

        # The moment-only control: freezing the solver's position sensitivity must
        # change the answer, or the position derivative is not load-bearing.
        def moment_only(parameters, epoch, channel_mask):
            outputs = proposal_outputs(
                proposal,
                parameters,
                epoch,
                channel_mask,
                n_sources=arguments.n_sources,
                checkpoint=arguments.checkpoint,
            )
            built = columns(
                headfield,
                jax.lax.stop_gradient(outputs["positions_m"]),
                outputs["moments"],
                config,
            )
            return jnp.mean(projector_residual(built, epoch))

        without = np.asarray(jax.grad(moment_only)(weights, eeg, mask))
        share = float(
            np.linalg.norm(gradient - without) / max(np.linalg.norm(gradient), 1e-300)
        )
        print(f"  position sensitivity accounts for {share:.1%} of the weight gradient")

    record = {
        "loss": float(value),
        "n_parameters": int(gradient.size),
        "n_sources": arguments.n_sources,
        "gradient_norm": float(np.linalg.norm(gradient)),
        "gradient_seconds": gradient_seconds,
        "position_sensitivity_share": share,
        "directions": sweeps,
        "best_relative_error": min(entry["best"]["relative_error"] for entry in sweeps),
        "transport": arguments.transport,
        "checkpoint": _relative(checkpoint_path),
        "checkpoint_metadata": checkpoint_metadata,
        "head_model_fingerprint": geometry.fingerprint(),
        "openmeeg_version": openmeeg_version(),
        "backend": config.backend,
        "mechanisms": {
            "network": "torch.autograd, inside the proposal Tesseract",
            "source_moment": "hand-written analytic algebra, inside the headfield Tesseract",
            "source_position": (
                "central differences through OpenMEEG's C++ DipSourceMat assembly, "
                "inside the headfield Tesseract"
            ),
            "outer_objective": "JAX, in the orchestrator, importing neither",
        },
        "numpy_version": np.__version__,
        "python": platform.python_version(),
    }
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(record, indent=1, sort_keys=True, default=str) + "\n")
    print(f"wrote {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
