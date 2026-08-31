#!/usr/bin/env python3
# Copyright 2026 NeuroLocate contributors
# SPDX-License-Identifier: Apache-2.0
"""Regenerate the checked-in Tesseract regression test cases.

Each component's ``test_cases/*.json`` is a frozen input/output pair, verified by
``Tesseract.test`` (see ``tests/test_component_cases.py``) and by
``tesseract run <image> test @<file>`` once images are built. They are the Gate A
determinism check: if a refactor changes a number, these fail.

Regenerate with::

    python scripts/gen_test_cases.py

and review the diff before committing. Regenerating is only correct when the
change in behaviour is intended.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tesseract_core import Tesseract

REPO_ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = REPO_ROOT / "components" / "tesseracts"

# Deliberately tiny and deliberately fixed: these are determinism anchors, not
# performance benchmarks.
N_ELECTRODES, N_TIMES, BATCH = 3, 4, 1
N_SCALP, N_SOURCES = 42, 16

#: The frozen 64-channel montage the proposal network was trained on, and the
#: shortest epoch that still defines a covariance. The network reads only
#: ``Y Yᵀ``, so one sample is a valid epoch and keeps the frozen JSON small.
PROPOSAL_CHANNELS, PROPOSAL_TIMES, PROPOSAL_SOURCES = 64, 1, 2

def _as_list(array: np.ndarray) -> list:
    return np.asarray(array, dtype=np.float64).tolist()


def headfield_inputs() -> dict:
    """Fixed tiny `headfield` payload."""
    rng = np.random.default_rng(2026)
    return {
        "electrode_vectors": _as_list(rng.standard_normal((N_ELECTRODES, 3))),
        "source_activity": _as_list(
            0.2 * rng.standard_normal((BATCH, N_SOURCES, N_TIMES))
        ),
        "kappa": 40.0,
        "n_scalp": N_SCALP,
        "n_sources": N_SOURCES,
    }


def localize_inputs() -> dict:
    """Fixed tiny ``mode="localize"`` payload on the OpenMEEG template head.

    Two sources rather than one: the K>1 plumbing is exercised long before K>1
    is claimed as a result, so a regression in the batching cannot hide.
    """
    rng = np.random.default_rng(4242)
    return {
        "mode": "localize",
        "backend": "openmeeg",
        "source_positions": [[-0.037, 0.023, 0.036], [0.028, -0.041, 0.048]],
        "source_timecourses": _as_list(2e-8 * rng.standard_normal((1, 2, 3, N_TIMES))),
    }


def write(path: Path, case: dict) -> None:
    """Write one test case as pretty-printed JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(case, indent=2) + "\n")
    print(f"wrote {path.relative_to(REPO_ROOT)}")


def main() -> None:
    """Regenerate every checked-in test case."""
    rng = np.random.default_rng(777)
    hf_inputs = headfield_inputs()

    with Tesseract.from_tesseract_api(COMPONENTS / "headfield" / "tesseract_api.py") as hf:
        applied = hf.apply(hf_inputs)
        eeg = np.asarray(applied["eeg"])
        electrode_xyz = np.asarray(applied["electrode_xyz"])

        write(
            COMPONENTS / "headfield" / "test_cases" / "apply.json",
            {
                "endpoint": "apply",
                "payload": {"inputs": hf_inputs},
                "expected_outputs": {
                    "eeg": _as_list(eeg),
                    "electrode_xyz": _as_list(electrode_xyz),
                },
                "atol": 1e-10,
                "rtol": 1e-8,
            },
        )

        hf_cotangent = {
            "eeg": _as_list(rng.standard_normal(eeg.shape)),
            "electrode_xyz": _as_list(rng.standard_normal(electrode_xyz.shape)),
        }
        hf_vjp = hf.vector_jacobian_product(
            hf_inputs,
            vjp_inputs=["electrode_vectors", "source_activity"],
            vjp_outputs=["eeg", "electrode_xyz"],
            cotangent_vector=hf_cotangent,
        )
        write(
            COMPONENTS / "headfield" / "test_cases" / "vector_jacobian_product.json",
            {
                "endpoint": "vector_jacobian_product",
                "payload": {
                    "inputs": hf_inputs,
                    "vjp_inputs": ["electrode_vectors", "source_activity"],
                    "vjp_outputs": ["eeg", "electrode_xyz"],
                    "cotangent_vector": hf_cotangent,
                },
                "expected_outputs": {
                    name: _as_list(hf_vjp[name])
                    for name in ("electrode_vectors", "source_activity")
                },
                "atol": 1e-10,
                "rtol": 1e-8,
            },
        )

        # --- mode="localize", backend="openmeeg" ---------------------------
        loc_inputs = localize_inputs()
        loc_applied = hf.apply(loc_inputs)
        loc_eeg = np.asarray(loc_applied["eeg"])
        write(
            COMPONENTS / "headfield" / "test_cases" / "localize_apply.json",
            {
                "endpoint": "apply",
                "payload": {"inputs": loc_inputs},
                "expected_outputs": {
                    "eeg": _as_list(loc_eeg),
                    "electrode_xyz": _as_list(loc_applied["electrode_xyz"]),
                },
                # Volts, so the numbers are ~1e-7: an absolute tolerance has to
                # be scaled to them or it is vacuous.
                "atol": 1e-18,
                "rtol": 1e-8,
            },
        )

        # A dedicated generator, so adding this case does not shift the draws
        # that the pre-existing frozen cases were generated with.
        loc_rng = np.random.default_rng(8080)
        loc_cotangent = {"eeg": _as_list(loc_rng.standard_normal(loc_eeg.shape))}
        loc_vjp = hf.vector_jacobian_product(
            loc_inputs,
            vjp_inputs=["source_positions", "source_timecourses"],
            vjp_outputs=["eeg"],
            cotangent_vector=loc_cotangent,
        )
        write(
            COMPONENTS / "headfield" / "test_cases" / "localize_vector_jacobian_product.json",
            {
                "endpoint": "vector_jacobian_product",
                "payload": {
                    "inputs": loc_inputs,
                    "vjp_inputs": ["source_positions", "source_timecourses"],
                    "vjp_outputs": ["eeg"],
                    "cotangent_vector": loc_cotangent,
                },
                "expected_outputs": {
                    name: _as_list(loc_vjp[name])
                    for name in ("source_positions", "source_timecourses")
                },
                "atol": 1e-18,
                "rtol": 1e-6,
            },
        )

        degenerate = json.loads(json.dumps(hf_inputs))
        degenerate["electrode_vectors"][0] = [0.0, 0.0, 0.0]
        write(
            COMPONENTS / "headfield" / "test_cases" / "apply_zero_vector_raises.json",
            {
                "endpoint": "apply",
                "payload": {"inputs": degenerate},
                "expected_exception": "ValueError",
                "expected_exception_regex": "norm",
            },
        )

    # --- proposal ---------------------------------------------------------
    #
    # The epoch is unit-scale rather than volt-scale on purpose: the network
    # trace-normalizes the covariance, so the absolute amplitude is not part of
    # its input, and unit-scale numbers keep the frozen tolerances meaningful
    # instead of vacuous against 1e-6 magnitudes.
    prop_rng = np.random.default_rng(20260825)
    prop_inputs = {
        "eeg": _as_list(
            prop_rng.standard_normal((BATCH, PROPOSAL_CHANNELS, PROPOSAL_TIMES))
        ),
        "n_sources": PROPOSAL_SOURCES,
    }
    with Tesseract.from_tesseract_api(
        COMPONENTS / "proposal" / "tesseract_api.py"
    ) as prop:
        prop_applied = prop.apply(prop_inputs)
        write(
            COMPONENTS / "proposal" / "test_cases" / "apply.json",
            {
                "endpoint": "apply",
                "payload": {"inputs": prop_inputs},
                "expected_outputs": {
                    name: _as_list(prop_applied[name])
                    for name in ("positions_m", "moments", "scores", "count_logits")
                },
                "atol": 1e-12,
                "rtol": 1e-8,
            },
        )

        # Differentiated with respect to the epoch, not the flattened weights:
        # `weights` is 1.15 M numbers, and a frozen JSON carrying that cotangent
        # would be a hundred megabytes for no extra coverage of the VJP itself.
        # Gate M is where dL/dweights is checked.
        positions = np.asarray(prop_applied["positions_m"])
        prop_cotangent = {
            "positions_m": _as_list(prop_rng.standard_normal(positions.shape))
        }
        prop_vjp = prop.vector_jacobian_product(
            prop_inputs,
            vjp_inputs=["eeg"],
            vjp_outputs=["positions_m"],
            cotangent_vector=prop_cotangent,
        )
        write(
            COMPONENTS / "proposal" / "test_cases" / "vector_jacobian_product.json",
            {
                "endpoint": "vector_jacobian_product",
                "payload": {
                    "inputs": prop_inputs,
                    "vjp_inputs": ["eeg"],
                    "vjp_outputs": ["positions_m"],
                    "cotangent_vector": prop_cotangent,
                },
                "expected_outputs": {"eeg": _as_list(prop_vjp["eeg"])},
                "atol": 1e-12,
                "rtol": 1e-8,
            },
        )


if __name__ == "__main__":
    main()
