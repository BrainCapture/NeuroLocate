# Copyright 2026 NeuroLocate contributors
# SPDX-License-Identifier: Apache-2.0
"""Gate B for the ``headfield`` Tesseract.

The headfield VJP is hand-written, so it is the single place in the project most
likely to be subtly wrong. Every differentiable input is checked against central
finite differences of the served ``apply`` endpoint — i.e. against the component
as the optimizer actually sees it, not against an inlined copy of the maths.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import (
    FD_TOLERANCE,
    central_difference,
    mask_from_indices,
    relative_error,
    sample_indices,
)

DIFFERENTIABLE_INPUTS = ("electrode_vectors", "source_activity")
DIFFERENTIABLE_OUTPUTS = ("eeg", "electrode_xyz")


@pytest.fixture(scope="module")
def problem(tiny_config):
    """A deterministic tiny primal point plus random output cotangents."""
    rng = np.random.default_rng(11)
    config = tiny_config
    return {
        "electrode_vectors": rng.standard_normal((config.n_electrodes, 3)),
        "source_activity": 0.2
        * rng.standard_normal((3, config.n_sources, config.n_times)),
        "cotangents": {
            "eeg": rng.standard_normal((3, config.n_electrodes, config.n_times)),
            "electrode_xyz": rng.standard_normal((config.n_electrodes, 3)),
        },
        "static": config.headfield_static(),
    }


def _apply(tesseracts, problem, **overrides) -> dict[str, np.ndarray]:
    inputs = {
        "electrode_vectors": problem["electrode_vectors"],
        "source_activity": problem["source_activity"],
        **problem["static"],
        **overrides,
    }
    return tesseracts.headfield.apply(inputs)


def test_apply_shapes_and_scalp_projection(tesseracts, problem, tiny_config) -> None:
    """`apply` must be schema-valid and put every electrode on the scalp."""
    out = _apply(tesseracts, problem)
    eeg = np.asarray(out["eeg"])
    xyz = np.asarray(out["electrode_xyz"])
    assert eeg.shape == (3, tiny_config.n_electrodes, tiny_config.n_times)
    assert xyz.shape == (tiny_config.n_electrodes, 3)
    assert np.isfinite(eeg).all()
    # Realized positions must lie on the scalp sphere regardless of input norms.
    from neurolayout_shared.headmodel import HeadModelSpec

    radius = HeadModelSpec().radius
    np.testing.assert_allclose(np.linalg.norm(xyz, axis=1), radius, rtol=1e-12)


def test_apply_is_invariant_to_design_vector_magnitude(tesseracts, problem) -> None:
    """Only the direction of `u` may matter; the optimizer relies on this."""
    baseline = _apply(tesseracts, problem)
    n_electrodes = problem["electrode_vectors"].shape[0]
    scales = np.geomspace(0.25, 17.0, n_electrodes)[:, None]
    scaled = _apply(
        tesseracts, problem, electrode_vectors=problem["electrode_vectors"] * scales
    )
    np.testing.assert_allclose(
        np.asarray(scaled["eeg"]), np.asarray(baseline["eeg"]), rtol=1e-12, atol=1e-14
    )


def test_zero_design_vector_is_rejected(tesseracts, problem) -> None:
    """A zero-norm design vector has no scalp direction; fail, don't guess."""
    vectors = problem["electrode_vectors"].copy()
    vectors[0] = 0.0
    with pytest.raises(Exception, match="norm"):
        _apply(tesseracts, problem, electrode_vectors=vectors)


def test_source_count_mismatch_is_rejected(tesseracts, problem) -> None:
    """Silently broadcasting a wrong source count would corrupt the physics."""
    with pytest.raises(Exception, match="sources"):
        _apply(
            tesseracts,
            problem,
            source_activity=problem["source_activity"][:, :-1, :],
        )


@pytest.mark.parametrize("input_name", DIFFERENTIABLE_INPUTS)
def test_vjp_matches_central_differences(tesseracts, problem, input_name) -> None:
    """Gate B: the analytic VJP must match finite differences of `apply`.

    A single scalar is formed by contracting *both* outputs with random
    cotangents, so `eeg` and `electrode_xyz` sensitivities are checked in the
    same pass — including the fact that `electrode_xyz` contributes an extra
    term to the `electrode_vectors` cotangent through `u -> u/||u||`.
    """
    cotangents = problem["cotangents"]

    def scalar(value: np.ndarray) -> float:
        out = _apply(tesseracts, problem, **{input_name: value})
        return float(
            sum(
                (np.asarray(out[name]) * cotangents[name]).sum()
                for name in DIFFERENTIABLE_OUTPUTS
            )
        )

    analytic = np.asarray(
        tesseracts.headfield.vector_jacobian_product(
            {
                "electrode_vectors": problem["electrode_vectors"],
                "source_activity": problem["source_activity"],
                **problem["static"],
            },
            vjp_inputs=list(DIFFERENTIABLE_INPUTS),
            vjp_outputs=list(DIFFERENTIABLE_OUTPUTS),
            cotangent_vector=cotangents,
        )[input_name]
    )

    primal = problem[input_name]
    indices = sample_indices(primal.shape, limit=48, seed=3)
    numeric = central_difference(scalar, primal, indices)
    mask = mask_from_indices(primal.shape, indices)

    assert analytic.shape == primal.shape
    assert relative_error(analytic, numeric, mask) < FD_TOLERANCE


def test_vjp_honours_a_partial_output_request(tesseracts, problem) -> None:
    """Requesting only `eeg` must drop the `electrode_xyz` contribution."""
    base_inputs = {
        "electrode_vectors": problem["electrode_vectors"],
        "source_activity": problem["source_activity"],
        **problem["static"],
    }
    eeg_only = np.asarray(
        tesseracts.headfield.vector_jacobian_product(
            base_inputs,
            vjp_inputs=["electrode_vectors"],
            vjp_outputs=["eeg"],
            cotangent_vector={"eeg": problem["cotangents"]["eeg"]},
        )["electrode_vectors"]
    )

    def scalar(value: np.ndarray) -> float:
        out = _apply(tesseracts, problem, electrode_vectors=value)
        return float((np.asarray(out["eeg"]) * problem["cotangents"]["eeg"]).sum())

    primal = problem["electrode_vectors"]
    indices = sample_indices(primal.shape, limit=48, seed=5)
    numeric = central_difference(scalar, primal, indices)
    mask = mask_from_indices(primal.shape, indices)
    assert relative_error(eeg_only, numeric, mask) < FD_TOLERANCE


def test_vjp_rejects_unknown_names(tesseracts, problem) -> None:
    """A non-differentiable input must not be accepted as a VJP target.

    ``kappa`` is declared non-differentiable, so Tesseract's generated VJP
    request schema rejects it before the endpoint runs.
    """
    with pytest.raises(Exception, match="kappa"):
        tesseracts.headfield.vector_jacobian_product(
            {
                "electrode_vectors": problem["electrode_vectors"],
                "source_activity": problem["source_activity"],
                **problem["static"],
            },
            vjp_inputs=["kappa"],
            vjp_outputs=["eeg"],
            cotangent_vector={"eeg": problem["cotangents"]["eeg"]},
        )
