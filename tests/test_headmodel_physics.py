# Copyright 2026 NeuroLocate contributors
# SPDX-License-Identifier: Apache-2.0
"""Physics regression tests for the fallback sphere solver and smooth sampler.

These guard the two things that could be quietly wrong without any gradient
check noticing: the analytic lead field, and the claim that softmax sampling
approximates the underlying field.
"""

from __future__ import annotations

import numpy as np
import pytest
from neurolayout_shared.geometry import fibonacci_directions, normalize
from neurolayout_shared.headmodel import (
    HeadModelSpec,
    get_head_model,
    load_head_model,
    save_head_model,
)
from neurolayout_shared.sampling import interpolation_weights, sample_lead_field
from neurolayout_shared.sphere_model import legendre_p_and_dp, sphere_lead_field

RADIUS = 0.09
SIGMA = 0.33


def test_legendre_matches_scipy() -> None:
    """The recurrence must reproduce the standard Legendre polynomials."""
    scipy_special = pytest.importorskip("scipy.special")
    x = np.linspace(-1.0, 1.0, 21)
    p, _ = legendre_p_and_dp(x, 12)
    for degree in range(1, 13):
        expected = scipy_special.eval_legendre(degree, x)
        np.testing.assert_allclose(p[degree - 1], expected, atol=1e-13)


def test_legendre_derivative_matches_finite_difference() -> None:
    """`dp` must be the derivative of `p`, not an unrelated recurrence."""
    x = np.linspace(-0.99, 0.99, 21)
    step = 1e-6
    _, dp = legendre_p_and_dp(x, 12)
    plus, _ = legendre_p_and_dp(x + step, 12)
    minus, _ = legendre_p_and_dp(x - step, 12)
    np.testing.assert_allclose(dp, (plus - minus) / (2 * step), atol=1e-6)


def test_centred_dipole_matches_closed_form() -> None:
    """As b -> 0 only the n=1 term survives: V = 3 (p.n) / (4 pi sigma R^2).

    This is the one configuration of the homogeneous sphere with an exactly known
    elementary answer, so it pins down every prefactor in the series at once.
    """
    directions = fibonacci_directions(64)
    moment = normalize(np.array([[0.3, -0.7, 0.5]]))
    potential = sphere_lead_field(
        directions, 1e-9 * moment, moment, radius=RADIUS, sigma=SIGMA
    )[:, 0]
    expected = 3.0 * (directions @ moment[0]) / (4.0 * np.pi * SIGMA * RADIUS**2)
    # Tolerance is relative to the field scale, not pointwise: `expected` passes
    # through zero where the observation direction is orthogonal to the moment.
    # The residual is the O(b/R) tail of the higher-degree terms, so it scales
    # linearly with the offset -- at b/R ~ 1e-8 it lands just under 1e-7.
    assert np.abs(potential - expected).max() < 1e-7 * np.abs(expected).max()


def test_series_is_converged_at_default_truncation() -> None:
    """The default truncation degree must not be the thing setting the answer."""
    directions = fibonacci_directions(64)
    position = np.array([[0.0, 0.0, 0.75 * RADIUS]])
    moment = np.array([[0.0, 0.0, 1.0]])
    common = dict(radius=RADIUS, sigma=SIGMA)
    coarse = sphere_lead_field(directions, position, moment, n_terms=80, **common)
    fine = sphere_lead_field(directions, position, moment, n_terms=200, **common)
    assert np.abs(coarse - fine).max() / np.abs(fine).max() < 1e-6


def test_no_monopole_term() -> None:
    """A current dipole injects no net charge, so the surface mean vanishes.

    The tolerance is set by the Fibonacci lattice as a quadrature rule, not by
    the solver, hence the loose bound.
    """
    model = get_head_model()
    column_rms = np.sqrt((model.lead_field**2).mean())
    assert np.abs(model.lead_field.mean(axis=0)).max() / column_rms < 1e-2


def test_source_at_or_beyond_scalp_is_rejected() -> None:
    """A source on the boundary makes the series diverge; fail loudly instead."""
    with pytest.raises(ValueError, match="strictly inside"):
        sphere_lead_field(
            fibonacci_directions(8),
            np.array([[0.0, 0.0, RADIUS]]),
            np.array([[0.0, 0.0, 1.0]]),
            radius=RADIUS,
            sigma=SIGMA,
        )


def test_head_model_is_deterministic_and_cached() -> None:
    """Rebuilding must be bit-identical, and the cache must actually hit."""
    first = get_head_model()
    second = get_head_model()
    assert first is second  # lru_cache
    fresh = get_head_model(HeadModelSpec())
    np.testing.assert_array_equal(fresh.lead_field, first.lead_field)


def test_head_model_roundtrips_through_npz(tmp_path) -> None:
    """The `.npz` artifact path is how the OpenMEEG lead field will arrive."""
    model = get_head_model(HeadModelSpec(n_scalp=42, n_sources=16))
    restored = load_head_model(save_head_model(model, tmp_path / "head.npz"))
    assert restored.spec == model.spec
    assert restored.solver_id == model.solver_id
    assert restored.lead_field_scale == pytest.approx(model.lead_field_scale)
    np.testing.assert_allclose(restored.lead_field, model.lead_field)


def test_interpolation_weights_are_a_partition_of_unity() -> None:
    """Softmax rows must sum to one, or the sampler rescales the lead field."""
    model = get_head_model(HeadModelSpec(n_scalp=162, n_sources=16))
    q = normalize(np.random.default_rng(0).standard_normal((7, 3)))
    weights = interpolation_weights(q, model.scalp_directions, kappa=60.0)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-12)
    assert (weights >= 0.0).all()


def test_sampling_at_a_lattice_vertex_approaches_that_vertex() -> None:
    """Large kappa must recover nearest-vertex lookup.

    This is the quantitative version of the claim in the smooth-sampling design
    note: the softmax is an interpolant of the dense field, not a different
    field. Checked at the vertices, where the exact answer is known.
    """
    model = get_head_model(HeadModelSpec(n_scalp=642, n_sources=16))
    vertices = model.scalp_directions[:32]
    sampled, _ = sample_lead_field(
        vertices, model.scalp_directions, model.lead_field, kappa=4000.0
    )
    exact = model.lead_field[:32]
    scale = np.sqrt((model.lead_field**2).mean())
    assert np.abs(sampled - exact).max() / scale < 1e-2


def test_sampling_error_decreases_with_kappa() -> None:
    """Interpolation must be a controlled approximation, not a fixed blur."""
    model = get_head_model(HeadModelSpec(n_scalp=642, n_sources=16))
    vertices = model.scalp_directions[:32]
    exact = model.lead_field[:32]
    errors = []
    for kappa in (20.0, 100.0, 500.0, 2500.0):
        sampled, _ = sample_lead_field(
            vertices, model.scalp_directions, model.lead_field, kappa=kappa
        )
        errors.append(np.abs(sampled - exact).max())
    assert errors == sorted(errors, reverse=True), errors
