# Copyright 2026 NeuroLocate contributors
# SPDX-License-Identifier: Apache-2.0
r"""Analytic spherical volume-conductor lead field (MVP fallback solver).

This is the project's *fallback* head solver. It exists so the
component contracts, VJPs, composition and tests can be made correct before
OpenMEEG is introduced. It is a **single homogeneous conducting sphere**, not a
BEM solve, and must never be described as one.

Physics
-------
For a homogeneous sphere of radius ``R`` and conductivity ``sigma``, the
potential on the outer surface produced by a current dipole at radius ``b < R``
has the classical spherical-harmonic (Legendre) series solution

.. math::

    V(R\hat n) = \frac{1}{4\pi\sigma R^2}
        \sum_{n=1}^{\infty} \frac{2n+1}{n} \left(\frac{b}{R}\right)^{n-1}
        \left[\, n\, p_r\, P_n(x) + (\mathbf p\cdot\hat n - p_r x)\, P_n'(x) \right]

where :math:`\hat n` is the unit direction of the observation point,
:math:`\hat e_r = \mathbf r_0 / b` the dipole's radial direction,
:math:`x = \hat n\cdot\hat e_r`, :math:`p_r = \mathbf p\cdot\hat e_r` the
radial dipole component, and :math:`P_n` the Legendre polynomials.

The bracketed tangential term is written so that the usual :math:`\sin\gamma`
in the numerator and denominator cancel analytically, which removes the
coordinate singularity at the poles and makes the expression safe to evaluate
for every scalp/source pair at once.

Two exactly-known limits are used as regression tests (see
``tests/test_headmodel_physics.py``):

* ``b -> 0`` (centred dipole) collapses to the single ``n = 1`` term and must
  reproduce :math:`V = 3(\mathbf p\cdot\hat n) / (4\pi\sigma R^2)`.
* the series contains no ``n = 0`` term, so the potential integrates to zero
  over the sphere; the mean over a quasi-uniform scalp lattice must vanish.

Two roles
---------
:func:`sphere_lead_field` produces the dense lead-field matrix that the montage
sampler interpolates. :func:`sphere_source_gain` exposes the *same* free-orientation
gain interface as the OpenMEEG backend, so NeuroLocate's source-localization path
can run on either solver without changing a line above the backend boundary —
which is what makes the sphere a real fallback rather than a decoration, and
what lets the two solvers be compared against each other.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import fibonacci_directions, normalize

__all__ = [
    "legendre_p_and_dp",
    "sphere_lead_field",
    "SphereHead",
    "DEFAULT_SPHERE_HEAD",
    "sphere_source_gain",
]


def legendre_p_and_dp(x: np.ndarray, n_terms: int) -> tuple[np.ndarray, np.ndarray]:
    """Legendre polynomials and their derivatives for degrees ``1..n_terms``.

    Args:
        x: Array of arguments in ``[-1, 1]`` of any shape.
        n_terms: Highest degree to return.

    Returns:
        ``(p, dp)``, each of shape ``(n_terms, *x.shape)``, where ``p[i]`` is
        :math:`P_{i+1}(x)` and ``dp[i]`` is :math:`P_{i+1}'(x)`.

    Uses the standard three-term recurrences
    ``(n+1) P_{n+1} = (2n+1) x P_n - n P_{n-1}`` and
    ``P'_{n+1} = (2n+1) P_n + P'_{n-1}``. The derivative recurrence is used in
    preference to ``(1-x^2) P'_n = n (P_{n-1} - x P_n)`` because it stays exact
    at ``x = +/-1``.
    """
    if n_terms < 1:
        raise ValueError(f"n_terms must be >= 1, got {n_terms}")
    x = np.asarray(x, dtype=np.float64)

    p = np.empty((n_terms + 1, *x.shape), dtype=np.float64)
    dp = np.empty((n_terms + 1, *x.shape), dtype=np.float64)
    p[0] = 1.0  # P_0
    dp[0] = 0.0  # P_0'
    if n_terms >= 1:
        p[1] = x  # P_1
        dp[1] = 1.0  # P_1'
    for n in range(1, n_terms):
        p[n + 1] = ((2 * n + 1) * x * p[n] - n * p[n - 1]) / (n + 1)
        dp[n + 1] = (2 * n + 1) * p[n] + dp[n - 1]
    # Drop degree 0: it carries no dipolar contribution.
    return p[1:], dp[1:]


def sphere_lead_field(
    scalp_directions: np.ndarray,
    source_positions: np.ndarray,
    source_moments: np.ndarray,
    *,
    radius: float,
    sigma: float,
    n_terms: int = 80,
) -> np.ndarray:
    """Dense lead field of a homogeneous sphere, evaluated at scalp directions.

    Args:
        scalp_directions: ``[J, 3]`` observation directions (normalized here).
        source_positions: ``[S, 3]`` dipole positions, in the same length units
            as ``radius``. Every radius must satisfy ``0 <= b < radius``.
        source_moments: ``[S, 3]`` dipole moment vectors.
        radius: Outer sphere radius ``R``.
        sigma: Uniform conductivity.
        n_terms: Legendre series truncation degree.

    Returns:
        ``[J, S]`` lead field: entry ``(j, s)`` is the surface potential at
        scalp direction ``j`` per unit amplitude of source ``s``.
    """
    scalp_unit = normalize(scalp_directions)  # [J, 3]
    positions = np.asarray(source_positions, dtype=np.float64)  # [S, 3]
    moments = np.asarray(source_moments, dtype=np.float64)  # [S, 3]
    if positions.shape != moments.shape:
        raise ValueError(
            f"source_positions {positions.shape} and source_moments "
            f"{moments.shape} must have the same shape"
        )

    b = np.linalg.norm(positions, axis=1)  # [S]
    if np.any(b >= radius):
        raise ValueError("all sources must lie strictly inside the sphere")
    source_unit = normalize(positions)  # [S, 3]

    x = np.clip(scalp_unit @ source_unit.T, -1.0, 1.0)  # [J, S] = cos(gamma)
    p_radial = np.einsum("sd,sd->s", moments, source_unit)  # [S]
    p_dot_n = scalp_unit @ moments.T  # [J, S]

    p, dp = legendre_p_and_dp(x, n_terms)  # [N, J, S] each
    degrees = np.arange(1, n_terms + 1, dtype=np.float64)  # [N]
    # (2n+1)/n * (b/R)^(n-1): depends on degree and on the source radius only.
    weights = ((2.0 * degrees + 1.0) / degrees)[:, None] * (
        (b[None, :] / radius) ** (degrees[:, None] - 1.0)
    )  # [N, S]

    radial_term = degrees[:, None, None] * p_radial[None, None, :] * p  # [N, J, S]
    tangential_term = (p_dot_n - p_radial[None, :] * x)[None] * dp  # [N, J, S]
    series = np.einsum("ns,njs->js", weights, radial_term + tangential_term)

    return series / (4.0 * np.pi * sigma * radius**2)


@dataclass(frozen=True)
class SphereHead:
    """A self-contained spherical head for the source-localization fallback.

    Deliberately depends on nothing on disk: no BEM artifact, no OpenMEEG, no
    downloaded anatomy. That is what makes it usable as the kill-rule fallback
    and as the backend for tests that must run in a bare checkout.

    Attributes:
        radius: Sphere radius in metres.
        sigma: Uniform conductivity in S/m.
        n_channels: Number of sensors, placed on a spherical Fibonacci lattice
            so the layout is deterministic and quasi-uniform.
        n_terms: Legendre truncation degree.
    """

    radius: float = 0.09
    sigma: float = 0.33
    n_channels: int = 64
    n_terms: int = 120

    def sensor_xyz(self) -> np.ndarray:
        """``[C, 3]`` sensor positions on the sphere, metres."""
        return self.radius * fibonacci_directions(self.n_channels)

    def reference_operator(self) -> np.ndarray:
        """``R = I - 11ᵀ/C`` over this sensor set."""
        n = self.n_channels
        return np.eye(n, dtype=np.float64) - np.full((n, n), 1.0 / n)


#: The spherical head every fallback path defaults to.
DEFAULT_SPHERE_HEAD = SphereHead()


def sphere_source_gain(
    head: SphereHead,
    positions: np.ndarray,
    *,
    reference: bool = True,
) -> np.ndarray:
    """Free-orientation gain of the analytic sphere, matching the OpenMEEG API.

    Args:
        head: The spherical head model.
        positions: ``[P, 3]`` dipole positions, metres, strictly inside the
            sphere.
        reference: Apply the average-reference operator, as the BEM backend does.

    Returns:
        ``[C, P, 3]`` volts per A·m — the same layout
        :meth:`neurolayout_shared.openmeeg_model.OpenMEEGForward.gain` returns,
        so the two backends are drop-in substitutes for each other.
    """
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must be [P, 3], got {positions.shape}")
    directions = normalize(head.sensor_xyz())
    columns = [
        sphere_lead_field(
            directions,
            positions,
            np.tile(axis, (positions.shape[0], 1)),
            radius=head.radius,
            sigma=head.sigma,
            n_terms=head.n_terms,
        )
        for axis in np.eye(3)
    ]
    gain = np.stack(columns, axis=-1)  # [C, P, 3]
    if reference:
        gain = np.einsum("cd,dpj->cpj", head.reference_operator(), gain)
    return gain
