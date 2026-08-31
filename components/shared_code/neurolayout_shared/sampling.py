# Copyright 2026 NeuroLocate contributors
# SPDX-License-Identifier: Apache-2.0
"""Smooth, differentiable sampling of a dense scalp field at continuous points.

This is the mathematical core of the ``headfield`` Tesseract's derivative and is
kept here (rather than inline in ``tesseract_api.py``) so it can be unit-tested
and finite-difference-checked without any Tesseract machinery.

Given unit electrode directions ``q`` and unit scalp-lattice directions ``v``,
the interpolation weights are a temperature-controlled softmax over cosine
similarity::

    a[k, j] = kappa * q[k] . v[j]
    w[k, :] = softmax(a[k, :])
    L[k, :] = sum_j w[k, j] * lead_field[j, :]

``kappa`` sets the locality: large ``kappa`` approaches nearest-vertex lookup,
small ``kappa`` blurs over the whole scalp. Unlike nearest-vertex or
barycentric-within-triangle sampling, this is smooth everywhere on the sphere,
so the electrode coordinates have a well-defined gradient with no boundary
crossings to special-case.

All functions here are plain NumPy with hand-written reverse-mode rules. That is
deliberate: it keeps the ``headfield`` Tesseract free of any AD framework, which
is exactly the situation Tesseract exists to handle (and the situation OpenMEEG
will actually be in).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "normalize_with_norms",
    "interpolation_weights",
    "sample_lead_field",
    "forward",
    "backward",
]


def normalize_with_norms(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return unit directions and the norms that produced them.

    Args:
        u: ``[K, 3]`` unconstrained electrode design vectors.

    Returns:
        ``(q, norms)`` with ``q`` of shape ``[K, 3]`` and ``norms`` of shape
        ``[K]``.
    """
    u = np.asarray(u, dtype=np.float64)
    norms = np.linalg.norm(u, axis=1)
    if np.any(norms < 1e-9):
        raise ValueError(
            "electrode design vectors must have norm >= 1e-9; a near-zero "
            "vector has no well-defined scalp direction"
        )
    return u / norms[:, None], norms


def interpolation_weights(
    q: np.ndarray, scalp_directions: np.ndarray, kappa: float
) -> np.ndarray:
    """Softmax interpolation weights of shape ``[K, J]``, rows summing to 1."""
    logits = kappa * (q @ scalp_directions.T)
    logits = logits - logits.max(axis=1, keepdims=True)  # numerically stable
    weights = np.exp(logits)
    return weights / weights.sum(axis=1, keepdims=True)


def sample_lead_field(
    q: np.ndarray,
    scalp_directions: np.ndarray,
    lead_field: np.ndarray,
    kappa: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate the dense lead field at continuous electrode directions.

    Returns:
        ``(sampled, weights)`` with ``sampled`` of shape ``[K, S]`` and
        ``weights`` of shape ``[K, J]``.
    """
    weights = interpolation_weights(q, scalp_directions, kappa)
    return weights @ lead_field, weights


def forward(
    u: np.ndarray,
    source_activity: np.ndarray,
    scalp_directions: np.ndarray,
    lead_field: np.ndarray,
    kappa: float,
    radius: float,
) -> dict[str, np.ndarray]:
    """Full headfield forward pass.

    Args:
        u: ``[K, 3]`` unconstrained electrode design vectors.
        source_activity: ``[B, S, T]`` source amplitudes over time.
        scalp_directions: ``[J, 3]`` unit scalp lattice directions.
        lead_field: ``[J, S]`` dense lead field.
        kappa: Softmax locality parameter.
        radius: Scalp radius, used to place the realized electrodes.

    Returns:
        Dict with ``eeg`` ``[B, K, T]``, ``electrode_xyz`` ``[K, 3]``, and the
        intermediates ``q``, ``norms``, ``weights``, ``sampled`` needed by
        :func:`backward`.
    """
    q, norms = normalize_with_norms(u)
    sampled, weights = sample_lead_field(q, scalp_directions, lead_field, kappa)
    eeg = np.einsum("ks,bst->bkt", sampled, source_activity)
    return {
        "eeg": eeg,
        "electrode_xyz": radius * q,
        "q": q,
        "norms": norms,
        "weights": weights,
        "sampled": sampled,
    }


def backward(
    cache: dict[str, np.ndarray],
    source_activity: np.ndarray,
    scalp_directions: np.ndarray,
    lead_field: np.ndarray,
    kappa: float,
    radius: float,
    grad_eeg: np.ndarray | None = None,
    grad_electrode_xyz: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Analytic reverse-mode sensitivities of :func:`forward`.

    Args:
        cache: The dict returned by :func:`forward` for the same primal point.
        source_activity: ``[B, S, T]`` source amplitudes (the primal input).
        scalp_directions: ``[J, 3]`` unit scalp lattice directions.
        lead_field: ``[J, S]`` dense lead field.
        kappa: Softmax locality parameter.
        radius: Scalp radius.
        grad_eeg: ``[B, K, T]`` cotangent on ``eeg``, or ``None`` for zero.
        grad_electrode_xyz: ``[K, 3]`` cotangent on ``electrode_xyz``, or
            ``None`` for zero.

    Returns:
        Dict with ``electrode_vectors`` ``[K, 3]`` and ``source_activity``
        ``[B, S, T]`` cotangents.
    """
    q = cache["q"]
    norms = cache["norms"]
    weights = cache["weights"]
    sampled = cache["sampled"]

    grad_source = np.zeros_like(source_activity)
    # d(loss)/d(sampled lead field), shape [K, S].
    if grad_eeg is not None:
        grad_sampled = np.einsum("bkt,bst->ks", grad_eeg, source_activity)
        grad_source = np.einsum("bkt,ks->bst", grad_eeg, sampled)
    else:
        grad_sampled = np.zeros_like(sampled)

    # Through `sampled = weights @ lead_field`.
    grad_weights = grad_sampled @ lead_field.T  # [K, J]
    # Through the row-wise softmax: dL/da = w * (dL/dw - <w, dL/dw>).
    grad_logits = weights * (
        grad_weights - np.sum(weights * grad_weights, axis=1, keepdims=True)
    )
    # Through `logits = kappa * q @ v.T`.
    grad_q = kappa * (grad_logits @ scalp_directions)  # [K, 3]
    # Through `electrode_xyz = radius * q`.
    if grad_electrode_xyz is not None:
        grad_q = grad_q + radius * grad_electrode_xyz
    # Through `q = u / ||u||`: project out the radial component, scale by 1/||u||.
    radial = np.sum(grad_q * q, axis=1, keepdims=True)
    grad_u = (grad_q - radial * q) / norms[:, None]

    return {"electrode_vectors": grad_u, "source_activity": grad_source}
