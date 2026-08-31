# Copyright 2026 NeuroLocate contributors
# SPDX-License-Identifier: Apache-2.0
r"""The composed objective: proposal Tesseract -> OpenMEEG Tesseract -> JAX.

This module is where the project's claim is cashed for the hybrid estimator. One
``jax.grad`` call runs

.. code-block:: text

    loss
      -> headfield  VJP : central differences through OpenMEEG's C++ symmetric BEM
                          (position) and exact hand-written algebra (moment)
      -> proposal   VJP : torch.autograd through the network
      -> network parameters

Three derivative technologies, two process boundaries, no autodiff framework in
common. The optimizer imports neither PyTorch nor OpenMEEG; neither component
imports JAX.

The objective: a projector residual
-----------------------------------
Each source contributes one topography column ``c_k = G(p_k) m_k``, and its time
course is a free ``T``-vector. Since the map is linear in those time courses they
can be **profiled out** in closed form, leaving a function of the positions and
moments alone:

.. math::

    L(p, m) \;=\;
    \frac{\lVert (I - P_{C})\, Y \rVert_F^2}{\lVert Y \rVert_F^2},
    \qquad C = [\,c_1 \cdots c_K\,]

with ``P_C`` the orthogonal projector onto the columns' span. This is the same
estimator as the ``free`` temporal model — the fit is identical at the optimum,
because the profiled variables are exactly the ones being minimized over — with
``K x T`` fewer parameters and a far better-conditioned landscape.

It is not a shortcut invented here. The profiled objective was checked against
the full optimizer, which carries the waveforms explicitly, before it was used
in place of it: 0.26 mm median disagreement, and a data fit at least as good
every time.

Where the columns come from
---------------------------
``headfield`` returns ``eeg``, a sum over sources, not the individual columns. The
columns are recovered exactly by asking it for ``T = K`` samples with the moment
of source ``k`` placed at time ``k`` and zero elsewhere: the sum then has one term
per sample, and ``eeg[:, :, k]`` is ``c_k``. One batched solver call for the whole
batch, no extra physics, and the derivative bookkeeping is the component's own.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from tesseract_jax import apply_tesseract

__all__ = [
    "RIDGE",
    "columns",
    "source_gains",
    "best_moments",
    "projector_residual",
    "proposal_outputs",
    "make_physics_loss",
    "containment_penalty",
    "physics_value_and_grad",
]

#: Ridge on ``CᵀC``, relative to its own trace, before the profiled solve.
#:
#: Two proposals can land close enough that their topography columns are
#: numerically collinear, and then ``CᵀC`` is singular and the residual and its
#: derivative are both meaningless. A relative ridge fixes that without changing
#: the objective anywhere it was already well posed: at 1e-9 of the trace it moves
#: a well-separated residual by less than the finite-difference error in the
#: position derivative itself.
RIDGE = 1e-9


def _static_inputs(config: Any) -> dict[str, Any]:
    """The headfield inputs that are constant under tracing, in batch mode.

    ``static_inputs`` carries a placeholder for every array field, including the
    batched positions, so the batched placeholder is dropped here: it would
    otherwise overwrite the traced argument when the two dicts are merged, and the
    component would silently compute a one-source forward on a zero position.
    That happened once; the shape check inside ``forward_batched`` is what caught
    it, which is why that check raises rather than broadcasting.
    """
    static = dict(config.static_inputs())
    static.pop("source_positions_batch", None)
    return {**static, "mode": "localize_batch", "source_positions": np.zeros((1, 3))}


def columns(headfield: Any, positions_m: jnp.ndarray, moments: jnp.ndarray, config: Any):
    """``[B, C, K]`` topography columns, through the OpenMEEG component.

    Args:
        headfield: An opened ``headfield`` Tesseract.
        positions_m: ``[B, K, 3]`` source positions, metres.
        moments: ``[B, K, 3]`` dipole moments, A·m (scale is irrelevant to the
            projector, but not to the conditioning, so keep it physical).
        config: A :class:`~neurolayout.localize.LocalizeConfig`.

    Returns:
        ``[B, C, K]`` with entry ``(b, :, k)`` the topography of source ``k``.
    """
    n_sources = positions_m.shape[1]
    # q[b, k, :, t] = m[b, k, :] * delta(k, t): the sum over sources then has one
    # surviving term per output sample, so eeg[b, :, k] is exactly c_k.
    selector = jnp.eye(n_sources, dtype=positions_m.dtype)
    timecourses = moments[:, :, :, None] * selector[None, :, None, :]
    outputs = apply_tesseract(
        headfield,
        {
            "source_positions_batch": positions_m,
            "source_timecourses": timecourses,
            **_static_inputs(config),
        },
    )
    return outputs["eeg"]


def source_gains(headfield: Any, positions_m: jnp.ndarray, config: Any) -> jnp.ndarray:
    """``[B, C, K, 3]`` free-orientation gain at each proposed position.

    The same construction :func:`columns` uses, one step further: with ``T = 3K``
    samples and a unit moment along axis ``j`` of source ``k`` placed at sample
    ``3k + j``, the component's output at that sample is the gain column itself.
    One batched solver call for the whole batch.
    """
    batch, n_sources = positions_m.shape[0], positions_m.shape[1]
    selector = jnp.eye(3 * n_sources, dtype=positions_m.dtype).reshape(
        n_sources, 3, 3 * n_sources
    )
    timecourses = jnp.broadcast_to(
        selector[None], (batch, n_sources, 3, 3 * n_sources)
    )
    outputs = apply_tesseract(
        headfield,
        {
            "source_positions_batch": positions_m,
            "source_timecourses": timecourses,
            **_static_inputs(config),
        },
    )
    return outputs["eeg"].reshape(batch, -1, n_sources, 3)


def best_moments(
    gains: jnp.ndarray, observed: jnp.ndarray, *, rank: int | None = None
) -> jnp.ndarray:
    r"""``[B, K, 3]`` unit moments that best align each column with the data subspace.

    For a source at a fixed position the topography column is ``c = G m``, and the
    orientation that explains the most of the measurement is the one maximizing

    .. math::  rac{m^	op G^	op P_S G\, m}{m^	op G^	op G\, m}

    with ``P_S`` the projector onto the data's leading ``rank`` left singular
    vectors. That is a 3x3 generalized eigenproblem per source, solved in closed
    form — the same orientation step a scanning method takes, and exactly what
    RAP-MUSIC does at each recursion.

    This exists so the gradient-only estimator is not handicapped. Starting the
    refinement from a *random* dipole orientation would make it a weaker baseline
    than the frozen benchmark's own, which warm-starts its moment by least
    squares, and a result measured against a weakened control is not a result.
    Every refinement method in the matrix gets this, so it changes no comparison.
    """
    batch, _, n_sources, _ = gains.shape
    rank = n_sources if rank is None else rank
    left, _, _ = jnp.linalg.svd(observed, full_matrices=False)
    basis = left[:, :, :rank]  # [B, C, r]
    projected = jnp.einsum("bcr,bckj->brkj", basis, gains)  # [B, r, K, 3]
    signal = jnp.einsum("brkj,brkl->bkjl", projected, projected)
    total = jnp.einsum("bckj,bckl->bkjl", gains, gains)
    # Whiten by the total energy, then take the leading eigenvector: the
    # generalized problem A m = lambda B m becomes symmetric once B is factored,
    # and B is 3x3 and positive definite for any position inside the head.
    scale = jnp.trace(total, axis1=2, axis2=3)[:, :, None, None] / 3.0
    factor = jnp.linalg.cholesky(total + 1e-12 * scale * jnp.eye(3))
    inverse = jnp.linalg.inv(factor)
    whitened = jnp.einsum("bkij,bkjl,bkml->bkim", inverse, signal, inverse)
    _, vectors = jnp.linalg.eigh(whitened)
    moments = jnp.einsum("bkji,bkj->bki", inverse, vectors[:, :, :, -1])
    return moments / jnp.maximum(
        jnp.linalg.norm(moments, axis=2, keepdims=True), 1e-30
    )


def projector_residual(
    column_matrix: jnp.ndarray, observed: jnp.ndarray, *, ridge: float = RIDGE
) -> jnp.ndarray:
    """``[B]`` relative residual after profiling the time courses out.

    Args:
        column_matrix: ``[B, C, K]`` topography columns.
        observed: ``[B, C, T]`` measured sensor signals.
        ridge: Relative ridge on ``CᵀC``.

    Returns:
        ``[B]`` in ``[0, 1]``: 1.0 means the columns explain nothing, 0.0 a
        perfect fit.
    """
    gram = jnp.einsum("bck,bcl->bkl", column_matrix, column_matrix)
    scale = jnp.trace(gram, axis1=1, axis2=2)[:, None, None] / column_matrix.shape[2]
    regularized = gram + ridge * scale * jnp.eye(column_matrix.shape[2])
    projection = jnp.linalg.solve(
        regularized, jnp.einsum("bck,bct->bkt", column_matrix, observed)
    )
    residual = observed - jnp.einsum("bck,bkt->bct", column_matrix, projection)
    energy = jnp.sum(observed**2, axis=(1, 2)) + 1e-300
    return jnp.sum(residual**2, axis=(1, 2)) / energy


def proposal_outputs(
    proposal: Any,
    weights: jnp.ndarray,
    eeg: jnp.ndarray,
    channel_mask: jnp.ndarray,
    *,
    n_sources: int,
    nms_radius_m: float = 0.010,
    checkpoint: str | None = None,
) -> dict[str, jnp.ndarray]:
    """Run the proposal Tesseract with ``weights`` as a differentiable input."""
    return apply_tesseract(
        proposal,
        {
            "eeg": eeg,
            "channel_mask": channel_mask,
            "weights": weights,
            "checkpoint": checkpoint,
            "n_sources": int(n_sources),
            "nms_radius_m": float(nms_radius_m),
        },
    )


def containment_penalty(positions_m: jnp.ndarray, containment: Any) -> jnp.ndarray:
    """``[B]`` smooth hinge on a proposal leaving the brain compartment.

    The lattice keeps 4 mm of clearance from the inner skull and the offsets reach
    one pitch, so a proposal *can* be placed outside it — where OpenMEEG's
    ``Brain`` domain assumption stops holding and the assembled source term is not
    a physical answer. The penalty pushes the network back inside instead of
    clamping it, which would put a non-differentiable fold in the middle of the
    training gradient. How often it engages is measured, not assumed.
    """
    centre = jnp.asarray(containment.centre_cm) * 1e-2
    axes = jnp.asarray(containment.semi_axes_cm) * 1e-2
    scaled = (positions_m - centre) / axes
    radius = jnp.sqrt(jnp.sum(scaled**2, axis=-1) + 1e-12)
    return jnp.sum(jnp.maximum(radius - 1.0, 0.0) ** 2, axis=-1)


def physics_terms(
    headfield: Any,
    proposal: Any,
    weights: jnp.ndarray,
    eeg: jnp.ndarray,
    channel_mask: jnp.ndarray,
    config: Any,
    containment: Any,
    *,
    n_sources: int,
    nms_radius_m: float = 0.010,
    checkpoint: str | None = None,
) -> dict[str, np.ndarray]:
    """The composed loss's terms separately, for reporting rather than for descent.

    The one that has to be watched is ``outside_fraction``. The lattice keeps 4 mm
    of clearance from the inner skull and the offsets reach a full pitch, so a
    proposal *can* be placed where OpenMEEG's ``Brain`` domain assumption does not
    hold. The penalty pushes it back; this is how often it had to.
    """
    outputs = proposal_outputs(
        proposal,
        weights,
        eeg,
        channel_mask,
        n_sources=n_sources,
        nms_radius_m=nms_radius_m,
        checkpoint=checkpoint,
    )
    positions = outputs["positions_m"]
    built = columns(headfield, positions, outputs["moments"], config)
    penalty = containment_penalty(positions, containment)
    return {
        "data": np.asarray(projector_residual(built, eeg)),
        "containment": np.asarray(penalty),
        "outside_fraction": float(np.mean(np.asarray(penalty) > 0.0)),
    }


def make_physics_loss(
    headfield: Any,
    proposal: Any,
    config: Any,
    containment: Any,
    *,
    n_sources: int,
    nms_radius_m: float = 0.010,
    checkpoint: str | None = None,
    containment_weight: float = 10.0,
):
    r"""Close the sensor-space term over everything but the network's parameters.

    .. math::

        L_{\text{physics}}(\theta) = \frac{1}{B}\sum_b
            \frac{\lVert (I - P_{C(\theta, y_b)})\, y_b \rVert^2}{\lVert y_b \rVert^2}
            + \lambda_c \,\mathrm{containment}

    Returns:
        ``loss(weights, eeg, channel_mask) -> scalar``, ready for ``jax.grad``
        with respect to its first argument.
    """

    def loss(
        weights: jnp.ndarray, eeg: jnp.ndarray, channel_mask: jnp.ndarray
    ) -> jnp.ndarray:
        outputs = proposal_outputs(
            proposal,
            weights,
            eeg,
            channel_mask,
            n_sources=n_sources,
            nms_radius_m=nms_radius_m,
            checkpoint=checkpoint,
        )
        column_matrix = columns(
            headfield, outputs["positions_m"], outputs["moments"], config
        )
        data = projector_residual(column_matrix, eeg)
        penalty = containment_penalty(outputs["positions_m"], containment)
        return jnp.mean(data + containment_weight * penalty)

    return loss


def physics_value_and_grad(loss_fn: Any):
    """``jax.value_and_grad`` of the composed loss with respect to the weights.

    A one-line wrapper that exists so the call site reads as what it is: the
    single gradient that crosses both component boundaries.
    """
    return jax.value_and_grad(loss_fn, argnums=0)
