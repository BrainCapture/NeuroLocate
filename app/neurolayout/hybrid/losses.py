# Copyright 2026 NeuroLocate contributors
# SPDX-License-Identifier: Apache-2.0
r"""The supervised source-set loss, and the metrics that judge it.

The target is a **set**, so the loss must not depend on an order. A heatmap has
none: the target field is

.. math::  \tau_v = \max_k \exp\!\left(-\lVert c_v - p_k \rVert^2 / 2\sigma^2\right)

over the ``K`` true sources, and swapping two sources leaves it identical. That is
why the training loss needs no assignment step — unlike direct set regression,
where a Hungarian match has to be re-solved every step and can flip between epochs
on nearly-tied configurations.

Scoring is a different matter. Every other method in this project is scored by the
minimum-total-distance assignment in :mod:`neurolayout.matching`, and the proposal
is scored the same way, so the numbers are comparable. The loss is
permutation-free; the metric is permutation-*invariant*; they are not the same
statement and both are needed.

Three terms
-----------
**Heatmap.** A penalty-reduced focal loss (Law & Deng 2018, as used by CenterNet).
The plain cross-entropy is hopeless here: with ``K <= 4`` sources over ~3000
voxels the target is 99.9% background, and a network that predicts zero everywhere
scores well. The focal form down-weights easy negatives, and the
``(1 - \tau_v)^\beta`` factor forgives a voxel *next to* a true source, which is
not really a mistake.

**Offset.** Smooth-L1 in millimetres, at every voxel within one pitch of a true
source — not only the nearest one. A decode that picks the neighbour of the best
voxel must still return the right coordinate, and that only holds if the
neighbour's offset was trained to point at the source.

**Moment.** ``1 - |cos|``, because a dipole is defined up to sign: ``(m, a(t))``
and ``(-m, -a(t))`` produce identical EEG, so a signed loss would be asking the
network to guess a gauge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F  # noqa: N812

__all__ = [
    "LossWeights",
    "heatmap_targets",
    "proposal_loss",
    "matched_error_mm",
]


@dataclass(frozen=True)
class LossWeights:
    """Relative weights of the supervised terms.

    Recorded in the checkpoint and never tuned against the frozen benchmark. The
    scales are chosen so each term is O(1) at initialization: the focal loss is
    normalized per source, the offset loss is in millimetres divided by the pitch,
    and the moment loss is in ``[0, 1]``.

    Attributes:
        heatmap: Weight on the focal term.
        offset: Weight on the position offsets.
        moment: Weight on the dipole directions.
        count: Weight on the source-count head.
        sigma_mm: Width of the Gaussian target, millimetres.
        focal_alpha: Exponent on the predicted probability.
        focal_beta: Exponent on ``1 - target``, which is what forgives a
            near-miss voxel.
    """

    heatmap: float = 1.0
    offset: float = 1.0
    moment: float = 0.2
    count: float = 0.1
    sigma_mm: float = 5.0
    focal_alpha: float = 2.0
    focal_beta: float = 4.0

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable record."""
        return {
            "heatmap": self.heatmap,
            "offset": self.offset,
            "moment": self.moment,
            "count": self.count,
            "sigma_mm": self.sigma_mm,
            "focal_alpha": self.focal_alpha,
            "focal_beta": self.focal_beta,
        }


def heatmap_targets(
    centres_m: torch.Tensor,
    positions_m: torch.Tensor,
    source_mask: torch.Tensor,
    sigma_mm: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The Gaussian target field and the distance to the nearest true source.

    Args:
        centres_m: ``[V, 3]`` lattice centres.
        positions_m: ``[B, K, 3]`` true source positions, padded.
        source_mask: ``[B, K]`` 1 for a real source.
        sigma_mm: Target width.

    Returns:
        ``(target [B, V], distance_mm [B, V, K])``. The distance to a padded slot
        is ``inf``, so it can never be the nearest.
    """
    delta = centres_m[None, :, None, :] - positions_m[:, None, :, :]  # [B, V, K, 3]
    distance = delta.norm(dim=3) * 1e3
    distance = distance.masked_fill(source_mask[:, None, :] <= 0.0, float("inf"))
    target = torch.exp(-0.5 * (distance / sigma_mm) ** 2).max(dim=2).values
    return target, distance


def proposal_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    centres_m: torch.Tensor,
    weights: LossWeights,
    *,
    pitch_m: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """The supervised set loss and its parts.

    Args:
        outputs: What :meth:`~neurolayout.hybrid.model.ProposalNet.forward` gave.
        batch: ``positions_m`` ``[B, K, 3]``, ``moments_nam`` ``[B, K, 3]``,
            ``source_mask`` ``[B, K]``, ``n_sources`` ``[B]``.
        centres_m: ``[V, 3]`` lattice centres.
        weights: Term weights.
        pitch_m: Lattice pitch, used to scale the offset loss and to choose which
            voxels are supervised for offset and moment.

    Returns:
        ``(total, parts)`` with ``parts`` a dict of floats for logging.
    """
    logits = outputs["logits"]
    target, distance = heatmap_targets(
        centres_m, batch["positions_m"], batch["source_mask"], weights.sigma_mm
    )
    n_sources = torch.clamp(batch["source_mask"].sum(dim=1), min=1.0)

    probability = torch.sigmoid(logits).clamp(1e-6, 1.0 - 1e-6)
    # "Positive" is the voxel nearest each true source, exactly one per source.
    nearest = distance.argmin(dim=1)  # [B, K]
    positive = torch.zeros_like(logits, dtype=torch.bool)
    valid = batch["source_mask"] > 0.0
    rows = torch.arange(logits.shape[0], device=logits.device)[:, None].expand_as(nearest)
    positive[rows[valid], nearest[valid]] = True

    focal_positive = -((1.0 - probability) ** weights.focal_alpha) * torch.log(probability)
    focal_negative = (
        -((1.0 - target) ** weights.focal_beta)
        * (probability**weights.focal_alpha)
        * torch.log(1.0 - probability)
    )
    focal = torch.where(positive, focal_positive, focal_negative)
    heatmap_loss = (focal.sum(dim=1) / n_sources).mean()

    # Offset and moment are supervised wherever a voxel is close enough to a true
    # source that the heatmap read-out could plausibly select it.
    within = (distance <= pitch_m * 1e3) & (batch["source_mask"][:, None, :] > 0.0)
    assigned = distance.argmin(dim=2)  # [B, V] which source each voxel serves
    supervised = within.any(dim=2)
    if supervised.any():
        gather = assigned[:, :, None].expand(-1, -1, 3)  # [B, V, 3]
        wanted_position = torch.gather(batch["positions_m"], 1, gather)
        wanted_moment = torch.gather(batch["moments_nam"], 1, gather)
        predicted = outputs["positions_m"][supervised]
        offset_loss = F.smooth_l1_loss(
            predicted * 1e3,
            wanted_position[supervised] * 1e3,
            beta=1.0,
        )
        direction = wanted_moment[supervised]
        direction = direction / torch.clamp(direction.norm(dim=1, keepdim=True), min=1e-30)
        cosine = (outputs["moments"][supervised] * direction).sum(dim=1)
        moment_loss = (1.0 - cosine.abs()).mean()
    else:  # no voxel within a pitch of any source: possible only for a degenerate batch
        offset_loss = logits.sum() * 0.0
        moment_loss = logits.sum() * 0.0

    count_loss = F.cross_entropy(outputs["count_logits"], batch["n_sources"] - 1)

    total = (
        weights.heatmap * heatmap_loss
        + weights.offset * offset_loss
        + weights.moment * moment_loss
        + weights.count * count_loss
    )
    parts = {
        "loss": float(total.detach()),
        "heatmap": float(heatmap_loss.detach()),
        "offset_mm": float(offset_loss.detach()),
        "moment": float(moment_loss.detach()),
        "count": float(count_loss.detach()),
        "supervised_voxels": float(supervised.float().sum(dim=1).mean().detach()),
    }
    return total, parts


def matched_error_mm(
    predicted_m: np.ndarray, truth_m: np.ndarray, source_mask: np.ndarray
) -> list[np.ndarray]:
    """Per-source localization error under the optimal assignment, in millimetres.

    Delegates to :func:`neurolayout.matching.match_sources`, so the proposal is
    scored by the same rule as every classical baseline and as the gradient-only
    estimator: a correct answer found in a different order is not a failure.

    Args:
        predicted_m: ``[B, K, 3]`` proposals.
        truth_m: ``[B, K, 3]`` true positions, padded.
        source_mask: ``[B, K]`` 1 for a real source.

    Returns:
        One array of errors per batch entry, in estimate order.
    """
    from neurolayout.matching import match_sources

    errors = []
    for index in range(len(predicted_m)):
        keep = source_mask[index] > 0.0
        errors.append(
            match_sources(predicted_m[index][keep], truth_m[index][keep]).errors_mm
        )
    return errors
