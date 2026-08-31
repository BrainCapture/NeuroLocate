# Copyright 2026 NeuroLocate contributors
# SPDX-License-Identifier: Apache-2.0
r"""The global source-set proposal network — the component's copy.

.. note::

   This file is generated from ``app/neurolayout/hybrid/model.py`` by
   ``scripts/sync_proposal_component.py`` and must not be edited here.
   The component image carries only PyTorch, so it cannot import the
   orchestrator package; the architecture is therefore duplicated rather
   than shared, and the sync script's ``--check`` mode is what keeps the
   two honest.

The global source-set proposal network.

The estimand is a *set* of continuous source positions, and the difficulty is
global rather than local: the correlated-source objective has many deep minima and
gradient descent finds whichever one it started nearest. So the network's job is
not to be accurate — the OpenMEEG refinement downstream does that — it is to be in
the right basin.

Representation: a coarse heatmap with continuous offsets
--------------------------------------------------------
The output is a logit and a continuous offset per voxel of a coarse grid inside
the inner skull, plus a moment vector per voxel and a count head. A proposal is
read off as the top-``K`` local maxima, each corrected by its own offset:

.. math::  \hat p_v = \mathrm{centre}_v + \mathrm{pitch}\;\tanh(o_v)

Two reasons for this over direct set regression. A heatmap is *multimodal*: when
two sources 15 mm apart are genuinely not separable from 64 electrodes, it can say
so by putting one broad ridge there, where a fixed set of ``K`` regressed
coordinates has to commit. And it is permutation-free by construction — a set has
no order, and neither does a heatmap — so the training loss needs no assignment
step and cannot be destabilized by one flipping between epochs. Scoring still goes
through the minimum-total-distance assignment in :mod:`neurolayout.matching`,
because that is how every other method in this project is scored.

The grid is not a source space
------------------------------
The voxel centres are an 8 mm lattice; the offsets are continuous and the
refinement that follows is continuous. Nothing in the pipeline can return a voxel
centre as an answer, and the reported positions are never on the lattice. The
lattice is where the *argmax* lives, not where the estimate does.

Input: the spatial covariance
-----------------------------
The network sees ``C = Y Yᵀ`` — the sensor covariance of the epoch — rather than
the epoch itself. That is what a subspace method uses, and it buys two exact
invariances for free: to the sign of any source's time course, and to any
invertible mixing of the time axis. Both are gauge in this problem, and a network
that has to learn them is a network spending capacity on nothing. The
consequence is that the network cannot estimate time courses, which is correct:
the physics loss and the refinement both profile the time courses out in closed
form.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

__all__ = [
    "ProposalConfig",
    "ProposalNet",
    "features",
    "decode",
    "save_checkpoint",
    "load_checkpoint",
    "flat_parameters",
    "load_flat_parameters",
]


@dataclass(frozen=True)
class ProposalConfig:
    """Shape of the network.

    Attributes:
        n_channels: Sensor count ``C``.
        width: Token width ``d`` of the channel encoder.
        depth: Encoder blocks.
        heads: Attention heads.
        voxel_dim: Width of the learned per-voxel embedding.
        max_sources: Largest ``K`` the count head can predict.
        grid_pitch_m: Lattice pitch, metres.
        offset_scale: Offset range as a multiple of the pitch. One full pitch,
            not a half: a voxel adjacent to a true source must be able to point
            at it, so that a decode which picks the neighbour of the best voxel
            still returns the right coordinate.
        dropout: Encoder dropout.
    """

    n_channels: int = 64
    width: int = 128
    depth: int = 4
    heads: int = 4
    voxel_dim: int = 48
    max_sources: int = 4
    grid_pitch_m: float = 0.008
    offset_scale: float = 1.0
    dropout: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable record."""
        return {
            "n_channels": self.n_channels,
            "width": self.width,
            "depth": self.depth,
            "heads": self.heads,
            "voxel_dim": self.voxel_dim,
            "max_sources": self.max_sources,
            "grid_pitch_m": self.grid_pitch_m,
            "offset_scale": self.offset_scale,
            "dropout": self.dropout,
        }


def features(
    eeg: torch.Tensor, channel_mask: torch.Tensor, sensor_xyz: torch.Tensor
) -> torch.Tensor:
    """``[B, C, C + 5]`` per-channel tokens.

    Args:
        eeg: ``[B, C, T]`` sensor signals, any scale.
        channel_mask: ``[B, C]`` 1 for a retained channel.
        sensor_xyz: ``[C, 3]`` electrode positions, metres.

    Returns:
        Per-channel tokens: a row of the normalized covariance, that channel's
        own variance, its mask bit, and its position.

    The covariance is normalized by its own trace rather than by the data's RMS,
    so the features are invariant to the absolute amplitude of the sources — which
    is unknowable from one epoch and is not part of the estimand.
    """
    masked = eeg * channel_mask[:, :, None]
    covariance = masked @ masked.transpose(1, 2)  # [B, C, C]
    trace = covariance.diagonal(dim1=1, dim2=2).sum(dim=1)[:, None, None]
    # Normalized by the trace *and* rescaled by the channel count, so the mean
    # diagonal entry is 1 rather than 1/C. Without the rescaling the covariance
    # features arrive sixty times smaller than the position and mask features
    # beside them, and a linear embedding at standard initialization spends its
    # early training undoing that rather than learning anything.
    covariance = covariance * (eeg.shape[1] / torch.clamp(trace, min=1e-30))
    variance = covariance.diagonal(dim1=1, dim2=2)[:, :, None]
    positions = sensor_xyz[None].expand(eeg.shape[0], -1, -1)
    return torch.cat(
        [covariance, variance, channel_mask[:, :, None], positions * 10.0], dim=2
    )


class ProposalNet(nn.Module):
    """EEG covariance in, a source-set heatmap out.

    Args:
        config: The shape.
        voxel_centres_m: ``[V, 3]`` lattice centres inside the inner skull,
            metres, head frame. Carried by the module (and by its checkpoint)
            rather than rebuilt, because rebuilding it needs MNE and the served
            component has only PyTorch.
        sensor_xyz_m: ``[C, 3]`` electrode positions, metres, in the frozen
            channel order.
    """

    def __init__(
        self,
        config: ProposalConfig,
        voxel_centres_m: np.ndarray,
        sensor_xyz_m: np.ndarray,
    ) -> None:
        """Build the encoder and the per-voxel heads."""
        super().__init__()
        self.config = config
        centres = torch.as_tensor(np.asarray(voxel_centres_m), dtype=torch.float32)
        sensors = torch.as_tensor(np.asarray(sensor_xyz_m), dtype=torch.float32)
        if centres.ndim != 2 or centres.shape[1] != 3:
            raise ValueError(f"voxel_centres_m must be [V, 3], got {tuple(centres.shape)}")
        if sensors.shape != (config.n_channels, 3):
            raise ValueError(
                f"sensor_xyz_m must be [{config.n_channels}, 3], got {tuple(sensors.shape)}"
            )
        self.register_buffer("voxel_centres", centres)
        self.register_buffer("sensor_xyz", sensors)

        width, voxel_dim = config.width, config.voxel_dim
        # LayerNorm before the embedding: the token concatenates a covariance
        # row, a variance, a mask bit and a position, four quantities with no
        # reason to share a scale, and normalizing them is cheaper than tuning
        # four constants.
        self.token_norm = nn.LayerNorm(config.n_channels + 5)
        self.embed = nn.Linear(config.n_channels + 5, width)
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=config.heads,
            dim_feedforward=4 * width,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.depth)
        self.norm = nn.LayerNorm(width)
        self.pool = nn.Sequential(
            nn.Linear(2 * width, 2 * width), nn.GELU(), nn.Linear(2 * width, 2 * width)
        )

        self.voxel_embedding = nn.Parameter(torch.randn(len(centres), voxel_dim) * 0.05)
        self.voxel_bias = nn.Parameter(torch.zeros(len(centres)))
        self.to_logit = nn.Linear(2 * width, voxel_dim)
        self.to_offset = nn.Linear(2 * width, voxel_dim)
        self.to_moment = nn.Linear(2 * width, voxel_dim)
        self.offset_head = nn.Linear(voxel_dim, 3)
        self.moment_head = nn.Linear(voxel_dim, 3)
        self.count_head = nn.Linear(2 * width, config.max_sources)
        # Start at "no source anywhere": a heatmap initialized near p = 0.5 would
        # spend its first thousand steps pushing 3000 voxels down.
        nn.init.constant_(self.voxel_bias, -4.0)
        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)

    @property
    def n_voxels(self) -> int:
        """``V``."""
        return int(self.voxel_centres.shape[0])

    def encode(self, eeg: torch.Tensor, channel_mask: torch.Tensor) -> torch.Tensor:
        """``[B, 2d]`` the pooled epoch representation."""
        tokens = self.embed(
            self.token_norm(features(eeg, channel_mask, self.sensor_xyz))
        )
        hidden = self.norm(
            self.encoder(tokens, src_key_padding_mask=(channel_mask <= 0.0))
        )
        weights = channel_mask[:, :, None]
        mean = (hidden * weights).sum(dim=1) / torch.clamp(weights.sum(dim=1), min=1.0)
        peak = hidden.masked_fill(weights <= 0.0, -1e9).max(dim=1).values
        return self.pool(torch.cat([mean, peak], dim=1))

    def forward(
        self, eeg: torch.Tensor, channel_mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Run the network.

        Args:
            eeg: ``[B, C, T]`` sensor signals, volts (any scale).
            channel_mask: ``[B, C]`` 1 for a retained channel.

        Returns:
            ``logits`` ``[B, V]``, ``positions_m`` ``[B, V, 3]`` (the centre plus
            its offset), ``moments`` ``[B, V, 3]`` (unit-norm directions), and
            ``count_logits`` ``[B, max_sources]``.
        """
        hidden = self.encode(eeg, channel_mask)
        embedding = self.voxel_embedding  # [V, D]

        logits = embedding @ self.to_logit(hidden).transpose(0, 1)  # [V, B]
        logits = logits.transpose(0, 1) + self.voxel_bias[None, :]

        gated = embedding[None] * self.to_offset(hidden)[:, None, :]  # [B, V, D]
        reach = self.config.offset_scale * self.config.grid_pitch_m
        offsets = reach * torch.tanh(self.offset_head(gated))
        positions = self.voxel_centres[None] + offsets

        moment_gate = embedding[None] * self.to_moment(hidden)[:, None, :]
        moments = self.moment_head(moment_gate)
        moments = moments / torch.clamp(
            moments.norm(dim=2, keepdim=True), min=1e-6
        )
        return {
            "logits": logits,
            "positions_m": positions,
            "moments": moments,
            "count_logits": self.count_head(hidden),
        }

    def n_parameters(self) -> int:
        """Trainable parameter count."""
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))


def decode(
    outputs: dict[str, torch.Tensor],
    n_sources: int | None = None,
    *,
    nms_radius_m: float = 0.010,
    max_sources: int = 4,
) -> dict[str, torch.Tensor]:
    """Read a source set out of the heatmap.

    Greedy non-maximum suppression: take the highest logit, suppress every voxel
    within ``nms_radius_m`` of it, repeat. The suppression radius is what stops a
    single broad blob from being read as ``K`` sources; it is set to a little over
    one grid pitch, so two sources the network genuinely separated stay two.

    Selection is a hard ``argmax`` and carries no gradient, which is the honest
    description: the *choice* of voxel is discrete. The returned coordinates do
    carry gradient, through each selected voxel's own offset and moment head, and
    that is what the physics term differentiates.

    Args:
        outputs: What :meth:`ProposalNet.forward` returned.
        n_sources: ``K`` to return. ``None`` uses the count head's argmax.
        nms_radius_m: Suppression radius, metres.
        max_sources: Cap when ``n_sources`` is ``None``.

    Returns:
        ``positions_m`` ``[B, K, 3]``, ``moments`` ``[B, K, 3]``,
        ``scores`` ``[B, K]`` (the selected logits), ``indices`` ``[B, K]``, and
        ``n_predicted`` ``[B]``.
    """
    logits = outputs["logits"]
    positions = outputs["positions_m"]
    batch, n_voxels = logits.shape
    predicted = outputs["count_logits"].argmax(dim=1) + 1
    k = int(predicted.max().item()) if n_sources is None else int(n_sources)
    k = max(1, min(k, max_sources, n_voxels))

    centres = outputs["positions_m"].detach()
    chosen = torch.zeros((batch, k), dtype=torch.long, device=logits.device)
    for sample in range(batch):
        available = logits[sample].detach().clone()
        for slot in range(k):
            index = int(torch.argmax(available).item())
            chosen[sample, slot] = index
            distance = (centres[sample] - centres[sample, index]).norm(dim=1)
            available = available.masked_fill(distance <= nms_radius_m, -float("inf"))
    gather = chosen[:, :, None].expand(-1, -1, 3)
    return {
        "positions_m": torch.gather(positions, 1, gather),
        "moments": torch.gather(outputs["moments"], 1, gather),
        "scores": torch.gather(logits, 1, chosen),
        "indices": chosen,
        "n_predicted": predicted,
    }


#: Buffers that are coordinates rather than parameters, and stay float64.
_COORDINATE_BUFFERS = ("voxel_centres", "sensor_xyz")


def save_checkpoint(
    path: str | Path, model: ProposalNet, metadata: dict[str, Any] | None = None
) -> Path:
    """Write weights, shape, lattice and provenance to one file.

    Everything needed to rebuild the module is in here, including the voxel
    centres and the sensor array, so the served component never has to reach for
    a head model or for MNE.

    Parameters are stored as float32 and cast back to float64 on load. Training
    runs in float64 to match the Tesseract boundary, but a checkpoint is halved by
    this and the network's output is a *proposal* that is refined afterwards
    through the real solver — 7 significant figures is far more than the position
    of a source in a head is defined to. The lattice stays float64, because it is
    a coordinate.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": model.config.to_dict(),
            "voxel_centres_m": model.voxel_centres.detach().cpu().numpy(),
            "sensor_xyz_m": model.sensor_xyz.detach().cpu().numpy(),
            "state_dict": {
                key: (
                    value.detach().cpu().float()
                    if value.is_floating_point() and key not in _COORDINATE_BUFFERS
                    else value.detach().cpu()
                )
                for key, value in model.state_dict().items()
            },
            "metadata": json.dumps(metadata or {}, sort_keys=True, default=str),
        },
        path,
    )
    return path


def load_checkpoint(
    path: str | Path, *, map_location: Any = "cpu"
) -> tuple[ProposalNet, dict[str, Any]]:
    """Rebuild a :class:`ProposalNet` from a checkpoint.

    Returns:
        ``(model, metadata)`` with the model in eval mode.
    """
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model = ProposalNet(
        ProposalConfig(**payload["config"]),
        payload["voxel_centres_m"],
        payload["sensor_xyz_m"],
    )
    model.load_state_dict(
        {
            key: (
                value.double()
                if value.is_floating_point()
                else value
            )
            for key, value in payload["state_dict"].items()
        }
    )
    model.double().eval()
    return model, json.loads(payload.get("metadata") or "{}")


def flat_parameters(model: ProposalNet) -> torch.Tensor:
    """Every trainable parameter as one 1-D tensor, in a fixed order.

    This is what crosses the Tesseract boundary. The weights are a *differentiable
    input* to the ``proposal`` component rather than state hidden inside it, which
    is what lets ``jax.grad`` of a loss defined downstream of OpenMEEG return
    ``dL/dweights`` and train the network through the solver.
    """
    return torch.cat([p.detach().reshape(-1) for p in _ordered_parameters(model)])


def load_flat_parameters(model: ProposalNet, flat: torch.Tensor) -> None:
    """Write a flat parameter vector back into a module, in place.

    The length is checked *before* anything is written. A vector of the wrong
    length is a wire-format mismatch, and half-loading one would leave the module
    in a state no caller expects while raising an error that names a reshape.
    """
    expected = sum(p.numel() for p in _ordered_parameters(model))
    if flat.numel() != expected:
        raise ValueError(
            f"parameter vector has {flat.numel()} entries but the model needs {expected}"
        )
    offset = 0
    for parameter in _ordered_parameters(model):
        size = parameter.numel()
        parameter.data.copy_(flat[offset : offset + size].reshape(parameter.shape))
        offset += size


def _ordered_parameters(model: ProposalNet) -> list[torch.nn.Parameter]:
    """Trainable parameters in a stable order.

    ``named_parameters`` is insertion-ordered and therefore stable for a fixed
    module definition, but sorting by name makes the flat layout independent of
    the order the submodules happen to be constructed in — which matters, because
    the layout is a wire format between the component and everything upstream.
    """
    return [value for _, value in sorted(model.named_parameters()) if value.requires_grad]
