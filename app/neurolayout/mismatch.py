# Copyright 2026 NeuroLocate contributors
# SPDX-License-Identifier: Apache-2.0
r"""An *independent* forward model, for generating data NeuroLocate did not make.

The single most flattering thing a source-localization paper can do is generate
its test data with the same forward operator it inverts. Everything unmodelled
then cancels exactly, the residual at the true source is machine zero, and the
reported error measures the optimizer rather than the method. That is the
**inverse crime**, and NeuroLocate's earlier K=1 numbers (median error ≈ 0 mm,
noise-free) are a textbook instance of it.

This module exists to commit no crime. It builds a forward operator that differs
from the one under test along four axes, any combination of which can be enabled:

**Solver.** MNE-Python's BEM is a *linear-collocation* boundary-element method
with isolated-skull handling; OpenMEEG's is the *symmetric* BEM of Kybic et al.
(2005). Two independent implementations of two different integral formulations,
written by different people in different languages.

**Discretization.** The generator runs on ico4 surfaces (2562 vertices each), the
solver under test on ico3 (642). Four times the elements.

**Skull conductivity.** The dominant physical uncertainty in EEG. The literature
brain-to-skull ratio spans roughly 15:1 to 80:1; the inference model is fixed at
50:1 (0.3 / 0.006 S/m), and the generator can be given something else. No real
head's skull conductivity is known, so an estimator that only works when it is
handed the right number is not an estimator.

**Electrode positions.** A template montage is not where the electrodes were. The
generator can displace each electrode tangentially along the scalp with a stated
RMS; the inference model keeps the canonical positions, exactly as an experimenter
who never digitized would.

The ladder in :data:`MISMATCH_LEVELS` is deliberately graded, so a result can say
*which* mismatch cost how much rather than reporting one lump.

What this module does **not** claim
-----------------------------------
It is not a ground-truth head model. It is a *different* head model, and the
difference between two template BEMs is much smaller than the difference between
a template BEM and a real head. Removing the inverse crime makes the numbers
honest about model error of a known kind; it does not make them a prediction of
accuracy on a person.

Cost
----
Building an ico4 three-layer BEM solution takes about 100 s and 500 MB, so
solutions are memoized per (subdivision, conductivity) and observations are
generated **offline** into an artifact by ``scripts/build_observations.py``. The
inference side never imports MNE.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "INFERENCE_SKULL_CONDUCTIVITY",
    "INFERENCE_ICO",
    "MismatchSpec",
    "MISMATCH_LEVELS",
    "IndependentForward",
]

#: Skull conductivity the *inference* model is fixed at, S/m (a 50:1 ratio
#: against 0.3 S/m brain and scalp). Every generator conductivity is stated
#: relative to this.
INFERENCE_SKULL_CONDUCTIVITY = 0.006

#: BEM subdivision the inference model uses. The generator uses a finer one.
INFERENCE_ICO = 3


@dataclass(frozen=True)
class MismatchSpec:
    """How the generating forward model differs from the inference one.

    Attributes:
        name: Short label used in result keys and figures.
        ico: BEM surface subdivision of the generator. ``None`` means "the same
            as inference", which only makes sense for :data:`MISMATCH_LEVELS`'s
            ``matched`` entry.
        skull_conductivity: Generator skull conductivity, S/m.
        electrode_error_mm: RMS tangential displacement applied to every
            electrode in the generator, millimetres.
        electrode_seed: Seed for the displacement, so the perturbed array is a
            fixed property of the spec rather than of the trial.
        description: One line for the results table.
    """

    name: str
    ico: int | None = 4
    skull_conductivity: float = INFERENCE_SKULL_CONDUCTIVITY
    electrode_error_mm: float = 0.0
    electrode_seed: int = 7
    description: str = ""

    @property
    def is_matched(self) -> bool:
        """Whether this spec asks for the inference model itself (the crime)."""
        return self.ico is None

    @property
    def skull_ratio(self) -> float:
        """Brain-to-skull conductivity ratio of the generator."""
        return 0.3 / self.skull_conductivity

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable description, written into every observation artifact."""
        return {
            "name": self.name,
            "description": self.description,
            "generator": "matched (inference model itself)" if self.is_matched else (
                f"mne-linear-collocation-bem ico{self.ico}"
            ),
            "inference": f"openmeeg-symmetric-bem ico{INFERENCE_ICO}",
            "ico": self.ico,
            "skull_conductivity": self.skull_conductivity,
            "skull_ratio": self.skull_ratio,
            "inference_skull_conductivity": INFERENCE_SKULL_CONDUCTIVITY,
            "electrode_error_mm": self.electrode_error_mm,
            "electrode_seed": self.electrode_seed,
        }


#: The graded mismatch ladder. Each rung adds one source of error, so the cost of
#: each can be read off separately instead of being attributed to the whole.
MISMATCH_LEVELS: dict[str, MismatchSpec] = {
    "matched": MismatchSpec(
        name="matched",
        ico=None,
        description="same forward model as inference — an inverse crime, kept as calibration",
    ),
    "solver": MismatchSpec(
        name="solver",
        ico=4,
        description="independent BEM implementation and formulation, 4x finer surfaces",
    ),
    "skull": MismatchSpec(
        name="skull",
        ico=4,
        skull_conductivity=0.0033,
        description="solver mismatch plus a 90:1 skull ratio against the assumed 50:1",
    ),
    "electrodes": MismatchSpec(
        name="electrodes",
        ico=4,
        electrode_error_mm=5.0,
        description="solver mismatch plus 5 mm RMS electrode displacement",
    ),
    "full": MismatchSpec(
        name="full",
        ico=4,
        skull_conductivity=0.0033,
        electrode_error_mm=5.0,
        description="solver, skull conductivity and electrode position error together",
    ),
}


def _displaced_electrodes(
    canonical_xyz: np.ndarray,
    scalp_vertices: np.ndarray,
    rms_mm: float,
    seed: int,
) -> np.ndarray:
    """Displace each electrode tangentially along the scalp with a stated RMS.

    The displacement is tangential rather than isotropic because a real
    misplacement slides an electrode across the scalp; a radial component would
    lift it off the head, which the BEM's sensor projection would simply undo. The
    local outward direction is taken radially from the scalp centroid, which is
    accurate to a couple of degrees on a head-shaped surface, and the residual
    radial error is second order in the displacement (0.14 mm for a 5 mm slide on
    a 9 cm radius).

    Args:
        canonical_xyz: ``[C, 3]`` template positions, metres, head frame.
        scalp_vertices: ``[V, 3]`` outer-skin vertices, used only for the centroid.
        rms_mm: Target RMS displacement magnitude, millimetres.
        seed: Seed for the realization.

    Returns:
        ``[C, 3]`` displaced positions, metres.
    """
    rng = np.random.default_rng(seed)
    centre = np.asarray(scalp_vertices, dtype=np.float64).mean(axis=0)
    displaced = np.array(canonical_xyz, dtype=np.float64, copy=True)
    # Two tangential degrees of freedom, so the magnitude is Rayleigh distributed
    # with scale sigma and RMS sqrt(2) * sigma.
    sigma = rms_mm * 1e-3 / np.sqrt(2.0)
    for index, point in enumerate(displaced):
        outward = point - centre
        outward /= np.linalg.norm(outward)
        step = rng.standard_normal(3)
        step -= (step @ outward) * outward
        step /= max(np.linalg.norm(step), 1e-30)
        displaced[index] = point + rng.rayleigh(sigma) * step
    return displaced


@lru_cache(maxsize=4)
def _bem_solution(ico: int, skull_conductivity: float):
    """Memoized MNE BEM solution — the expensive part (~100 s, ~500 MB at ico4)."""
    import mne

    surfaces = mne.make_bem_model(
        "fsaverage",
        ico=ico,
        conductivity=(0.3, skull_conductivity, 0.3),
        subjects_dir=_subjects_dir(),
        verbose="ERROR",
    )
    return mne.make_bem_solution(surfaces, verbose="ERROR")


@lru_cache(maxsize=1)
def _subjects_dir() -> Path:
    import mne

    return Path(mne.datasets.fetch_fsaverage(verbose="ERROR")).parent


@lru_cache(maxsize=1)
def _head_to_mri():
    """The fsaverage head↔MRI transform, the same one the OpenMEEG builder used."""
    import mne

    return mne.read_trans(_subjects_dir() / "fsaverage" / "bem" / "fsaverage-trans.fif")


class IndependentForward:
    """MNE-BEM forward operator used to *generate* observations.

    Deliberately mirrors
    :meth:`neurolayout_shared.openmeeg_model.OpenMEEGForward.gain` so the two are
    drop-in substitutes, and deliberately shares nothing else with it.

    Args:
        spec: Which mismatches to introduce. Must not be the ``matched`` spec —
            that condition is served by the inference model itself, and asking
            this class for it would silently make the crime look like a control.
        channel_names: Channel names in the frozen canonical order.
        canonical_sensor_xyz: ``[C, 3]`` template electrode positions, metres,
            head frame — what the inference model believes.
        scalp_vertices: ``[V, 3]`` outer-skin vertices, for the centroid used by
            the electrode displacement.
        sfreq: Sampling rate written into the MNE ``Info``; the forward does not
            depend on it.
    """

    def __init__(
        self,
        spec: MismatchSpec,
        channel_names: tuple[str, ...],
        canonical_sensor_xyz: np.ndarray,
        scalp_vertices: np.ndarray,
        *,
        sfreq: float = 160.0,
    ) -> None:
        """Build the MNE ``Info`` and BEM solution for ``spec``."""
        import mne

        if spec.is_matched:
            raise ValueError(
                "MismatchSpec 'matched' has no independent forward model by "
                "construction; generate that condition with the inference model"
            )
        self.spec = spec
        self.channel_names = tuple(channel_names)
        self.canonical_sensor_xyz = np.asarray(canonical_sensor_xyz, dtype=np.float64)

        self.sensor_xyz = (
            self.canonical_sensor_xyz
            if spec.electrode_error_mm <= 0.0
            else _displaced_electrodes(
                self.canonical_sensor_xyz,
                scalp_vertices,
                spec.electrode_error_mm,
                spec.electrode_seed,
            )
        )
        self.realized_electrode_rms_mm = float(
            np.sqrt(
                np.mean(np.sum((self.sensor_xyz - self.canonical_sensor_xyz) ** 2, axis=1))
            )
            * 1e3
        )

        montage = mne.channels.make_dig_montage(
            ch_pos=dict(zip(self.channel_names, self.sensor_xyz, strict=True)),
            coord_frame="head",
        )
        self._info = mne.create_info(list(self.channel_names), sfreq, "eeg")
        self._info.set_montage(montage, verbose="ERROR")
        self._trans = _head_to_mri()
        self._rotation = np.asarray(self._trans["trans"])[:3, :3]
        self._bem = _bem_solution(spec.ico, spec.skull_conductivity)
        from neurolayout_shared.source_model import average_reference_operator

        self._reference = average_reference_operator(len(self.channel_names))

    @property
    def n_channels(self) -> int:
        """``C``."""
        return len(self.channel_names)

    def gain(self, positions_m: np.ndarray) -> np.ndarray:
        """Free-orientation, average-referenced gain at arbitrary head-frame points.

        Args:
            positions_m: ``[P, 3]`` dipole positions, metres, MNE head frame.

        Returns:
            ``[C, P, 3]`` volts per A·m, with the same channel order, orientation
            convention and average reference as the OpenMEEG forward.

        Raises:
            RuntimeError: If MNE discarded any requested source, which would
                silently misalign the returned columns with the request.
        """
        import mne
        from mne.transforms import apply_trans

        positions = np.asarray(positions_m, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(f"positions must be [P, 3], got {positions.shape}")
        n_points = positions.shape[0]

        # A discrete source space at exactly the requested points. The normals are
        # required by the constructor but unused: a discrete source space yields a
        # free-orientation forward whose three columns per source are the head
        # frame's own axes.
        source_space = mne.setup_volume_source_space(
            pos={
                "rr": apply_trans(self._trans, positions),
                "nn": np.tile([0.0, 0.0, 1.0], (n_points, 1)),
            },
            verbose="ERROR",
        )
        forward = mne.make_forward_solution(
            self._info,
            trans=self._trans,
            src=source_space,
            bem=self._bem,
            eeg=True,
            meg=False,
            mindist=0.0,
            verbose="ERROR",
        )
        if forward["nsource"] != n_points:
            raise RuntimeError(
                f"MNE kept {forward['nsource']} of {n_points} requested sources; the "
                "returned gain would not line up with the request"
            )
        if list(forward["info"]["ch_names"]) != list(self.channel_names):
            raise RuntimeError("MNE reordered the channels; the forward would be scrambled")
        gain = np.asarray(forward["sol"]["data"], dtype=np.float64)
        gain = gain.reshape(self.n_channels, n_points, 3)
        return np.einsum("cd,dpj->cpj", self._reference, gain)

    def provenance(self) -> dict[str, Any]:
        """What generated the data, in enough detail to be checked."""
        import mne

        return {
            **self.spec.to_dict(),
            "mne_version": mne.__version__,
            "realized_electrode_rms_mm": self.realized_electrode_rms_mm,
            "n_channels": self.n_channels,
            "coord_frame": "mne-head",
            "units": "m",
        }
