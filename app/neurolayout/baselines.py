# Copyright 2026 NeuroLocate contributors
# SPDX-License-Identifier: Apache-2.0
r"""Classical inverse baselines, on identical physics.

Three established estimators, run on the *same* observations and the *same*
forward operator as NeuroLocate, so the comparison isolates the estimator rather
than measuring the difference between two head models:

``scan`` — discrete dipole scan / orthogonal matching pursuit
    Evaluate the OpenMEEG gain at all 20 484 cortical candidates, fit the moment in
    closed form at each, keep the best; for ``K > 1``, greedily add the candidate
    that most reduces the residual with all moments refit jointly. This is the
    oldest sparse method there is and the fairest possible control: identical
    physics, identical sparsity assumption, no gradient. Its resolution floor is
    half the source-space spacing (2.1 mm median), by construction.

``irmxne`` — iteratively reweighted mixed-norm estimate
    MNE-Python's :func:`mne.inverse_sparse.mixed_norm`, an :math:`\ell_{2,1}`
    sparse estimator with reweighting (Gramfort et al. 2012, 2013). The modern
    reference method for sparse multi-source EEG/MEG. Run in MNE's standard
    cortically-constrained configuration; see :data:`DEFAULT_LOOSE` and
    :data:`DEFAULT_MXNE_ALPHA` for how those settings were fixed, and why the
    orientation constraint is a favour to the baseline rather than a handicap.

``dspm`` — dynamic statistical parametric mapping
    :func:`mne.minimum_norm.apply_inverse`, a distributed :math:`\ell_2` estimator.
    Included because it is what most EEG papers actually use, and because its
    failure mode — a smooth blob whose peak is biased toward the sensors — is worth
    showing next to a sparse method's.

All three are given NeuroLocate's forward operator by surgery: MNE builds the
``Forward`` bookkeeping (source space, channel info, orientation handling) and its
gain matrix is then replaced by OpenMEEG's, verified to be on the same source
space to 0 mm. Without that, a baseline would be running MNE's own BEM and every
difference in the results would be confounded by the solver.

What the comparison is for
--------------------------
Not for NeuroLocate to win. The scan baseline *should* be very hard to beat on
matched physics, and dSPM *should* lose on a sparse simulation — neither would be
news. The informative axes are (a) sub-grid resolution, where a continuous
estimator can go and a dictionary method cannot, (b) behaviour under model
mismatch, and (c) runtime, where a 20 000-column dictionary and an iterative
solver have very different costs.

Handicaps, stated
-----------------
* Both classical estimators are given a diagonal noise covariance at the true
  noise level. NeuroLocate's plain least-squares loss makes the same white-noise
  assumption, so under correlated noise both are handicapped equally.
* ``irmxne`` and ``dspm`` estimate an unknown number of sources; they are scored on
  their ``K`` strongest, and the number they actually found is reported alongside,
  because "found 1 of 4" is a failure the localization error would hide.
* The scan and ``irmxne`` are restricted to the ico5 candidate set. That is not a
  handicap imposed here — it is what a dictionary method is.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "BASELINE_NAMES",
    "PEAK_EXCLUSION_MM",
    "DEFAULT_LOOSE",
    "DEFAULT_DEPTH",
    "DEFAULT_MXNE_ALPHA",
    "BaselineResult",
    "DipoleDictionary",
    "MneInverse",
    "peak_locations",
]

#: Baselines this module provides, cheapest first.
BASELINE_NAMES: tuple[str, ...] = ("scan", "irmxne", "dspm")

#: Radius excluded around an already-taken peak when reading ``K`` locations out of
#: a distributed or multi-source estimate, in millimetres. 15 mm is roughly the
#: width of a dSPM point spread at 64 channels, so two peaks closer than this are
#: one blob and counting them separately would flatter the method.
PEAK_EXCLUSION_MM = 15.0

#: Orientation constraint for both MNE estimators: ``0.0`` is the cortically
#: constrained (fixed-normal) model, MNE's documented setting for a surface source
#: space and the standard configuration for sparse inverse solvers on one.
#:
#: This was *measured*, not assumed. On a noise-free single-source trial with the
#: correct forward operator, irMxNE localizes to 1.2 mm at ``loose=0`` and to
#: 28–36 mm at ``loose=1``: free orientation triples the dictionary and destroys the
#: identifiability the l21 penalty depends on. Note that the constraint also
#: *favours* the baselines here, since every ground-truth source in this benchmark
#: is oriented along an interpolated cortical normal — which is the direction a
#: fairness argument should err in.
DEFAULT_LOOSE = 0.0

#: Depth weighting exponent, MNE's documented default.
DEFAULT_DEPTH = 0.8

#: Regularization for irMxNE, as a percentage of the maximum (MNE's convention).
#:
#: ``"sure"`` selects it automatically by Stein's unbiased risk estimate, which is
#: the setting that removes "you tuned the regularizer against the baseline" as an
#: objection. It costs 0.3–15 s per trial at ``loose=0`` (it was minutes at
#: ``loose=1``, which is why an earlier draft avoided it). A fixed ``alpha=30``
#: gives comparable accuracy an order of magnitude faster and can be passed
#: explicitly; 10, 30 and 60 were all measured on a known single source and 10–30
#: recover it exactly.
DEFAULT_MXNE_ALPHA: float | str = "sure"


@dataclass
class BaselineResult:
    """What one baseline produced on one trial.

    Attributes:
        name: Baseline identifier.
        positions_m: ``[K, 3]`` estimated positions, metres, head frame. Padded
            with ``nan`` rows if the method found fewer than ``K`` sources.
        seconds: Wall-clock fitting time, excluding any shared precomputation.
        n_found: How many sources the method actually reported, before truncation
            or padding to ``K``.
        residual_fraction: Fraction of the observation's energy left unexplained,
            where the method makes that available.
        power: ``[S]`` per-candidate activity, for the methods that produce a
            map. Deliberately **not** serialized by :meth:`to_dict`: it is 8196
            floats per session per method, which would dominate every result
            shard, and everything the report needs from it — the peak, and the
            spread about the peak — is a scalar computed at scoring time.
        detail: Method-specific extras for the report.
    """

    name: str
    positions_m: np.ndarray
    seconds: float
    n_found: int
    residual_fraction: float | None = None
    power: np.ndarray | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable record."""
        return {
            "name": self.name,
            "positions_m": np.asarray(self.positions_m).tolist(),
            "seconds": self.seconds,
            "n_found": self.n_found,
            "residual_fraction": self.residual_fraction,
            "detail": self.detail,
        }


def peak_locations(
    power: np.ndarray,
    positions_m: np.ndarray,
    n_peaks: int,
    *,
    exclusion_mm: float = PEAK_EXCLUSION_MM,
) -> tuple[np.ndarray, int]:
    """Read ``K`` well-separated maxima out of a per-source power map.

    Greedy: take the strongest source, exclude everything within ``exclusion_mm``,
    repeat. Without the exclusion, a distributed estimate's ``K`` strongest sources
    are ``K`` adjacent vertices of one blob, which would score as ``K`` correct
    detections of a single source.

    Args:
        power: ``[S]`` non-negative activity per candidate location.
        positions_m: ``[S, 3]`` candidate locations, metres.
        n_peaks: How many to extract.
        exclusion_mm: Exclusion radius.

    Returns:
        ``(positions [n_peaks, 3], n_found)``. Rows beyond ``n_found`` are ``nan``.
    """
    power = np.array(power, dtype=np.float64, copy=True)
    positions = np.asarray(positions_m, dtype=np.float64)
    found: list[np.ndarray] = []
    for _ in range(n_peaks):
        if not np.isfinite(power).any() or power.max() <= 0.0:
            break
        best = int(np.argmax(power))
        found.append(positions[best])
        power[np.linalg.norm(positions - positions[best], axis=1) * 1e3 < exclusion_mm] = -1.0
    n_found = len(found)
    padded = np.full((n_peaks, 3), np.nan)
    if n_found:
        padded[:n_found] = np.stack(found)
    return padded, n_found


class DipoleDictionary:
    """The OpenMEEG gain at every cortical candidate, and the scan built on it.

    Precomputing this costs about two minutes and 31 MB; every scan afterwards is
    milliseconds. The **unreferenced** gain is stored so that a channel subset can
    be given the average reference its own channel count implies rather than a row
    selection of the 64-channel one.

    Args:
        forward: An :class:`~neurolayout_shared.openmeeg_model.OpenMEEGForward`
            constructed with ``reference=False``.
        chunk: Locations per solver call; the intermediate source-term matrix is
            what limits this.
        stride: Keep every ``stride``-th candidate. ``1`` is the real baseline; a
            larger value is for tests, and coarsens the resolution floor
            proportionally, so it must never be used for a reported number.
        cache: Optional ``.npz`` to read the gain from, or write it to. Keyed by the
            head model's fingerprint, so a cache built for another geometry is
            rejected rather than silently used. Ignored when ``stride > 1``.
    """

    def __init__(
        self,
        forward: Any,
        *,
        chunk: int = 512,
        stride: int = 1,
        cache: str | Path | None = None,
    ) -> None:
        """Evaluate and cache the gain at every source-space location."""
        if getattr(forward, "reference", True):
            raise ValueError(
                "DipoleDictionary needs an unreferenced forward, so that channel "
                "subsets can carry their own average reference"
            )
        geometry = forward.geometry
        self.stride = int(stride)
        self.positions = np.asarray(geometry.source_space, dtype=np.float64)[:: self.stride]
        self.n_locations = int(self.positions.shape[0])
        fingerprint = geometry.fingerprint()

        cache_path = None if cache is None or self.stride != 1 else Path(cache)
        loaded = None
        if cache_path is not None and cache_path.exists():
            with np.load(cache_path, allow_pickle=False) as data:
                if str(data["fingerprint"]) == fingerprint:
                    loaded = np.asarray(data["gain_unreferenced"], dtype=np.float64)
                else:
                    raise ValueError(
                        f"dictionary cache {cache_path} belongs to another head model"
                    )
        if loaded is None:
            blocks = [
                forward.gain(self.positions[start : start + chunk])
                for start in range(0, self.n_locations, chunk)
            ]
            loaded = np.concatenate(blocks, axis=1)
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(
                    cache_path,
                    gain_unreferenced=loaded,
                    fingerprint=np.array(fingerprint),
                )
        #: ``[C, S, 3]`` unreferenced free-orientation gain, volts per A·m.
        self.gain_unreferenced = loaded
        self.from_cache = cache_path is not None and cache_path.exists()

    def gain(self, channels: tuple[int, ...] | None) -> np.ndarray:
        """``[C, S, 3]`` referenced gain for a channel subset (or all channels)."""
        from neurolayout_shared.source_model import average_reference_operator

        selected = (
            self.gain_unreferenced
            if channels is None
            else self.gain_unreferenced[list(channels)]
        )
        reference = average_reference_operator(selected.shape[0])
        return np.einsum("cd,dpj->cpj", reference, selected)

    def _explained_given(
        self,
        target: np.ndarray,
        fixed: np.ndarray | None,
        channels: tuple[int, ...] | None,
        forbidden: np.ndarray | None,
    ) -> np.ndarray:
        r"""Energy each candidate would explain *in addition to* a fixed subspace.

        For a candidate location ``p`` with topography subspace :math:`G_p`, and a
        fixed subspace spanned by the other sources' topographies,

        .. math::
            \lVert P_{\mathrm{span}(\text{fixed}, G_p)} y \rVert^2
                = \lVert P_{\text{fixed}} y\rVert^2
                + \lVert P_{Q G_p} (Q y) \rVert^2 ,
                \qquad Q = I - P_{\text{fixed}},

        so ranking candidates by the second term is exact rather than greedy. Every
        candidate is handled in one batched contraction plus a batch of 3x3 solves.

        Args:
            target: ``[C]`` the time-collapsed measurement.
            fixed: ``[C, m]`` columns already in the model, or ``None``.
            channels: Channel subset.
            forbidden: Boolean ``[S]`` mask of candidates to rule out.

        Returns:
            ``[S]`` additional explained energy, ``-inf`` where forbidden.
        """
        gain = self.gain(channels)  # [C, S, 3]
        if fixed is None or fixed.shape[1] == 0:
            projected = gain
            residual = target
        else:
            basis, _ = np.linalg.qr(fixed)
            residual = target - basis @ (basis.T @ target)
            projected = gain - np.einsum(
                "cm,mdj->cdj", basis, np.einsum("cm,cdj->mdj", basis, gain)
            )
        moved = np.moveaxis(projected, 1, 0)  # [S, C, 3]
        gram = np.einsum("scj,sck->sjk", moved, moved)
        rhs = np.einsum("scj,c->sj", moved, residual)
        # Ridge on the Gram matrix: a candidate whose projected subspace is nearly
        # degenerate -- a deep source, or one almost inside the fixed subspace --
        # would otherwise win by inverting numerical noise.
        scale = np.trace(gram, axis1=1, axis2=2)[:, None, None] / 3.0
        coefficients = np.linalg.solve(
            gram + 1e-12 * np.maximum(scale, 1e-300) * np.eye(3), rhs[..., None]
        )[..., 0]
        explained = np.einsum("sj,sj->s", coefficients, rhs)
        return explained if forbidden is None else np.where(forbidden, -np.inf, explained)

    def scan(
        self,
        observed: np.ndarray,
        waveform: np.ndarray,
        n_sources: int,
        *,
        channels: tuple[int, ...] | None = None,
        exclusion_mm: float = PEAK_EXCLUSION_MM,
        refine_sweeps: int = 3,
    ) -> BaselineResult:
        """Discrete dipole scan over the cortical dictionary, for ``K`` sources.

        With a known waveform the time axis collapses: the data is projected onto
        the waveform and a single topography is fitted. For ``K = 1`` this is
        exactly classical dipole scanning, and it is exhaustive — the best
        dictionary entry, found by looking at all of them.

        For ``K > 1`` a purely greedy pass is a **weak** baseline, and measurably
        so: the best single dipole for a two-source field is usually neither of the
        two, and the greedy pair then leaves ~3% of the energy unexplained on a
        problem where the right pair explains all of it. Since a weak baseline would
        flatter the method under test, the greedy pass is followed by **alternating
        refinement** — re-scan each source's location exhaustively with the others
        held fixed, repeat until nothing moves — which is the classical multi-dipole
        search and, on that same problem, recovers both locations exactly. It costs
        ``K`` extra scans per sweep, tenths of a second.

        This remains a *dictionary* method: it cannot place a source between
        candidates, which is the one thing the continuous estimator can do.

        Args:
            observed: ``[C, T]`` measured signals, volts.
            waveform: ``[T]`` the known source waveform.
            n_sources: ``K``.
            channels: Channel subset, or ``None`` for all.
            exclusion_mm: Minimum separation between chosen locations, so the
                search cannot spend two sources on one topography.
            refine_sweeps: Maximum alternating passes after the greedy
                initialization. ``0`` reduces this to plain orthogonal matching
                pursuit.

        Returns:
            A :class:`BaselineResult`.
        """
        start = time.perf_counter()
        gain = self.gain(channels)  # [C, S, 3]
        target = np.asarray(observed, dtype=np.float64) @ (
            np.asarray(waveform) / (np.asarray(waveform) @ np.asarray(waveform))
        )
        total = float(target @ target)

        def forbidden_mask(others: list[int]) -> np.ndarray | None:
            if not others:
                return None
            mask = np.zeros(self.n_locations, dtype=bool)
            for index in others:
                mask |= (
                    np.linalg.norm(self.positions - self.positions[index], axis=1) * 1e3
                    < exclusion_mm
                )
            return mask

        def columns(indices: list[int]) -> np.ndarray | None:
            if not indices:
                return None
            return np.concatenate([gain[:, index, :] for index in indices], axis=1)

        chosen: list[int] = []
        first_pass: np.ndarray | None = None
        for _ in range(n_sources):
            explained = self._explained_given(
                target, columns(chosen), channels, forbidden_mask(chosen)
            )
            if first_pass is None:
                # The single-source explained-energy map over every candidate.
                # It is the scan's own activity map, and the only cost of keeping
                # it is a reference, since it is computed either way.
                first_pass = np.maximum(explained, 0.0)
            chosen.append(int(np.argmax(explained)))

        sweeps_used = 0
        for _ in range(refine_sweeps):
            moved = False
            for slot in range(n_sources):
                others = [index for i, index in enumerate(chosen) if i != slot]
                explained = self._explained_given(
                    target, columns(others), channels, forbidden_mask(others)
                )
                best = int(np.argmax(explained))
                if best != chosen[slot]:
                    chosen[slot] = best
                    moved = True
            sweeps_used += 1
            if not moved:  # a fixed point; further sweeps cannot change anything
                break

        design = columns(chosen)
        fit, *_ = np.linalg.lstsq(design, target, rcond=None)
        residual = target - design @ fit
        return BaselineResult(
            name="scan",
            positions_m=self.positions[chosen],
            seconds=time.perf_counter() - start,
            n_found=len(chosen),
            residual_fraction=float(residual @ residual) / max(total, 1e-30),
            power=first_pass,
            detail={
                "n_locations": self.n_locations,
                "vertices": chosen,
                "exclusion_mm": exclusion_mm,
                "refine_sweeps_used": sweeps_used,
                "stride": self.stride,
            },
        )



class MneInverse:
    """MNE-Python's sparse and distributed inverse solvers, on OpenMEEG physics.

    Builds one ``Forward`` with MNE's bookkeeping and OpenMEEG's gain, then serves
    per-channel-subset copies of it. Construction needs MNE, a downloaded
    fsaverage and a few seconds; it is done once per benchmark run.

    Args:
        dictionary: The precomputed OpenMEEG gain, which also fixes the source
            space the ``Forward`` must be on.
        channel_names: Channel names in the frozen canonical order.
        sfreq: Sampling rate written into the ``Info``.
        verify_tolerance_mm: The MNE and OpenMEEG source spaces must agree to
            within this, or the gain substitution would scramble the solution.
    """

    def __init__(
        self,
        dictionary: DipoleDictionary,
        channel_names: tuple[str, ...],
        *,
        sfreq: float = 160.0,
        verify_tolerance_mm: float = 1e-6,
    ) -> None:
        """Build the substituted forward operator."""
        import mne

        from neurolayout.mismatch import (
            INFERENCE_ICO,
            INFERENCE_SKULL_CONDUCTIVITY,
            _bem_solution,
            _head_to_mri,
            _subjects_dir,
        )
        from neurolayout.montage import MONTAGE_NAME

        self.dictionary = dictionary
        self.channel_names = tuple(channel_names)

        info = mne.create_info(list(self.channel_names), sfreq, "eeg")
        info.set_montage(mne.channels.make_standard_montage(MONTAGE_NAME), verbose="ERROR")
        self._info = info

        source_space = mne.setup_source_space(
            "fsaverage",
            spacing="ico5",
            subjects_dir=_subjects_dir(),
            add_dist=False,
            verbose="ERROR",
        )
        forward = mne.make_forward_solution(
            info,
            trans=_head_to_mri(),
            src=source_space,
            bem=_bem_solution(INFERENCE_ICO, INFERENCE_SKULL_CONDUCTIVITY),
            eeg=True,
            meg=False,
            mindist=0.0,
            verbose="ERROR",
        )
        positions = np.concatenate(
            [hemi["rr"][hemi["vertno"]] for hemi in forward["src"]], axis=0
        )
        offset_mm = float(np.abs(positions - dictionary.positions).max() * 1e3)
        if offset_mm > verify_tolerance_mm:
            raise RuntimeError(
                "MNE's ico5 source space disagrees with the cached one by "
                f"{offset_mm:.3g} mm; substituting the OpenMEEG gain would pair "
                "each column with the wrong location"
            )
        self._forward = forward
        self._per_subset: dict[tuple[int, ...] | None, Any] = {}

    @classmethod
    def from_parts(
        cls,
        dictionary: DipoleDictionary,
        forward: Any,
        info: Any,
        *,
        verify_tolerance_mm: float = 1e-6,
    ) -> MneInverse:
        """Build from an already-constructed ``Forward``, skipping the fsaverage path.

        The template benchmark needs MNE to *build* a forward on fsaverage before
        the OpenMEEG gain can be substituted into it. A caller that already holds
        a ``Forward`` on the right source space does not, and rebuilding one would
        risk landing on a different source space from the one the dictionary was
        evaluated at. This constructor takes the pieces directly and re-runs the
        same source-space agreement check, which is what makes the gain
        substitution safe.

        Args:
            dictionary: The OpenMEEG gain over the same source space.
            forward: An MNE ``Forward`` whose ``src`` matches ``dictionary``.
            info: An ``Info`` carrying the channel names, order and montage.
            verify_tolerance_mm: Largest tolerated source-space disagreement.

        Returns:
            The configured :class:`MneInverse`.

        Raises:
            RuntimeError: If the two source spaces disagree, which would pair
                every gain column with the wrong location.
        """
        instance = cls.__new__(cls)
        instance.dictionary = dictionary
        instance.channel_names = tuple(str(name) for name in info["ch_names"])
        instance._info = info
        positions = np.concatenate(
            [hemi["rr"][hemi["vertno"]] for hemi in forward["src"]], axis=0
        )
        offset_mm = float(np.abs(positions - dictionary.positions).max() * 1e3)
        if offset_mm > verify_tolerance_mm:
            raise RuntimeError(
                "the supplied forward's source space disagrees with the cached one "
                f"by {offset_mm:.3g} mm; substituting the OpenMEEG gain would pair "
                "each column with the wrong location"
            )
        instance._forward = forward
        instance._per_subset = {}
        return instance

    def forward_for(self, channels: tuple[int, ...] | None) -> Any:
        """A ``Forward`` restricted to a channel subset, carrying OpenMEEG's gain."""
        import mne

        if channels in self._per_subset:
            return self._per_subset[channels]
        names = (
            list(self.channel_names)
            if channels is None
            else [self.channel_names[index] for index in channels]
        )
        forward = mne.pick_channels_forward(
            self._forward, include=names, ordered=True, verbose="ERROR"
        )
        gain = self.dictionary.gain(channels)  # [C, S, 3], referenced
        flat = gain.reshape(gain.shape[0], -1)
        if flat.shape != forward["sol"]["data"].shape:
            raise RuntimeError(
                f"substituted gain {flat.shape} does not match MNE's "
                f"{forward['sol']['data'].shape}"
            )
        forward["sol"]["data"] = np.asarray(flat, dtype=np.float64)
        # `convert_forward_solution` rebuilds `sol` from `_orig_sol`, so both have
        # to be replaced or the substitution would silently be undone.
        forward["_orig_sol"] = np.asarray(flat, dtype=np.float64)
        self._per_subset[channels] = forward
        return forward

    def _evoked(self, observed: np.ndarray, channels: tuple[int, ...] | None) -> Any:
        import mne

        names = (
            list(self.channel_names)
            if channels is None
            else [self.channel_names[index] for index in channels]
        )
        info = mne.create_info(names, self._info["sfreq"], "eeg")
        info.set_montage(self._info.get_montage(), on_missing="ignore", verbose="ERROR")
        evoked = mne.EvokedArray(
            np.asarray(observed, dtype=np.float64), info, tmin=0.0, verbose="ERROR"
        )
        # MNE's inverse solvers require the average reference to be represented as
        # a projector, not merely applied. The data already carries the reference,
        # so adding the projector is numerically idempotent — but MNE also applies
        # the same projector to the forward operator during whitening, which is
        # exactly the consistency we want, since the substituted gain is referenced
        # too.
        return evoked.set_eeg_reference("average", projection=True, verbose="ERROR")

    def _covariance(
        self, noise_rms_v: float, signal_rms_v: float, channels: tuple[int, ...] | None
    ) -> Any:
        """A diagonal covariance at the true noise level.

        Diagonal rather than the true correlated one because NeuroLocate's plain
        least-squares loss makes exactly the white-noise assumption too; giving the
        baselines the generating covariance while the method under test does not
        have it would be the comparison rigged the other way. The floor keeps the
        noise-free conditions solvable — a zero-variance covariance has no inverse.
        """
        import mne

        names = (
            list(self.channel_names)
            if channels is None
            else [self.channel_names[index] for index in channels]
        )
        variance = max(noise_rms_v, 1e-4 * signal_rms_v) ** 2
        return mne.Covariance(
            data=np.eye(len(names)) * variance,
            names=names,
            bads=[],
            projs=[],
            nfree=1,
            verbose="ERROR",
        )

    def irmxne(
        self,
        observed: np.ndarray,
        n_sources: int,
        *,
        noise_rms_v: float,
        channels: tuple[int, ...] | None = None,
        n_iter: int = 5,
        alpha: float | str = DEFAULT_MXNE_ALPHA,
        loose: float = DEFAULT_LOOSE,
        depth: float | None = DEFAULT_DEPTH,
        exclusion_mm: float = PEAK_EXCLUSION_MM,
    ) -> BaselineResult:
        """Iteratively reweighted mixed-norm estimate (irMxNE).

        Args:
            observed: ``[C, T]`` measured signals, volts.
            n_sources: ``K``, used only for scoring — MxNE chooses its own count.
            noise_rms_v: True sensor-noise RMS, for the covariance.
            channels: Channel subset, or ``None``.
            n_iter: Reweighting iterations; 1 is plain MxNE.
            alpha: Regularization, as a percentage of the maximum. ``"sure"``
                selects it by Stein's unbiased risk estimate instead, at a cost of
                minutes per trial.
            loose: Orientation constraint; see :data:`DEFAULT_LOOSE` for the
                measurement behind the default.
            depth: Depth-weighting exponent.
            exclusion_mm: Peak separation when reading ``K`` locations out.

        Returns:
            A :class:`BaselineResult`. On solver failure — including the common
            case of MxNE returning an empty solution at high regularization — the
            result carries ``n_found=0`` and the reason, rather than raising.
        """
        from mne.inverse_sparse import mixed_norm

        observed = np.asarray(observed, dtype=np.float64)
        evoked = self._evoked(observed, channels)
        covariance = self._covariance(
            noise_rms_v, float(np.sqrt(np.mean(observed**2))), channels
        )
        start = time.perf_counter()
        try:
            stc = mixed_norm(
                evoked,
                self.forward_for(channels),
                covariance,
                alpha=alpha,
                n_mxne_iter=n_iter,
                loose=loose,
                depth=depth,
                weights=None,
                verbose="ERROR",
            )
        except Exception as error:  # noqa: BLE001 - a failed baseline is a datum
            return BaselineResult(
                name="irmxne",
                positions_m=np.full((n_sources, 3), np.nan),
                seconds=time.perf_counter() - start,
                n_found=0,
                detail={
                    "error": f"{type(error).__name__}: {error}",
                    "alpha": alpha,
                    "loose": loose,
                    "depth": depth,
                },
            )
        seconds = time.perf_counter() - start

        active = np.concatenate([hemi for hemi in stc.vertices])
        power = np.zeros(self.dictionary.n_locations)
        if len(active):
            index = self._active_indices(stc)
            power[index] = np.max(np.abs(stc.data), axis=1)
        positions, n_found = peak_locations(
            power, self.dictionary.positions, n_sources, exclusion_mm=exclusion_mm
        )
        return BaselineResult(
            name="irmxne",
            positions_m=positions,
            seconds=seconds,
            n_found=n_found,
            power=power,
            detail={
                "n_active": int(len(active)),
                "alpha": alpha,
                "n_iter": n_iter,
                "loose": loose,
                "depth": depth,
                "truncated": bool(len(active) > n_sources),
            },
        )

    def _active_indices(self, stc: Any) -> np.ndarray:
        """Map an ``stc``'s per-hemisphere vertices onto dictionary rows."""
        # The dictionary is ordered left-hemisphere vertices then right, exactly as
        # `build_openmeeg_headmodel.py` concatenated them, and the forward was
        # verified against it to 0 mm. So a vertex's row is its rank within its
        # hemisphere's `vertno`, offset by the left hemisphere's size.
        left_vertno, right_vertno = (hemi["vertno"] for hemi in self._forward["src"])
        offsets = (0, len(left_vertno))
        rows = []
        for hemi_index, (vertices, vertno) in enumerate(
            zip(stc.vertices, (left_vertno, right_vertno), strict=True)
        ):
            lookup = {int(v): i for i, v in enumerate(vertno)}
            rows.extend(offsets[hemi_index] + lookup[int(v)] for v in vertices)
        return np.asarray(rows, dtype=int)

    def dspm(
        self,
        observed: np.ndarray,
        n_sources: int,
        *,
        noise_rms_v: float,
        channels: tuple[int, ...] | None = None,
        snr: float = 3.0,
        loose: float = DEFAULT_LOOSE,
        depth: float | None = DEFAULT_DEPTH,
        exclusion_mm: float = PEAK_EXCLUSION_MM,
    ) -> BaselineResult:
        """dSPM: a distributed minimum-norm estimate, scored on its ``K`` peaks.

        Args:
            observed: ``[C, T]`` measured signals, volts.
            n_sources: ``K`` peaks to read out.
            noise_rms_v: True sensor-noise RMS, for the covariance.
            channels: Channel subset, or ``None``.
            snr: Assumed amplitude SNR setting ``lambda2 = 1 / snr^2``, MNE's
                convention. Left at its documented default of 3 rather than tuned.
            loose: Orientation constraint; see :data:`DEFAULT_LOOSE`.
            depth: Depth-weighting exponent.
            exclusion_mm: Peak separation.

        Returns:
            A :class:`BaselineResult`.
        """
        from mne.minimum_norm import apply_inverse, make_inverse_operator

        observed = np.asarray(observed, dtype=np.float64)
        evoked = self._evoked(observed, channels)
        covariance = self._covariance(
            noise_rms_v, float(np.sqrt(np.mean(observed**2))), channels
        )
        start = time.perf_counter()
        operator = make_inverse_operator(
            evoked.info,
            self.forward_for(channels),
            covariance,
            loose=loose,
            depth=depth,
            verbose="ERROR",
        )
        stc = apply_inverse(
            evoked, operator, lambda2=1.0 / snr**2, method="dSPM", verbose="ERROR"
        )
        seconds = time.perf_counter() - start

        power = np.max(np.abs(stc.data), axis=1)
        if power.shape[0] != self.dictionary.n_locations:
            raise RuntimeError(
                f"dSPM returned {power.shape[0]} sources, expected "
                f"{self.dictionary.n_locations}"
            )
        positions, n_found = peak_locations(
            power, self.dictionary.positions, n_sources, exclusion_mm=exclusion_mm
        )
        return BaselineResult(
            name="dspm",
            positions_m=positions,
            seconds=seconds,
            n_found=n_found,
            power=power,
            detail={"snr": snr, "lambda2": 1.0 / snr**2, "loose": loose, "depth": depth},
        )
