# Copyright 2026 NeuroLocate contributors
# SPDX-License-Identifier: Apache-2.0
r"""Fixed electrode subsets for the sensor-count sweep.

How many electrodes an inverse method needs is a real question, and it is easy to
answer dishonestly: pick the subset that works. These subsets are therefore
**standard clinical and commercial arrays**, named and frozen here before any
result was produced, and never re-chosen. They are nested —
``SUBSET_16 ⊂ SUBSET_32 ⊂`` the full 64 — so the sweep varies electrode count and
nothing else.

``all`` (64)
    The frozen 64-channel array, :data:`neurolayout.montage.CANONICAL_CHANNELS`.

``cap32`` (32)
    A standard 32-electrode 10–10 cap layout. The usual ``FT9/FT10/TP9/TP10``
    positions are replaced by their inner neighbours ``FT7/FT8/TP7/TP8``, because
    the frozen 64-channel array does not contain the outer ring.

``clinical16`` (16)
    The classical 10–20 array without its midline chain (``Fz``, ``Cz``, ``Pz``) —
    the layout of a conventional 16-channel clinical recording. Dropping the
    midline rather than a lateral pair is what keeps the remaining array
    left-right symmetric, which matters for a method whose failure mode is
    mislateralization.

Each name resolves to indices into the canonical 64-channel order, which is what
the ``headfield`` Tesseract's ``channel_subset`` field takes. Because subsetting
happens before referencing, a ``clinical16`` run carries a 16-channel average
reference — the reference such a recording would actually have.
"""

from __future__ import annotations

from neurolayout.montage import CANONICAL_CHANNELS

__all__ = [
    "SUBSET_NAMES",
    "CHANNEL_SUBSETS",
    "subset_names",
    "subset_indices",
]

#: 32-electrode 10–10 cap, in the conventional cap ordering.
_CAP32 = (
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "FT7", "FC5", "FC1", "FC2", "FC6", "FT8",
    "T7", "C3", "Cz", "C4", "T8",
    "TP7", "CP5", "CP1", "CP2", "CP6", "TP8",
    "P7", "P3", "Pz", "P4", "P8",
    "O1", "Oz", "O2",
)  # fmt: skip

#: Classical 10–20 array minus the midline chain: a 16-channel clinical montage.
_CLINICAL16 = (
    "Fp1", "Fp2", "F7", "F3", "F4", "F8",
    "T7", "C3", "C4", "T8",
    "P7", "P3", "P4", "P8",
    "O1", "O2",
)  # fmt: skip

#: name -> channel names, in the order the subset is conventionally listed.
CHANNEL_SUBSETS: dict[str, tuple[str, ...]] = {
    "all": CANONICAL_CHANNELS,
    "cap32": _CAP32,
    "clinical16": _CLINICAL16,
}

#: Subset names, largest first.
SUBSET_NAMES: tuple[str, ...] = ("all", "cap32", "clinical16")


def subset_names(name: str) -> tuple[str, ...]:
    """Channel names of a named subset, in canonical (not cap) order."""
    try:
        wanted = set(CHANNEL_SUBSETS[name])
    except KeyError:
        raise KeyError(
            f"unknown channel subset {name!r}; known: {sorted(CHANNEL_SUBSETS)}"
        ) from None
    return tuple(channel for channel in CANONICAL_CHANNELS if channel in wanted)


def subset_indices(name: str) -> tuple[int, ...] | None:
    """Indices into the canonical order, or ``None`` for the full array.

    ``None`` rather than ``tuple(range(64))`` because that is what the Tesseract's
    "keep everything" sentinel is, and passing an explicit full list would make
    every 64-channel run take the subsetting code path for no reason.

    Raises:
        KeyError: If ``name`` is not a known subset.
        ValueError: If a subset names a channel the canonical array lacks, which
            would silently shrink the subset.
    """
    if name == "all":
        if set(CHANNEL_SUBSETS["all"]) != set(CANONICAL_CHANNELS):
            raise ValueError("the 'all' subset has drifted from the canonical array")
        return None
    wanted = CHANNEL_SUBSETS[name]
    missing = [channel for channel in wanted if channel not in CANONICAL_CHANNELS]
    if missing:
        raise ValueError(
            f"channel subset {name!r} names channels the canonical 64 do not have: {missing}"
        )
    return tuple(
        index for index, channel in enumerate(CANONICAL_CHANNELS) if channel in set(wanted)
    )
