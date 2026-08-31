# Copyright 2026 NeuroLocate contributors
# SPDX-License-Identifier: Apache-2.0
"""The frozen 64-channel sensor array, and the standard montage it comes from.

Every array in the project is indexed by the same channel position: the
``[B, 64, T]`` epoch tensor, the ``[64, 3]`` electrode geometry, the ``[64, 64]``
average-reference operator and the ``[64, S]`` forward matrix.
:data:`CANONICAL_CHANNELS` is that index. It is written down here rather than
read out of a file at runtime, so a divergence becomes a test failure instead of
a subtly wrong inverse solution.

The order is the one the PhysioNet EEG Motor Movement/Imagery recordings store,
after MNE's ``eegbci.standardize`` strips the padding dots and applies 10-05
capitalization (``Fc5.`` -> ``FC5``). That corpus is not used anywhere in this
repository; only its channel order is, because the packaged proposal checkpoint
and the cached head model were both built against it.
"""

from __future__ import annotations

__all__ = ["MONTAGE_NAME", "CANONICAL_CHANNELS"]

#: MNE standard montage the template sensor geometry comes from. The 64-channel
#: layout is the Sharbrough 10-10 montage, a subset of 10-05.
MONTAGE_NAME = "standard_1005"

#: The 64 EEG channels, in the frozen order.
CANONICAL_CHANNELS: tuple[str, ...] = (
    "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
    "Fp1", "Fpz", "Fp2",
    "AF7", "AF3", "AFz", "AF4", "AF8",
    "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
    "FT7", "FT8", "T7", "T8", "T9", "T10", "TP7", "TP8",
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POz", "PO4", "PO8",
    "O1", "Oz", "O2", "Iz",
)  # fmt: skip
