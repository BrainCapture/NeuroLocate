#!/usr/bin/env python
# Copyright 2026 NeuroLocate contributors
# SPDX-License-Identifier: Apache-2.0
"""Copy the proposal architecture into the ``proposal`` component.

The component image carries PyTorch and nothing else — no JAX, no OpenMEEG, no
MNE, and not the orchestrator package. So the network definition is duplicated
into it rather than imported, and this script is the only thing allowed to write
that copy. ``tests/test_components.py`` checks that the two agree.

Usage::

    python scripts/sync_proposal_component.py           # write
    python scripts/sync_proposal_component.py --check   # verify, exit 1 on drift
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPO_ROOT / "app" / "neurolayout" / "hybrid" / "model.py"
TARGET = REPO_ROOT / "components" / "tesseracts" / "proposal" / "proposal_model.py"

#: Everything after this marker must match the source file exactly. The component
#: copy carries an extra "generated, do not edit" note above it.
MARKER = "The global source-set proposal network."


def rendered() -> str:
    """What the component copy should contain."""
    text = SOURCE.read_text()
    body = text[text.index(MARKER) :]
    header = (
        'r"""The global source-set proposal network — the component\'s copy.\n\n'
        ".. note::\n\n"
        "   This file is generated from ``app/neurolayout/hybrid/model.py`` by\n"
        "   ``scripts/sync_proposal_component.py`` and must not be edited here.\n"
        "   The component image carries only PyTorch, so it cannot import the\n"
        "   orchestrator package; the architecture is therefore duplicated rather\n"
        "   than shared, and the sync script's ``--check`` mode is what keeps the\n"
        "   two honest.\n\n"
    )
    prefix = text[: text.index('r"""The global source-set proposal network.')]
    return prefix + header + body


def main() -> int:
    """Write or check the component copy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    wanted = rendered()
    if arguments.check:
        if not TARGET.exists() or TARGET.read_text() != wanted:
            print(f"{TARGET} is out of date; run scripts/sync_proposal_component.py")
            return 1
        print(f"{TARGET} is up to date")
        return 0
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(wanted)
    print(f"wrote {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
