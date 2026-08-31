#!/usr/bin/env python
# Copyright 2026 NeuroLocate contributors
# SPDX-License-Identifier: Apache-2.0
r"""Build the cortical surface mesh the K=2 visual is drawn on.

The cached head-model artifact carries the 20 484 ico5 source-space positions but
not their triangles, and a surface cannot be drawn from a point cloud. This
writes ``results/cortex_ico5.npz``: the same vertices, in the same order, plus
the faces and the FreeSurfer curvature used for shading. The vertex order is
checked against the artifact's own ``source_space`` to 0 mm, so a per-location
value can be painted straight onto the mesh with no resampling.

Needs MNE and the fsaverage anatomy, which it will download on first use. The
output is committed, so this only has to run if the mesh itself is rebuilt.

Usage::

    make cortex-mesh
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "app"))
sys.path.insert(0, str(REPO_ROOT / "components" / "shared_code"))

from neurolayout_shared.openmeeg_model import (  # noqa: E402
    HeadGeometry,
    default_artifact_path,
)


def build_cortex(out: Path, geometry: HeadGeometry) -> None:
    """Write the ico5 cortical mesh in head coordinates."""
    import mne
    import nibabel.freesurfer as freesurfer
    from mne.transforms import apply_trans, invert_transform

    subjects_dir = mne.datasets.fetch_fsaverage(verbose="ERROR").parent
    source_space = mne.setup_source_space(
        "fsaverage", spacing="ico5", subjects_dir=subjects_dir,
        add_dist=False, verbose="ERROR",
    )
    mri_to_head = invert_transform(
        mne.read_trans(Path(subjects_dir) / "fsaverage" / "bem" / "fsaverage-trans.fif")
    )

    vertices, triangles, offset = [], [], 0
    for hemisphere in source_space:
        used = hemisphere["vertno"]
        vertices.append(hemisphere["rr"][used])
        # `use_tris` indexes the full surface; remap it into `vertno` order, which
        # is the order every reported per-location quantity is in.
        remap = np.full(hemisphere["np"], -1)
        remap[used] = np.arange(len(used))
        faces = remap[hemisphere["use_tris"]]
        if (faces < 0).any():
            raise RuntimeError("a decimated triangle refers to an unused vertex")
        triangles.append(faces + offset)
        offset += len(used)

    vertices = apply_trans(mri_to_head, np.concatenate(vertices))
    triangles = np.concatenate(triangles)

    disagreement_mm = float(
        np.linalg.norm(vertices - geometry.source_space, axis=1).max() * 1e3
    )
    if disagreement_mm > 1e-9:
        raise RuntimeError(
            f"the rebuilt source space disagrees with the cached head model by "
            f"{disagreement_mm:.3g} mm; painting a per-location value onto this "
            "mesh would put every value on the wrong vertex"
        )

    curvature = np.concatenate([
        freesurfer.read_morph_data(
            Path(subjects_dir) / "fsaverage" / "surf" / f"{hemi}.curv"
        )[space["vertno"]]
        for hemi, space in zip(("lh", "rh"), source_space, strict=True)
    ])

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        vertices=vertices,
        triangles=triangles,
        n_lh=len(source_space[0]["vertno"]),
        curv=curvature,
    )
    print(
        f"wrote {out} — {len(vertices)} vertices, {len(triangles)} triangles, "
        f"aligned to the cached source space to {disagreement_mm:.1e} mm"
    )


def main() -> int:
    """Build the mesh."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "results" / "cortex_ico5.npz"
    )
    arguments = parser.parse_args()
    build_cortex(arguments.out, HeadGeometry.load(default_artifact_path()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
