# Copyright 2026 NeuroLocate contributors
# SPDX-License-Identifier: Apache-2.0
"""Gate G, part one — the cached template head model is geometrically sane.

Coordinate errors in a forward model are the classic silent failure: nothing
crashes, every number is finite, and the recovered source is confidently wrong.
So the frame, the units, the anatomical orientation, the channel order and the
reference operator are each asserted here rather than assumed.

Everything in these tests is cheap: the artifact is loaded once and no BEM
system is assembled.
"""

from __future__ import annotations

import numpy as np
import pytest
from neurolayout.montage import CANONICAL_CHANNELS
from neurolayout_shared.openmeeg_model import (
    DEFAULT_CONDUCTIVITIES,
    SURFACE_NAMES,
    HeadGeometry,
    default_artifact_path,
    is_closed,
    signed_volume,
)


@pytest.fixture(scope="module")
def geometry() -> HeadGeometry:
    path = default_artifact_path()
    if not path.exists():
        pytest.skip(f"no head-model artifact at {path}; run scripts/build_openmeeg_headmodel.py")
    return HeadGeometry.load(path)


def test_artifact_ships_with_the_package(geometry: HeadGeometry) -> None:
    """The artifact must be importable package data, not a path on someone's disk.

    This is what lets the Tesseract image run the BEM forward without MNE,
    nibabel, or a downloaded fsaverage.
    """
    assert default_artifact_path().is_file()
    assert geometry.metadata["subject"] == "fsaverage"
    assert geometry.metadata["openmeeg_version"].startswith("2.")


def test_fingerprint_detects_tampering(geometry: HeadGeometry) -> None:
    """A modified geometry must not keep its old identity."""
    original = geometry.fingerprint()
    moved = np.array(geometry.sensor_xyz, copy=True)
    moved[0, 0] += 1e-6
    tampered = HeadGeometry(
        vertices=geometry.vertices,
        triangles=geometry.triangles,
        conductivities=geometry.conductivities,
        sensor_xyz=moved,
        channel_names=geometry.channel_names,
        source_space=geometry.source_space,
        source_normals=geometry.source_normals,
        metadata=geometry.metadata,
    )
    assert tampered.fingerprint() != original


def test_frame_and_units_are_declared(geometry: HeadGeometry) -> None:
    assert geometry.coord_frame == "mne-head"
    assert geometry.units == "m"
    # Metres, not millimetres: a head is ~0.1 across, not ~100.
    radii = np.linalg.norm(geometry.sensor_xyz, axis=1)
    assert 0.05 < radii.min() < 0.2
    assert radii.max() < 0.2


def test_channel_order_is_the_frozen_benchmark_order(geometry: HeadGeometry) -> None:
    """The forward operator is indexed by the frozen canonical channel list.

    If these ever diverge, every recovered source is wrong and nothing fails.
    """
    assert geometry.channel_names == tuple(CANONICAL_CHANNELS)
    assert geometry.sensor_xyz.shape == (64, 3)


def test_anatomical_orientation(geometry: HeadGeometry) -> None:
    """+x is right, +y is anterior, +z is superior — checked against electrodes."""
    index = {name: i for i, name in enumerate(geometry.channel_names)}
    xyz = geometry.sensor_xyz
    assert xyz[index["T7"], 0] < 0 < xyz[index["T8"], 0], "T7/T8 straddle the midline"
    assert xyz[index["C3"], 0] < xyz[index["C4"], 0]
    assert xyz[index["Fpz"], 1] > xyz[index["Oz"], 1], "Fpz is anterior to Oz"
    assert xyz[index["Cz"], 2] > xyz[index["Iz"], 2], "Cz is superior to Iz"
    # Midline electrodes really are on the midline.
    midline = [index[name] for name in ("Fpz", "Fz", "Cz", "Pz", "Oz", "Iz")]
    assert np.abs(xyz[midline, 0]).max() < 5e-3


def test_surfaces_are_closed_outward_and_nested(geometry: HeadGeometry) -> None:
    volumes = []
    for name, verts, tris in zip(
        SURFACE_NAMES, geometry.vertices, geometry.triangles, strict=True
    ):
        assert is_closed(tris), f"{name} is not a closed manifold"
        volume = signed_volume(verts, tris)
        assert volume > 0, f"{name} is wound inward"
        volumes.append(volume)
    # inner skull < outer skull < scalp
    assert volumes[0] < volumes[1] < volumes[2]
    # A human head is roughly 3-5 litres of scalp-enclosed volume.
    assert 2e-3 < volumes[2] < 8e-3


def test_conductivities_are_the_documented_values(geometry: HeadGeometry) -> None:
    np.testing.assert_allclose(geometry.conductivities, DEFAULT_CONDUCTIVITIES)
    brain, skull, scalp = geometry.conductivities
    assert brain / skull == pytest.approx(50.0), "the brain:skull ratio is the number that matters"
    assert scalp == brain


def test_sensors_sit_on_the_scalp(geometry: HeadGeometry) -> None:
    """Template electrodes and the fsaverage scalp must actually coincide.

    Compared against the nearest *vertex* of a decimated mesh, so a few
    millimetres is expected — half an edge length. Centimetres would mean the
    montage and the anatomy are in different frames.
    """
    scalp = geometry.vertices[2]
    distance = np.linalg.norm(
        geometry.sensor_xyz[:, None, :] - scalp[None], axis=-1
    ).min(axis=1)
    assert distance.max() < 0.012, f"worst electrode is {distance.max() * 1e3:.1f} mm off the scalp"
    assert np.median(distance) < 0.008


def test_source_space_is_inside_the_inner_skull(geometry: HeadGeometry) -> None:
    """The reference cortical source space is at the ~20k scale and inside the brain."""
    assert 15000 < geometry.n_sources < 25000
    low, high = geometry.brain_extent()
    assert (geometry.source_space >= low).all()
    assert (geometry.source_space <= high).all()
    np.testing.assert_allclose(
        np.linalg.norm(geometry.source_normals, axis=1), 1.0, atol=1e-6
    )


def test_source_space_is_bilateral(geometry: HeadGeometry) -> None:
    """Both hemispheres are present and roughly balanced."""
    left = (geometry.source_space[:, 0] < 0).sum()
    right = (geometry.source_space[:, 0] > 0).sum()
    assert 0.4 < left / (left + right) < 0.6


def test_reference_operator_is_a_projector(geometry: HeadGeometry) -> None:
    """``R`` is symmetric, idempotent, and annihilates the common mode."""
    operator = geometry.reference_operator()
    assert operator.shape == (64, 64)
    np.testing.assert_allclose(operator, operator.T, atol=0)
    np.testing.assert_allclose(operator @ operator, operator, atol=1e-12)
    np.testing.assert_allclose(operator @ np.ones(64), np.zeros(64), atol=1e-12)
