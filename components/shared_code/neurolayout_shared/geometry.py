# Copyright 2026 NeuroLocate contributors
# SPDX-License-Identifier: Apache-2.0
"""Deterministic geometry helpers shared by the NeuroLayout Tesseracts.

Everything in this module is pure NumPy and deterministic: given the same
integer counts it returns bit-identical arrays on every call. That property is
what lets the head model be rebuilt from scratch inside a container instead of
shipping a binary artifact.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "fibonacci_directions",
    "normalize",
    "pairwise_angles",
    "min_pairwise_angle",
    "icosphere",
]


def normalize(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Scale each row of ``vectors`` to unit length."""
    vectors = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, eps)


def fibonacci_directions(n: int) -> np.ndarray:
    """Return ``n`` near-uniformly spread unit vectors on the sphere.

    Uses the spherical Fibonacci (golden-angle) lattice. Deterministic and
    quasi-uniform, which is what we want for both the scalp sampling mesh and
    the cortical source basis.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    k = np.arange(n, dtype=np.float64)
    # z runs from near +1 to near -1 with equal spacing (equal-area in z).
    z = 1.0 - (2.0 * k + 1.0) / n
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    theta = golden_angle * k
    return np.stack([radius * np.cos(theta), radius * np.sin(theta), z], axis=1)


def pairwise_angles(directions: np.ndarray) -> np.ndarray:
    """Great-circle angles (radians) between every pair of unit directions."""
    unit = normalize(directions)
    cos = np.clip(unit @ unit.T, -1.0, 1.0)
    return np.arccos(cos)


def min_pairwise_angle(directions: np.ndarray) -> float:
    """Smallest great-circle angle (radians) between two distinct directions."""
    angles = pairwise_angles(directions)
    n = angles.shape[0]
    if n < 2:
        return float(np.pi)
    off_diagonal = angles[~np.eye(n, dtype=bool)]
    return float(off_diagonal.min())


def icosphere(subdivisions: int) -> tuple[np.ndarray, np.ndarray]:
    """A closed unit-sphere triangulation by recursive icosahedron subdivision.

    Used to build concentric-sphere BEM geometries for the physics gate: a
    three-layer BEM with equal conductivities must reproduce the analytic
    homogeneous-sphere potential, which is the check that pins down OpenMEEG's
    winding convention and quantifies the discretization error.

    Args:
        subdivisions: 0 gives the bare icosahedron (12 vertices); each level
            quadruples the triangle count. Level 3 gives 642 vertices / 1280
            triangles, matching the default BEM decimation.

    Returns:
        ``(vertices, triangles)`` with unit-norm vertices and triangles wound so
        the right-hand-rule normal points outward.
    """
    if subdivisions < 0:
        raise ValueError(f"subdivisions must be >= 0, got {subdivisions}")
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = [
        np.array(v, dtype=np.float64)
        for v in (
            (-1, phi, 0), (1, phi, 0), (-1, -phi, 0), (1, -phi, 0),
            (0, -1, phi), (0, 1, phi), (0, -1, -phi), (0, 1, -phi),
            (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1),
        )
    ]
    vertices = [v / np.linalg.norm(v) for v in vertices]
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]

    for _ in range(subdivisions):
        midpoints: dict[tuple[int, int], int] = {}

        def midpoint(a: int, b: int, cache: dict = midpoints) -> int:
            key = (min(a, b), max(a, b))
            if key not in cache:
                point = vertices[a] + vertices[b]
                vertices.append(point / np.linalg.norm(point))
                cache[key] = len(vertices) - 1
            return cache[key]

        # Each triangle splits into four, all inheriting the parent's winding.
        faces = [
            new_face
            for a, b, c in faces
            for new_face in (
                (a, midpoint(a, b), midpoint(c, a)),
                (b, midpoint(b, c), midpoint(a, b)),
                (c, midpoint(c, a), midpoint(b, c)),
                (midpoint(a, b), midpoint(b, c), midpoint(c, a)),
            )
        ]

    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)
