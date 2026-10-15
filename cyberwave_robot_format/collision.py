# Copyright [2025] Cyberwave
# Licensed under the Apache License, Version 2.0

"""Collision-shape helpers shared by the importers.

MuJoCo collides a mesh as its convex hull, so a hollow scene (a room, a tunnel, a
shelf) imported as one mesh collides as a solid block. :func:`hull_volume_ratio`
measures how far a mesh is from convex, and :func:`hollow_collision_warning` turns a
low ratio into the sentence shown to the uploader. Neither needs USD.
"""

from __future__ import annotations

import numpy as np

#: Below this mesh-volume / hull-volume ratio a mesh is treated as hollow.
HOLLOW_HULL_RATIO = 0.35

#: The hull is built from at most this many vertices (a fixed-seed sample above it),
#: which bounds the cost on multi-hundred-thousand-triangle scenes.
_MAX_HULL_POINTS = 100_000


def hull_volume_ratio(vertices: np.ndarray, faces: np.ndarray) -> float | None:
    """Mesh volume divided by convex-hull volume, in ``[0, 1]``.

    The mesh volume is the signed sum over triangles (divergence theorem), so it does
    not need the mesh to be watertight; overlapping solids can only push it up, never
    down, so a hollow scene is never mistaken for a solid one. ``None`` when the
    ratio cannot be computed (degenerate input, or ``scipy`` is not installed).
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if len(vertices) < 4 or len(faces) == 0:
        return None
    try:
        from scipy.spatial import ConvexHull
    except ImportError:
        return None

    tri = vertices[faces]
    volume = abs(
        float(np.einsum("ij,ij->", tri[:, 0], np.cross(tri[:, 1], tri[:, 2]))) / 6.0
    )
    points = np.unique(np.round(vertices, 6), axis=0)
    if len(points) > _MAX_HULL_POINTS:
        rng = np.random.default_rng(0)
        points = points[rng.choice(len(points), _MAX_HULL_POINTS, replace=False)]
    try:
        hull_volume = float(ConvexHull(points).volume)
    except Exception:  # Qhull raises its own error type for flat/degenerate input
        return None
    if hull_volume <= 0:
        return None
    return min(1.0, volume / hull_volume)


def hollow_collision_warning(ratio: float | None) -> str | None:
    """The uploader-facing warning for a hollow mesh, or ``None`` if it is fine."""
    if ratio is None or ratio >= HOLLOW_HULL_RATIO:
        return None
    return (
        f"This asset looks hollow (its volume is {ratio:.0%} of its convex hull), "
        "and the collision will be a solid block."
    )
