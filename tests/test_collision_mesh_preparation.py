"""Collision preparation must preserve cavities instead of filling a convex hull."""

import numpy as np
import pytest
import trimesh

from cyberwave_robot_format.mesh import decompose_collision_mesh


def test_decomposition_keeps_concave_grasp_space_open():
    pytest.importorskip("coacd")
    from scipy.spatial import ConvexHull

    outline = np.array([[0, 0], [3, 0], [3, 1], [1, 1], [1, 3], [0, 3]]) * 0.01
    vertices = np.vstack([np.column_stack([outline, np.full(6, z)]) for z in (0, 0.01)])
    cap = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 5], [3, 4, 5]])
    faces = [*cap[:, ::-1], *(cap + 6)]
    for a in range(6):
        b = (a + 1) % 6
        faces.extend([[a, b, b + 6], [a, b + 6, a + 6]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    original = mesh.vertices.copy()
    parts = decompose_collision_mesh(mesh, threshold_m=0.0002, max_hulls=12)
    assert 1 < len(parts) <= 12
    np.testing.assert_array_equal(mesh.vertices, original)
    np.testing.assert_allclose(np.vstack([p.vertices for p in parts]).min(axis=0), mesh.bounds[0], atol=0.0004)
    np.testing.assert_allclose(np.vstack([p.vertices for p in parts]).max(axis=0), mesh.bounds[1], atol=0.0004)

    def contains(points, point):
        planes = ConvexHull(points).equations
        return np.all(planes[:, :3] @ point + planes[:, 3] <= 1e-8)

    cavity = np.array([0.017, 0.017, 0.005])
    assert contains(mesh.vertices, cavity)  # A single MuJoCo hull fills this void.
    assert not any(contains(part.vertices, cavity) for part in parts)
    for solid in ([0.025, 0.005, 0.005], [0.005, 0.025, 0.005]):
        assert any(contains(part.vertices, solid) for part in parts)


@pytest.mark.parametrize(
    "options", [{"threshold_m": 0}, {"threshold_m": float("nan")}, {"max_hulls": 0}, {"max_hulls": True}]
)
def test_decomposition_rejects_invalid_calibration(options):
    with pytest.raises(ValueError):
        decompose_collision_mesh(trimesh.creation.box(), **options)
