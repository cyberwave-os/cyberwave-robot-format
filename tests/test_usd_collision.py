# Copyright [2025] Cyberwave
# Licensed under the Apache License, Version 2.0

"""Box colliders and the hollow-mesh check for imported USD scenes.

The scene is a Y-up, centimetre room authored in code: a floor and ceiling (Cubes),
two walls (meshes) and a small prop. Everything is checked after the Y-up -> Z-up,
cm -> m conversion, because that is where a mismatch with the GLB would show.
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from cyberwave_robot_format.collision import (
    HOLLOW_HULL_RATIO,
    hollow_collision_warning,
    hull_volume_ratio,
)
from cyberwave_robot_format.usd.scene_to_glb import USDSceneToGLB

_CORNERS = np.array(
    [(x, y, z) for z in (-1, 1) for y in (-1, 1) for x in (-1, 1)], dtype=float
)
_QUADS = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2),
          (1, 3, 7, 5)]  # fmt: skip


@pytest.fixture(scope="module")
def pxr():
    pytest.importorskip("pxr")
    pytest.importorskip("PIL")
    from cyberwave_robot_format.usd import common

    return common.require_pxr()


def _stage(tmp_path, pxr):
    path = tmp_path / "room.usda"
    stage = pxr.Usd.Stage.CreateNew(str(path))
    pxr.UsdGeom.SetStageUpAxis(stage, "Y")
    pxr.UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    pxr.UsdGeom.Xform.Define(stage, "/World")
    return stage, path


def _cube(stage, pxr, name, size, at):
    cube = pxr.UsdGeom.Cube.Define(stage, f"/World/{name}")
    cube.CreateSizeAttr(1.0)
    api = pxr.UsdGeom.XformCommonAPI(cube)
    api.SetScale(tuple(size))
    api.SetTranslate(tuple(at))
    return cube


def _mesh_box(stage, pxr, name, size, at):
    """A closed box mesh whose points already sit at ``at`` (no xform)."""
    mesh = pxr.UsdGeom.Mesh.Define(stage, f"/World/{name}")
    points = _CORNERS * (np.asarray(size) / 2) + np.asarray(at)
    mesh.CreatePointsAttr([pxr.Gf.Vec3f(*p) for p in points])
    mesh.CreateFaceVertexCountsAttr([4] * 6)
    mesh.CreateFaceVertexIndicesAttr([i for quad in _QUADS for i in quad])
    return mesh


def _room(tmp_path, pxr):
    stage, path = _stage(tmp_path, pxr)
    _cube(stage, pxr, "floor", (260, 10, 490), (0, -5, 0))
    _cube(stage, pxr, "ceiling", (260, 10, 490), (0, 244, 0))
    _mesh_box(stage, pxr, "left_wall", (5, 244, 488), (-125, 122, 0))
    _mesh_box(stage, pxr, "right_wall", (5, 244, 488), (125, 122, 0))
    _mesh_box(stage, pxr, "prop", (10, 20, 10), (0, 30, 0))
    stage.Save()
    return path


def _by_name(report):
    return {box["name"].split(".")[-1]: box for box in report.collision}


class TestColliders:
    def test_room_gives_four_boxes_in_metres_and_z_up(self, tmp_path, pxr):
        report = USDSceneToGLB().convert(_room(tmp_path, pxr)).report
        boxes = _by_name(report)
        assert set(boxes) == {"floor", "ceiling", "left_wall", "right_wall"}

        # cm -> m, and USD's (x, y, z) becomes (x, -z, y): the 490 cm depth ends up
        # along Y, the 10 cm thickness of the floor along Z.
        floor = boxes["floor"]
        assert floor["size"] == pytest.approx([2.6, 4.9, 0.1], abs=1e-3)
        assert floor["xyz"] == pytest.approx([0, 0, -0.05], abs=1e-3)
        assert floor["rpy"] == pytest.approx([0, 0, 0], abs=1e-6)
        assert boxes["ceiling"]["xyz"] == pytest.approx([0, 0, 2.44], abs=1e-3)

        left = boxes["left_wall"]
        assert left["size"] == pytest.approx([0.05, 4.88, 2.44], abs=1e-3)
        assert left["xyz"] == pytest.approx([-1.25, 0, 1.22], abs=1e-3)
        assert boxes["right_wall"]["xyz"] == pytest.approx([1.25, 0, 1.22], abs=1e-3)

    def test_boxes_enclose_the_geometry_they_stand_for(self, tmp_path, pxr):
        result = USDSceneToGLB().convert(_room(tmp_path, pxr))
        lo = np.min([np.subtract(b["xyz"], np.divide(b["size"], 2))
                     for b in result.report.collision], axis=0)  # fmt: skip
        hi = np.max([np.add(b["xyz"], np.divide(b["size"], 2))
                     for b in result.report.collision], axis=0)  # fmt: skip
        import io

        bounds = trimesh.load(io.BytesIO(result.glb), file_type="glb").bounds
        assert lo == pytest.approx(bounds[0], abs=1e-3)
        assert hi == pytest.approx(bounds[1], abs=1e-3)

    def test_prop_below_the_threshold_is_skipped(self, tmp_path, pxr):
        report = USDSceneToGLB().convert(_room(tmp_path, pxr)).report
        assert "prop" not in _by_name(report)
        # A lower threshold picks it up: the skip is the threshold, not the prim.
        (tmp_path / "low").mkdir()  # USD refuses to create the same layer twice
        low = USDSceneToGLB(min_collider_size_m=0.1).convert(
            _room(tmp_path / "low", pxr)
        )
        assert "prop" in _by_name(low.report)

    def test_collision_api_forces_a_box_for_a_small_mesh(self, tmp_path, pxr):
        stage, path = _stage(tmp_path, pxr)
        mesh = _mesh_box(stage, pxr, "post", (10, 20, 10), (0, 10, 0))
        mesh.GetPrim().ApplyAPI("PhysicsCollisionAPI")
        stage.Save()
        report = USDSceneToGLB().convert(path).report
        assert list(_by_name(report)) == ["post"]

    def test_collision_disabled_is_the_opt_out(self, tmp_path, pxr):
        stage, path = _stage(tmp_path, pxr)
        _cube(stage, pxr, "floor", (260, 10, 490), (0, -5, 0))
        wall = _mesh_box(stage, pxr, "wall", (5, 244, 488), (125, 122, 0))
        wall.GetPrim().CreateAttribute(
            "physics:collisionEnabled", pxr.Sdf.ValueTypeNames.Bool
        ).Set(False)
        stage.Save()
        assert list(_by_name(USDSceneToGLB().convert(path).report)) == ["floor"]

    def test_rotated_cube_keeps_its_pose(self, tmp_path, pxr):
        stage, path = _stage(tmp_path, pxr)
        cube = _cube(stage, pxr, "ramp", (200, 10, 100), (0, 0, 0))
        pxr.UsdGeom.XformCommonAPI(cube).SetRotate((0, 0, 30))  # about USD Z
        stage.Save()
        box = _by_name(USDSceneToGLB().convert(path).report)["ramp"]
        assert box["size"] == pytest.approx([2.0, 0.1, 1.0], abs=1e-3)
        # USD Z (depth) is the output's -Y, so a +30 deg turn about it is -30 deg
        # about the output's Y axis... which is a pitch of -30 deg about +Y.
        assert abs(box["rpy"][1]) == pytest.approx(np.radians(30), abs=1e-3)

    def test_can_be_switched_off(self, tmp_path, pxr):
        result = USDSceneToGLB(generate_colliders=False).convert(_room(tmp_path, pxr))
        assert result.report.collision == []


class TestHollowWarning:
    @pytest.fixture(autouse=True)
    def _needs_scipy(self):
        pytest.importorskip("scipy")

    def test_a_room_without_boxes_warns_that_it_is_a_solid_block(self, tmp_path, pxr):
        report = (
            USDSceneToGLB(generate_colliders=False).convert(_room(tmp_path, pxr)).report
        )
        assert report.hull_ratio is not None and report.hull_ratio < HOLLOW_HULL_RATIO
        assert any("solid block" in w for w in report.warnings)

    def test_boxes_make_the_warning_unnecessary(self, tmp_path, pxr):
        report = USDSceneToGLB().convert(_room(tmp_path, pxr)).report
        assert not any("solid block" in w for w in report.warnings)

    def test_report_serialises_the_new_fields(self, tmp_path, pxr):
        data = USDSceneToGLB().convert(_room(tmp_path, pxr)).report.as_dict()
        assert len(data["collision"]) == 4 and "hull_ratio" in data


def _box_mesh(extents):
    return trimesh.creation.box(extents=extents)


def _hollow_box():
    """A 2 m cube with a 1.8 m cavity: outer shell minus inner, as one closed mesh."""
    outer = _box_mesh([2, 2, 2])
    inner = _box_mesh([1.8, 1.8, 1.8])
    inner.invert()
    return trimesh.util.concatenate([outer, inner])


class TestHullRatio:
    @pytest.fixture(autouse=True)
    def _needs_scipy(self):
        pytest.importorskip("scipy")

    def test_solid_box_does_not_warn(self):
        box = _box_mesh([1, 2, 3])
        ratio = hull_volume_ratio(box.vertices, box.faces)
        assert ratio == pytest.approx(1.0, abs=1e-6)
        assert hollow_collision_warning(ratio) is None

    def test_hollow_box_warns(self):
        mesh = _hollow_box()
        ratio = hull_volume_ratio(mesh.vertices, mesh.faces)
        assert ratio == pytest.approx(1 - 0.9**3, abs=1e-6)
        assert "solid block" in hollow_collision_warning(ratio)

    def test_unknown_ratio_does_not_warn(self):
        assert hollow_collision_warning(None) is None
        assert hull_volume_ratio(np.zeros((3, 3)), np.zeros((0, 3), dtype=int)) is None
