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
    fit_collision_boxes,
    hollow_collision_warning,
    hull_volume_ratio,
)
from cyberwave_robot_format.usd import scene_to_glb
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


def _mesh(stage, pxr, name, shape):
    """``shape`` (a trimesh, in stage units and axes) as one Mesh prim."""
    mesh = pxr.UsdGeom.Mesh.Define(stage, f"/World/{name}")
    mesh.CreatePointsAttr([pxr.Gf.Vec3f(*map(float, p)) for p in shape.vertices])
    mesh.CreateFaceVertexCountsAttr([3] * len(shape.faces))
    mesh.CreateFaceVertexIndicesAttr([int(i) for i in shape.faces.ravel()])
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


def _y_up_cm(shape):
    """A Z-up, metre ``shape`` in the stage's Y-up centimetres."""
    shape.apply_transform(
        trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
    )
    return shape.apply_scale(100)


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

    def test_walls_in_one_mesh_leave_the_room_open(self, tmp_path, pxr):
        pytest.importorskip("scipy")
        stage, path = _stage(tmp_path, pxr)
        # Y-up centimetres: a 20 x 12 m ring of walls, 3 m high, as a single mesh.
        _mesh(stage, pxr, "walls", _y_up_cm(_wall_ring()))
        stage.Save()
        boxes = USDSceneToGLB().convert(path).report.collision

        # One bounding box would be the whole 20 x 12 x 3 m room, solid.
        assert len(boxes) == 4
        assert len({box["name"] for box in boxes}) == 4
        assert boxes[0]["name"].endswith("walls")
        for x, y in [(0, 0), (8, 4), (-8, -4), (9.5, 0), (0, -5.5)]:
            assert not _blocked(boxes, (x, y, 0.5)), (x, y)
        for x, y in [(9.9, 0), (-9.9, 3), (0, 5.9), (4, -5.9)]:
            assert _blocked(boxes, (x, y, 0.5)), (x, y)

    def test_box_names_never_clash_with_a_prim_name(self, tmp_path, pxr):
        pytest.importorskip("scipy")
        stage, path = _stage(tmp_path, pxr)
        _mesh(stage, pxr, "walls", _y_up_cm(_wall_ring()))
        _cube(stage, pxr, "walls_1", (200, 200, 200), (0, 100, 0))
        stage.Save()
        names = [box["name"] for box in USDSceneToGLB().convert(path).report.collision]
        # Duplicate geom names make MuJoCo refuse the whole scene.
        assert len(names) == 5 and len(set(names)) == 5

    def test_a_mesh_past_the_cap_keeps_its_bounding_box(
        self, tmp_path, pxr, monkeypatch
    ):
        pytest.importorskip("scipy")
        monkeypatch.setattr(scene_to_glb, "_MAX_COLLIDERS", 6)
        stage, path = _stage(tmp_path, pxr)
        _mesh(stage, pxr, "inner", _y_up_cm(_wall_ring()))
        _mesh(stage, pxr, "outer", _y_up_cm(_wall_ring().apply_scale(2)))
        stage.Save()
        report = USDSceneToGLB().convert(path).report

        # The outer ring's four boxes no longer fit; it still collides, as one box.
        names = [box["name"].split(".")[-1] for box in report.collision]
        assert names == ["inner", "inner-1", "inner-2", "inner-3", "outer"]
        assert report.collision[-1]["size"] == pytest.approx([40, 24, 6], abs=0.01)
        assert any(
            w.startswith("/World/outer:") and "bounding box" in w
            for w in report.warnings
        )

    def test_meshes_that_fill_the_cap_leave_a_box_for_a_later_prim(
        self, tmp_path, pxr, monkeypatch
    ):
        pytest.importorskip("scipy")
        monkeypatch.setattr(scene_to_glb, "_MAX_COLLIDERS", 8)
        stage, path = _stage(tmp_path, pxr)
        _mesh(stage, pxr, "inner", _y_up_cm(_wall_ring()))
        _mesh(stage, pxr, "outer", _y_up_cm(_wall_ring().apply_scale(2)))
        _cube(stage, pxr, "floor", (4000, 10, 2400), (0, -5, 0))
        stage.Save()
        report = USDSceneToGLB().convert(path).report

        # The two rings alone fill the cap; the floor after them must still collide.
        names = [box["name"].split(".")[-1] for box in report.collision]
        assert names == ["inner", "inner-1", "inner-2", "inner-3", "outer", "floor"]
        assert not any("left out" in w for w in report.warnings)

    def test_a_mesh_that_falls_back_to_a_block_says_so(self, tmp_path, pxr):
        pytest.importorskip("scipy")
        stage, path = _stage(tmp_path, pxr)
        _mesh(stage, pxr, "pillars", _y_up_cm(_pillars()))
        stage.Save()
        report = USDSceneToGLB().convert(path).report
        # Otherwise the import of the old solid block looks like a fit that worked.
        assert len(report.collision) == 1
        assert any(
            w.startswith("/World/pillars:") and "solid block" in w
            for w in report.warnings
        )

    def test_y_up_output_keeps_the_bounding_box(self, tmp_path, pxr):
        # The fitter takes axis 2 as up; a Y-up output would close passages sideways.
        stage, path = _stage(tmp_path, pxr)
        _mesh(stage, pxr, "walls", _y_up_cm(_wall_ring()))
        stage.Save()
        (box,) = USDSceneToGLB(z_up=False).convert(path).report.collision
        assert box["size"] == pytest.approx([20, 3, 12], abs=0.01)

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


def _box_mesh(extents, at=(0, 0, 0)):
    box = trimesh.creation.box(extents=extents)
    box.apply_translation(at)
    return box


def _wall_ring():
    """Four 0.2 m walls around a 20 x 12 m room, 3 m high, Z-up, as one mesh."""
    return trimesh.util.concatenate(
        [
            _box_mesh([20, 0.2, 3], (0, -5.9, 1.5)),
            _box_mesh([20, 0.2, 3], (0, 5.9, 1.5)),
            _box_mesh([0.2, 12, 3], (-9.9, 0, 1.5)),
            _box_mesh([0.2, 12, 3], (9.9, 0, 1.5)),
        ]
    )


def _open_ring(outward):
    """The ring's walls as four quads with no thickness and no roof, sharing corners."""
    floor = np.array([[-10, -6, 0], [10, -6, 0], [10, 6, 0], [-10, 6, 0]], float)
    vertices = np.concatenate([floor, floor + [0, 0, 3]])
    faces = []
    for i in range(4):
        j = (i + 1) % 4
        faces += [[i, j, j + 4], [i, j + 4, i + 4]]
    faces = np.array(faces)
    return vertices, faces if outward else faces[:, ::-1]


def _booth(at=(0, 0)):
    """A 3 x 3 m booth, 3 m high, with 0.3 m walls welded at the corners, no roof."""
    x, y = at
    return trimesh.util.concatenate(
        [
            _box_mesh([3, 0.3, 3], (x, y - 1.35, 1.5)),
            _box_mesh([3, 0.3, 3], (x, y + 1.35, 1.5)),
            _box_mesh([0.3, 2.4, 3], (x - 1.35, y, 1.5)),
            _box_mesh([0.3, 2.4, 3], (x + 1.35, y, 1.5)),
        ]
    )


def _pillars():
    """100 separate 0.5 m pillars, 5 m apart, as one mesh: too many pieces to split."""
    return trimesh.util.concatenate(
        [
            _box_mesh([0.5, 0.5, 3], (5 * i, 5 * j, 1.5))
            for i in range(10)
            for j in range(10)
        ]
    )


def _blocked(boxes, point):
    """Is ``point`` inside one of the ``{"size", "xyz", "rpy"}`` boxes?"""
    for box in boxes:
        yaw = box["rpy"][2]
        dx, dy, dz = np.subtract(point, box["xyz"])
        local = (
            dx * np.cos(yaw) + dy * np.sin(yaw),
            dy * np.cos(yaw) - dx * np.sin(yaw),
            dz,
        )
        if np.all(np.abs(local) <= np.divide(box["size"], 2) + 1e-6):
            return True
    return False


def _volume(boxes):
    return sum(float(np.prod(box["size"])) for box in boxes)


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


class TestFitCollisionBoxes:
    """One mesh -> the boxes that stand for it, without closing the space it leaves."""

    @pytest.fixture(autouse=True)
    def _needs_scipy(self):
        pytest.importorskip("scipy")

    @staticmethod
    def _fit(mesh, notes=None):
        return fit_collision_boxes(mesh.vertices, mesh.faces, notes=notes)

    def test_solid_box_is_its_own_box(self):
        (box,) = self._fit(_box_mesh([2, 1, 0.5], (3, -1, 0.25)))
        assert box["size"] == pytest.approx([2, 1, 0.5])
        assert box["xyz"] == pytest.approx([3, -1, 0.25])
        assert box["rpy"] == pytest.approx([0, 0, 0])

    def test_wall_at_an_angle_is_turned_not_enlarged(self):
        wall = _box_mesh([10, 0.2, 3], (0, 0, 1.5))
        wall.apply_transform(
            trimesh.transformations.rotation_matrix(np.radians(40), [0, 0, 1])
        )
        (box,) = self._fit(wall)
        # Its world-axis bounding box is 7.8 x 6.6 m of floor; the wall covers 2 m2.
        assert sorted(box["size"]) == pytest.approx([0.2, 3, 10], abs=1e-6)
        assert box["rpy"][2] % (np.pi / 2) == pytest.approx(np.radians(40))
        assert not _blocked([box], (2.5, -2.5, 1.0))
        assert _blocked([box], (3.0, 2.5, 1.0))

    def test_ring_of_walls_keeps_the_room_open(self):
        notes = []
        boxes = self._fit(_wall_ring(), notes)
        assert len(boxes) == 4 and notes == []
        assert not _blocked(boxes, (0, 0, 0.5))
        assert not _blocked(boxes, (9.4, 5.4, 0.5))  # half a metre from the corner
        # Every wall is still there, and nothing like the 720 m3 block it replaces.
        ring = _wall_ring()
        assert all(_blocked(boxes, v) for v in ring.vertices)
        assert _volume(boxes) < 50

    def test_ring_of_walls_at_an_angle_keeps_the_room_open(self):
        ring = _wall_ring()
        turn = trimesh.transformations.rotation_matrix(np.radians(30), [0, 0, 1])
        ring.apply_transform(turn)
        boxes = self._fit(ring)
        assert not _blocked(boxes, (0, 0, 0.5))
        centre_of_east_wall = turn[:3, :3] @ [9.9, 0, 0.5]
        assert _blocked(boxes, centre_of_east_wall)
        assert _volume(boxes) < 60

    def test_building_with_floor_and_roof_is_not_a_block(self):
        shell = trimesh.util.concatenate(
            [
                _wall_ring(),
                _box_mesh([20, 12, 0.2], (0, 0, -0.1)),
                _box_mesh([20, 12, 0.2], (0, 0, 3.1)),
            ]
        )
        boxes = self._fit(shell)
        assert not _blocked(boxes, (0, 0, 1.0))
        assert _blocked(boxes, (0, 0, -0.1))  # the floor still carries a robot
        assert _blocked(boxes, (0, 0, 3.1))
        assert _blocked(boxes, (9.9, 0, 1.0))

    def test_aisle_between_two_rack_rows_stays_open(self):
        parts = []
        for y in (-2, 2):
            for x in np.arange(-8, 8.1, 2):
                parts += [
                    _box_mesh([0.1, 0.1, 4], (x, y + dy, 2)) for dy in (-0.5, 0.5)
                ]
            for z in (1, 2, 3, 4):
                parts.append(_box_mesh([16, 1.0, 0.05], (0, y, z)))
        parts.append(_box_mesh([18, 6, 0.1], (0, 0, -0.05)))
        boxes = self._fit(trimesh.util.concatenate(parts))
        for x in (-7, 0, 7):
            assert not _blocked(boxes, (x, 0, 0.5)), x
        assert _blocked(boxes, (0, 2, 1.0))  # a shelf
        assert _blocked(boxes, (-8, 1.5, 0.5))  # an upright

    @pytest.mark.parametrize("outward", [True, False])
    def test_open_ring_keeps_the_room_open_whichever_way_it_faces(self, outward):
        # Facing out, its signed volume is two thirds of its box: not a full part.
        boxes = fit_collision_boxes(*_open_ring(outward))
        assert not _blocked(boxes, (0, 0, 1.0))
        assert not _blocked(boxes, (9.5, 5.5, 1.0))
        for point in [(0, -6, 1.5), (10, 0, 1.5), (-4, 6, 0.5), (-10, 3, 2.5)]:
            assert _blocked(boxes, point), point

    def test_racks_joined_at_one_end_keep_the_aisle_open(self):
        # Welded into one closed part that fills 62 % of its box.
        racks = trimesh.util.concatenate(
            [
                _box_mesh([1, 8, 2.5], (0.5, 4, 1.25)),
                _box_mesh([1, 8, 2.5], (3.0, 4, 1.25)),
                _box_mesh([1.5, 1, 2.5], (1.75, 7.5, 1.25)),
            ]
        )
        boxes = self._fit(racks)
        for y in (0.5, 3.5, 6.5):
            assert not _blocked(boxes, (1.75, y, 1.0)), y
        # The racks are solid inside, not hollow shells.
        for point in [(0.5, 4, 1.25), (3.0, 1, 2.0), (1.75, 7.5, 1.25)]:
            assert _blocked(boxes, point), point

    def test_wall_with_a_doorway_keeps_the_doorway_open(self):
        wall = trimesh.util.concatenate(
            [
                _box_mesh([3.5, 0.2, 3], (-3.25, 0, 1.5)),
                _box_mesh([3.5, 0.2, 3], (3.25, 0, 1.5)),
                _box_mesh([3, 0.2, 0.5], (0, 0, 2.75)),
            ]
        )
        boxes = self._fit(wall)
        for point in [(0, 0, 0.5), (-1.2, 0, 2.2), (1.2, 0, 1.0)]:
            assert not _blocked(boxes, point), point
        for point in [(0, 0, 2.75), (-3, 0, 1.0), (4.5, 0, 2.9)]:
            assert _blocked(boxes, point), point

    def test_booth_with_thick_walls_keeps_its_inside_open(self):
        boxes = self._fit(_booth())
        assert not _blocked(boxes, (0, 0, 1.0))
        assert not _blocked(boxes, (1.1, -1.1, 2.5))
        for point in [(0, 1.35, 1.5), (-1.35, 0, 0.5), (1.4, 1.4, 2.9)]:
            assert _blocked(boxes, point), point

    def test_round_tank_is_one_box(self):
        # Not a box, but no robot fits between it and its box's sides.
        tank = trimesh.creation.cylinder(radius=1, height=2, sections=64)
        (box,) = self._fit(tank)
        assert box["size"] == pytest.approx([2, 2, 2], abs=1e-3)

    def test_a_mesh_of_many_hollow_parts_stays_within_its_limit(self):
        booths = [_booth((4 * i, 4 * j)) for i in range(5) for j in range(4)]
        notes = []
        boxes = self._fit(trimesh.util.concatenate(booths), notes)
        # Each booth alone takes four boxes: 80 for the mesh, over its 64. Six become
        # one box each, and the 1 m corridors between the booths stay open.
        assert len(boxes) == 62
        assert notes == [
            "6 of its pieces collide as solid blocks, to keep it within 64 collision "
            "boxes"
        ]
        inside = [(4 * i, 4 * j, 1.0) for i in range(5) for j in range(4)]
        assert sum(not _blocked(boxes, point) for point in inside) == 14
        for x, y in [(2, 0), (2, 6), (14, 10), (0, 2), (8, 6), (16, 10)]:
            assert not _blocked(boxes, (x, y, 1.0)), (x, y)

    def test_a_mesh_too_intricate_for_boxes_says_it_is_a_block(self):
        # Over 64 pieces, so fitted as one part, which needs a box per pillar.
        notes = []
        (box,) = self._fit(_pillars(), notes)
        assert box["size"] == pytest.approx([45.5, 45.5, 3])
        assert notes == [
            "it is too intricate for collision boxes and collides as a solid block"
        ]

    def test_a_coarse_grid_says_which_gaps_it_may_close(self):
        # 144 posts take 144 boxes up to a 0.8 m grid; at 1.6 m the 1.9 m aisles close.
        posts = trimesh.util.concatenate(
            [
                _box_mesh([0.1, 0.1, 2], (2 * i, 2 * j, 1))
                for i in range(12)
                for j in range(12)
            ]
        )
        notes = []
        boxes = self._fit(posts, notes)
        assert _blocked(boxes, (1, 1, 1))
        assert notes == [
            "its collision boxes were cut on a 1.6 m grid, so gaps narrower than "
            "about 3.2 m may be closed"
        ]

    def test_unused_vertices_are_ignored(self):
        box = _box_mesh([2, 1, 0.5])
        # Sorted first when welded, so it would be a piece of its own with no faces.
        vertices = np.concatenate([[[-50, -50, -50]], box.vertices])
        (fitted,) = fit_collision_boxes(vertices, box.faces + 1)
        assert fitted["size"] == pytest.approx([2, 1, 0.5])

    def test_sheet_without_thickness_is_a_thin_box(self):
        floor = trimesh.Trimesh(
            [[0, 0, 0], [30, 0, 0], [30, 20, 0], [0, 20, 0]], [[0, 1, 2], [0, 2, 3]]
        )
        (box,) = self._fit(floor)
        assert box["size"][:2] == pytest.approx([30, 20])
        assert 0 < box["size"][2] <= 0.011

    def test_nothing_to_fit(self):
        assert fit_collision_boxes(np.zeros((0, 3)), np.zeros((0, 3))) == []
        assert (
            fit_collision_boxes([[0, 0, 0], [1, 0, np.nan], [0, 1, 0]], [[0, 1, 2]])
            == []
        )
