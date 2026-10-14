# Copyright [2025] Cyberwave
# Licensed under the Apache License, Version 2.0

"""Tests for ``export_scene_usd`` — the multi-robot USD scene bundle.

The three fixtures stand in for the robot families the Isaac integration has to
carry (primitives only, so no mesh files are needed):

* a Kinova-like arm: fixed base, revolute joints with position drives, a
  prismatic gripper with a mimic finger, wrist colour + depth cameras;
* a Go2-like quadruped: floating base, 12 revolute joints each with a position
  *and* a torque actuator, a front camera, a 4D lidar and an IMU;
* a DJI-like quadrotor: a single body with four MJCF site motors (rotors), a
  first-person camera and a GNSS receiver (no v1 sensor kind).

Assertions go through the pxr API against the composed stage, not file text,
except for the determinism tests, which compare bytes on purpose.
"""

from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pxr")

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

from cyberwave_robot_format.schema import (
    Actuator,
    ActuatorType,
    Collision,
    CollisionConfig,
    CollisionGroup,
    CommonSchema,
    Geometry,
    GeometryType,
    Inertia,
    Joint,
    JointDynamics,
    JointLimits,
    JointType,
    Link,
    Metadata,
    MimicJoint,
    Physics,
    Pose,
    Quaternion,
    Sensor,
    Vector3,
    Visual,
)
from cyberwave_robot_format.usd import (
    SceneExportOptions,
    SceneInstance,
    StaticBoxFixture,
    USDParser,
    ViewportOptions,
    export_scene_usd,
)
from cyberwave_robot_format.usd import semantics as sem
from cyberwave_robot_format.usd import tokens as t
from cyberwave_robot_format.usd.scene import viewport_matrix

ARM_UUID = "11111111-1111-4111-8111-111111111111"
ARM2_UUID = "22222222-2222-4222-8222-222222222222"
DOG_UUID = "33333333-3333-4333-8333-333333333333"
DRONE_UUID = "44444444-4444-4444-8444-444444444444"


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def _body(name: str, mass: float, geometry: Geometry) -> Link:
    return Link(
        name=name,
        mass=mass,
        inertia=Inertia(ixx=0.01, iyy=0.01, izz=0.01),
        visuals=[Visual(geometry=geometry)],
        collisions=[Collision(geometry=geometry)],
    )


def _box(x: float, y: float, z: float) -> Geometry:
    return Geometry(type=GeometryType.BOX, size=Vector3(x, y, z))


def _cylinder(radius: float, length: float) -> Geometry:
    return Geometry(type=GeometryType.CYLINDER, radius=radius, length=length)


def arm_schema() -> CommonSchema:
    """Kinova-like: fixed base, 3 revolute joints, a mimic gripper, 2 cameras."""
    links = [_body("base_link", 1.5, _cylinder(0.05, 0.1))]
    joints = [
        Joint(
            name="world_to_base",
            type=JointType.FIXED,
            parent_link="world",
            child_link="base_link",
        )
    ]
    parent = "base_link"
    for index in range(1, 4):
        name = f"link_{index}"
        links.append(_body(name, 1.0, _cylinder(0.04, 0.2)))
        joints.append(
            Joint(
                name=f"joint_{index}",
                type=JointType.REVOLUTE,
                parent_link=parent,
                child_link=name,
                pose=Pose(position=Vector3(0.0, 0.0, 0.15)),
                axis=Vector3(0.0, 1.0, 0.0) if index % 2 == 0 else Vector3(0.0, 0.0, 1.0),
                limits=JointLimits(lower=-2.5, upper=2.5, effort=39.0, velocity=1.4),
                dynamics=JointDynamics(damping=0.1, armature=0.01),
                home_position=0.25 * index,
            )
        )
        parent = name
    links.append(_body("end_effector", 0.3, _box(0.06, 0.06, 0.04)))
    joints.append(
        Joint(
            name="ee_fixed",
            type=JointType.FIXED,
            parent_link="link_3",
            child_link="end_effector",
            pose=Pose(position=Vector3(0.0, 0.0, 0.2)),
        )
    )
    for side, sign in (("left", 1.0), ("right", -1.0)):
        links.append(_body(f"{side}_finger", 0.05, _box(0.01, 0.02, 0.05)))
        joints.append(
            Joint(
                name=f"{side}_finger_joint",
                type=JointType.PRISMATIC,
                parent_link="end_effector",
                child_link=f"{side}_finger",
                pose=Pose(position=Vector3(0.0, sign * 0.02, 0.05)),
                axis=Vector3(0.0, sign, 0.0),
                limits=JointLimits(lower=0.0, upper=0.04, effort=20.0, velocity=0.1),
                mimic=MimicJoint(joint="left_finger_joint") if side == "right" else None,
            )
        )
    actuators = [
        Actuator(
            name=f"joint_{i}_servo",
            joint=f"joint_{i}",
            type=ActuatorType.POSITION,
            kp=100.0,
            kd=10.0,
            force_range=(-39.0, 39.0),
        )
        for i in range(1, 4)
    ]
    actuators.append(
        Actuator(
            name="gripper",
            joint="left_finger_joint",
            type=ActuatorType.POSITION,
            kp=50.0,
            kd=1.0,
        )
    )
    sensors = [
        Sensor(
            name="wrist_camera",
            type="camera",
            parent_link="end_effector",
            pose=Pose(position=Vector3(0.0, 0.05, 0.0)),
            parameters={"width": 640, "height": 480, "fovy": 60.0, "fps": 30},
        ),
        Sensor(
            name="wrist_depth",
            type="depth_camera",
            parent_link="end_effector",
            parameters={"width": 480, "height": 270, "fx": 336.0, "min_depth": 0.1},
        ),
    ]
    return CommonSchema(
        metadata=Metadata(name="Kinova Gen3 Lite"),
        links=links,
        joints=joints,
        actuators=actuators,
        sensors=sensors,
    )


def quadruped_schema() -> CommonSchema:
    """Go2-like: floating base, 12 joints each with position + torque actuators."""
    links = [_body("base", 6.9, _box(0.38, 0.09, 0.11))]
    joints = [
        Joint(
            name="floating_base",
            type=JointType.FLOATING,
            parent_link="world",
            child_link="base",
            pose=Pose(position=Vector3(0.0, 0.0, 0.4)),
        )
    ]
    actuators = []
    for leg, (x, y) in {
        "FL": (0.19, 0.05),
        "FR": (0.19, -0.05),
        "RL": (-0.19, 0.05),
        "RR": (-0.19, -0.05),
    }.items():
        parent = "base"
        offsets = (Vector3(x, y, 0.0), Vector3(0.0, 0.0955 * np.sign(y), 0.0), Vector3(0.0, 0.0, -0.213))
        for part, axis, offset in zip(
            ("hip", "thigh", "calf"),
            (Vector3(1.0, 0.0, 0.0), Vector3(0.0, 1.0, 0.0), Vector3(0.0, 1.0, 0.0)),
            offsets,
        ):
            name = f"{leg}_{part}"
            links.append(_body(name, 0.6, _cylinder(0.02, 0.2)))
            joints.append(
                Joint(
                    name=f"{name}_joint",
                    type=JointType.REVOLUTE,
                    parent_link=parent,
                    child_link=name,
                    pose=Pose(position=offset),
                    axis=axis,
                    limits=JointLimits(lower=-1.0, upper=1.0, effort=23.7, velocity=30.1),
                )
            )
            actuators.append(
                Actuator(
                    name=f"{name}_pos",
                    joint=f"{name}_joint",
                    type=ActuatorType.POSITION,
                    kp=20.0,
                    kd=0.5,
                )
            )
            actuators.append(
                Actuator(
                    name=f"{name}_torque",
                    joint=f"{name}_joint",
                    type=ActuatorType.TORQUE,
                    force_range=(-23.7, 23.7),
                )
            )
            parent = name
    sensors = [
        Sensor(
            name="front_camera",
            type="camera",
            parent_link="base",
            pose=Pose(position=Vector3(0.3, 0.0, 0.05)),
            parameters={"width": 1280, "height": 720, "hfov_deg": 120.0},
        ),
        Sensor(
            name="lidar",
            type="lidar_4d",
            parent_link="base",
            pose=Pose(position=Vector3(0.28, 0.0, -0.05)),
            update_rate=10.0,
            parameters={"hfov_deg": 360.0, "vfov_deg": 90.0, "max_range": 30.0},
        ),
        Sensor(name="imu", type="imu", parent_link="base", update_rate=500.0),
    ]
    return CommonSchema(
        metadata=Metadata(name="Unitree Go2"),
        links=links,
        joints=joints,
        actuators=actuators,
        sensors=sensors,
    )


ROTOR_POSITIONS = {
    "rotor_fl": (0.12, 0.12),
    "rotor_fr": (0.12, -0.12),
    "rotor_rl": (-0.12, 0.12),
    "rotor_rr": (-0.12, -0.12),
}


def quadrotor_schema() -> CommonSchema:
    """DJI-like: one body, four thrust site motors, an FPV camera and GNSS."""
    body = _body("body", 0.9, _box(0.3, 0.3, 0.08))
    body.extensions["mjcf_sites"] = [{"name": name, "pos": f"{x} {y} 0.04"} for name, (x, y) in ROTOR_POSITIONS.items()]
    spins = {"rotor_fl": 1.0, "rotor_fr": -1.0, "rotor_rl": -1.0, "rotor_rr": 1.0}
    actuators = [
        Actuator(
            name=f"thrust_{name.split('_')[1]}",
            joint="",
            type=ActuatorType.DC_MOTOR,
            control_range=(0.0, 1.0),
            extensions={
                "mjcf_site_transmission": {
                    "site": name,
                    "gear": f"0 0 13 0 0 {0.26 * spins[name]}",
                }
            },
        )
        for name in ROTOR_POSITIONS
    ]
    sensors = [
        Sensor(
            name="main_camera",
            type="camera",
            parent_link="body",
            pose=Pose(position=Vector3(0.16, 0.0, -0.03)),
            parameters={"width": 1920, "height": 1080, "fovy": 50.0},
        ),
        Sensor(name="gnss", type="gps", parent_link="body", update_rate=10.0),
    ]
    return CommonSchema(
        metadata=Metadata(name="DJI Mavic"),
        links=[body],
        joints=[],
        actuators=actuators,
        sensors=sensors,
    )


def arm_spec(instance_id: str = ARM_UUID, **kwargs) -> SceneInstance:
    kwargs.setdefault("spawn_pose", Pose(position=Vector3(1.0, 2.0, 0.0)))
    return SceneInstance(instance_id=instance_id, schema=arm_schema(), **kwargs)


def dog_spec(**kwargs) -> SceneInstance:
    return SceneInstance(
        instance_id=DOG_UUID,
        schema=quadruped_schema(),
        spawn_pose=Pose(position=Vector3(-1.0, 0.0, 0.0)),
        sensor_ids={"front_camera": "go2-front"},
        **kwargs,
    )


def drone_spec(**kwargs) -> SceneInstance:
    kwargs.setdefault("fixed_base", False)
    kwargs.setdefault("spawn_pose", Pose(position=Vector3(0.0, -2.0, 1.5)))
    kwargs.setdefault("first_person_camera", "main_camera")
    return SceneInstance(instance_id=DRONE_UUID, schema=quadrotor_schema(), **kwargs)


def options(**kwargs) -> SceneExportOptions:
    return SceneExportOptions(**kwargs)


@pytest.fixture
def scene(tmp_path):
    """The three-robot scene, exported once, with its composed stage."""
    result = export_scene_usd([drone_spec(), arm_spec(), dog_spec()], tmp_path, options=options())
    stage = Usd.Stage.Open(str(tmp_path / result.root_layer))
    assert stage is not None
    return result, stage, tmp_path


SCENE = "/World/Scene"


def instance_path(instance_id: str) -> str:
    return f"{SCENE}/instance_{instance_id.replace('-', '')}"


def by(entries, **match):
    found = [e for e in entries if all(e.get(k) == v for k, v in match.items())]
    assert len(found) == 1, (match, entries)
    return found[0]


def world_translation(stage: Usd.Stage, path: str) -> Gf.Vec3d:
    prim = stage.GetPrimAtPath(path)
    assert prim, path
    return UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()


# --------------------------------------------------------------------------
# Bundle layout and stage metadata
# --------------------------------------------------------------------------


class TestBundleLayout:
    def test_files_are_listed_sorted_with_hashes_and_sizes(self, scene):
        result, _, out = scene
        paths = [entry["path"] for entry in result.files]
        assert paths == sorted(paths)
        assert result.root_layer == "scene.usda" and "scene.usda" in paths
        assert len([p for p in paths if p.startswith("assets/")]) == 3
        for entry in result.files:
            data = (out / entry["path"]).read_bytes()
            assert entry["size"] == len(data)
            assert len(entry["sha256"]) == 64

    def test_stage_units_and_axis(self, scene):
        result, stage, _ = scene
        assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
        assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
        assert UsdPhysics.GetStageKilogramsPerUnit(stage) == 1.0
        assert result.units == {"meters_per_unit": 1.0, "kilograms_per_unit": 1.0, "up_axis": "Z"}
        assert stage.GetDefaultPrim().GetPath() == "/World"

    def test_physics_scene_and_schema_version(self, scene):
        result, stage, _ = scene
        physics = UsdPhysics.Scene(stage.GetPrimAtPath("/World/PhysicsScene"))
        assert physics
        assert physics.GetGravityDirectionAttr().Get() == Gf.Vec3f(0.0, 0.0, -1.0)
        assert physics.GetGravityMagnitudeAttr().Get() == pytest.approx(9.81)
        assert result.physics_scene["usd_path"] == "/World/PhysicsScene"
        world = stage.GetPrimAtPath("/World")
        assert world.GetCustomData()["cyberwave"]["usdSchemaVersion"] == t.USD_SCHEMA_VERSION
        assert result.exporter["usd_schema_version"] == t.USD_SCHEMA_VERSION
        # The asset layers' own PhysicsScene is outside their defaultPrim, so the
        # composed stage has exactly one.
        scenes = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
        assert [str(p.GetPath()) for p in scenes] == ["/World/PhysicsScene"]

    def test_instance_prim_records_its_base_and_nothing_else(self, scene):
        _, stage, _ = scene
        # What the root layer itself authors; the rest is the referenced asset's.
        layer = stage.GetRootLayer()
        assert dict(layer.GetPrimAtPath(SCENE).customData) == {}
        assert dict(layer.GetPrimAtPath(instance_path(DOG_UUID)).customData) == {"cyberwave": {"base": "floating"}}

    def test_caller_custom_data_is_written_verbatim(self, tmp_path):
        result = export_scene_usd(
            dog_spec(custom_data={"acme:robotId": "go2-7", "acme:serial": None}),
            tmp_path,
            options=options(
                scene_prim_name="lab-1",
                scene_name="Lab 1",
                scene_custom_data={"acme:siteId": "lab-1"},
            ),
        )
        stage = Usd.Stage.Open(str(tmp_path / result.root_layer))
        layer = stage.GetRootLayer()
        assert dict(layer.GetPrimAtPath("/World/lab_1").customData) == {"acme": {"siteId": "lab-1"}}
        assert stage.GetPrimAtPath("/World/lab_1").GetAttribute("cyberwave:name").Get() == "Lab 1"
        instance = layer.GetPrimAtPath(f"/World/lab_1/instance_{DOG_UUID.replace('-', '')}")
        # ``None`` is skipped, as for every customData value.
        assert dict(instance.customData) == {"acme": {"robotId": "go2-7"}, "cyberwave": {"base": "floating"}}
        assert result.identity_map["instances"][0]["usd_path"].startswith("/World/lab_1/")

    def test_namespace_overrides_the_instance_prim_name(self, tmp_path):
        result = export_scene_usd(dog_spec(namespace="go2-front"), tmp_path, options=options())
        assert result.identity_map["instances"] == [{"instance_id": DOG_UUID, "usd_path": f"{SCENE}/go2_front"}]

    def test_instance_prim_is_the_articulation_root(self, scene):
        result, stage, _ = scene
        for instance_id in (ARM_UUID, DOG_UUID, DRONE_UUID):
            prim = stage.GetPrimAtPath(instance_path(instance_id))
            assert prim.HasAPI(UsdPhysics.ArticulationRootAPI)
            assert by(result.articulations, instance_id=instance_id)["root_usd_path"] == instance_path(instance_id)

    def test_no_compatibility_errors_for_primitive_robots(self, scene):
        result, _, _ = scene
        assert result.ok, result.compatibility["errors"]

    def test_result_is_json_serializable(self, scene):
        result, _, _ = scene
        assert json.loads(json.dumps(result.to_dict())) == result.to_dict()

    def test_identity_paths_all_resolve_and_are_unique(self, scene):
        result, stage, _ = scene
        non_camera = []
        for section, entries in result.identity_map.items():
            for entry in entries:
                assert stage.GetPrimAtPath(entry["usd_path"]), (section, entry)
                if section != "cameras":
                    non_camera.append(entry["usd_path"])
        assert len(non_camera) == len(set(non_camera))
        sensor_ids = [e["sensor_id"] for e in result.identity_map["sensors"]]
        assert len(sensor_ids) == len(set(sensor_ids))


# --------------------------------------------------------------------------
# Bases and placement
# --------------------------------------------------------------------------


class TestBases:
    def test_fixed_base_gets_a_world_fixed_joint_at_the_spawn_pose(self, scene):
        result, stage, _ = scene
        articulation = by(result.articulations, instance_id=ARM_UUID)
        assert articulation["base"] == "fixed"
        anchor = stage.GetPrimAtPath(articulation["world_anchor_usd_path"])
        assert anchor.IsA(UsdPhysics.FixedJoint)
        assert str(anchor.GetPath()) == f"{instance_path(ARM_UUID)}/Joints/world_to_base"
        joint = UsdPhysics.FixedJoint(anchor)
        assert joint.GetBody0Rel().GetTargets() == []
        assert [str(p) for p in joint.GetBody1Rel().GetTargets()] == [f"{instance_path(ARM_UUID)}/Links/base_link"]
        assert joint.GetLocalPos0Attr().Get() == Gf.Vec3f(1.0, 2.0, 0.0)
        assert anchor.GetCustomData()["cyberwave"]["generated"] is True

    def test_floating_base_has_no_joint_to_the_world(self, scene):
        result, stage, _ = scene
        for instance_id in (DOG_UUID, DRONE_UUID):
            articulation = by(result.articulations, instance_id=instance_id)
            assert articulation["base"] == "floating"
            assert articulation["world_anchor_usd_path"] is None
            for prim in Usd.PrimRange(stage.GetPrimAtPath(instance_path(instance_id))):
                if prim.IsA(UsdPhysics.Joint):
                    assert UsdPhysics.Joint(prim).GetBody0Rel().GetTargets() != []

    def test_world_joint_offset_composes_with_the_spawn_pose(self, scene):
        result, stage, _ = scene
        # The Go2 world joint lifts the base 0.4 m; the spawn pose moves it -1 m in x.
        base = world_translation(stage, f"{instance_path(DOG_UUID)}/Links/base")
        assert tuple(base) == pytest.approx((-1.0, 0.0, 0.4))
        reset = by(result.articulations, instance_id=DOG_UUID)["reset"]
        assert reset["base_pose"]["position"] == pytest.approx([-1.0, 0.0, 0.4])
        assert reset["base_pose"]["orientation_wxyz"] == pytest.approx([1.0, 0.0, 0.0, 0.0])

    def test_link_world_poses_follow_the_instance_transform(self, scene):
        _, stage, _ = scene
        link = world_translation(stage, f"{instance_path(ARM_UUID)}/Links/link_2")
        assert tuple(link) == pytest.approx((1.0, 2.0, 0.30))

    def test_rotated_spawn_pose(self, tmp_path):
        quarter = Quaternion(x=0.0, y=0.0, z=math.sin(math.pi / 4), w=math.cos(math.pi / 4))
        result = export_scene_usd(
            arm_spec(spawn_pose=Pose(position=Vector3(0.0, 0.0, 0.5), orientation=quarter)),
            tmp_path,
            options=options(),
        )
        stage = Usd.Stage.Open(str(tmp_path / result.root_layer))
        anchor = UsdPhysics.FixedJoint(
            stage.GetPrimAtPath(by(result.articulations, instance_id=ARM_UUID)["world_anchor_usd_path"])
        )
        rot = anchor.GetLocalRot0Attr().Get()
        assert rot.GetReal() == pytest.approx(math.cos(math.pi / 4), abs=1e-6)
        assert rot.GetImaginary()[2] == pytest.approx(math.sin(math.pi / 4), abs=1e-6)
        # The camera sits +0.05 in y on the end effector; rotated 90 degrees about
        # z it ends up at -0.05 in x.
        camera = by(result.sensors, name="wrist_camera")["usd_path"]
        assert world_translation(stage, camera)[0] == pytest.approx(-0.05, abs=1e-6)

    def test_explicit_fixed_base_overrides_a_floating_world_joint(self, tmp_path):
        result = export_scene_usd(dog_spec(fixed_base=True), tmp_path, options=options())
        articulation = by(result.articulations, instance_id=DOG_UUID)
        assert articulation["base"] == "fixed"
        assert articulation["world_anchor_usd_path"].endswith("/Joints/floating_base")

    def test_schema_without_world_joint_defaults_to_fixed(self, tmp_path):
        result = export_scene_usd(
            SceneInstance(instance_id=DRONE_UUID, schema=quadrotor_schema()),
            tmp_path,
            options=options(),
        )
        articulation = by(result.articulations, instance_id=DRONE_UUID)
        assert articulation["base"] == "fixed"
        assert articulation["world_anchor_usd_path"].endswith("/Joints/world_anchor")

    def test_docked_instance_is_welded_to_the_parent_link(self, tmp_path):
        docked = drone_spec(
            attach_to_instance=DOG_UUID,
            attach_to_link="base",
            spawn_pose=Pose(position=Vector3(0.0, 0.0, 0.2)),
        )
        result = export_scene_usd([dog_spec(), docked], tmp_path, options=options())
        stage = Usd.Stage.Open(str(tmp_path / result.root_layer))
        articulation = by(result.articulations, instance_id=DRONE_UUID)
        assert articulation["base"] == "fixed"
        assert articulation["attached_to"] == {"instance_id": DOG_UUID, "link": "base"}
        joint = UsdPhysics.FixedJoint(stage.GetPrimAtPath(articulation["world_anchor_usd_path"]))
        assert [str(p) for p in joint.GetBody0Rel().GetTargets()] == [f"{instance_path(DOG_UUID)}/Links/base"]
        assert joint.GetLocalPos0Attr().Get() == Gf.Vec3f(0.0, 0.0, 0.2)
        body = world_translation(stage, f"{instance_path(DRONE_UUID)}/Links/body")
        assert tuple(body) == pytest.approx((-1.0, 0.0, 0.6))
        assert "twin_docked" in {w["code"] for w in result.compatibility["warnings"]}

    def test_docking_onto_an_instance_written_after_it(self, tmp_path):
        # Instances are written in id order; the arm (1111...) comes before the dog
        # (3333...) it is docked onto, so the dog's prim path must not depend on
        # the dog having been written yet.
        docked = arm_spec(attach_to_instance=DOG_UUID, attach_to_link="base")
        result = export_scene_usd([dog_spec(), docked], tmp_path, options=options())
        assert result.ok
        stage = Usd.Stage.Open(str(tmp_path / result.root_layer))
        articulation = by(result.articulations, instance_id=ARM_UUID)
        assert articulation["attached_to"] == {"instance_id": DOG_UUID, "link": "base"}
        joint = UsdPhysics.FixedJoint(stage.GetPrimAtPath(articulation["world_anchor_usd_path"]))
        assert [str(p) for p in joint.GetBody0Rel().GetTargets()] == [f"{instance_path(DOG_UUID)}/Links/base"]

    def test_dock_cycle_keeps_the_spawn_pose(self, tmp_path):
        arm = arm_spec(attach_to_instance=DOG_UUID, attach_to_link="base")
        dog = dog_spec(attach_to_instance=ARM_UUID, attach_to_link="base_link")
        result = export_scene_usd([arm, dog], tmp_path, options=options())
        errors = result.compatibility["errors"]
        assert [(e["code"], e["instance_id"]) for e in errors] == [("dock_cycle", ARM_UUID)]
        articulation = by(result.articulations, instance_id=ARM_UUID)
        assert "attached_to" not in articulation
        assert articulation["reset"]["base_pose"]["position"] == [1.0, 2.0, 0.0]
        stage = Usd.Stage.Open(str(tmp_path / result.root_layer))
        base = world_translation(stage, f"{instance_path(ARM_UUID)}/Links/base_link")
        assert tuple(base) == pytest.approx((1.0, 2.0, 0.0))
        # The other half of the cycle stays docked onto the arm, now well placed.
        assert by(result.articulations, instance_id=DOG_UUID)["attached_to"] == {
            "instance_id": ARM_UUID,
            "link": "base_link",
        }

    def test_docking_onto_a_missing_instance_is_an_error(self, tmp_path):
        result = export_scene_usd(drone_spec(attach_to_instance=ARM_UUID), tmp_path, options=options())
        assert [e["code"] for e in result.compatibility["errors"]] == ["dock_parent_missing"]
        assert by(result.articulations, instance_id=DRONE_UUID)["base"] == "fixed"

    def test_kept_world_joint_is_reanchored_in_world_frame(self, tmp_path):
        schema = arm_schema()
        schema.joints[0] = Joint(
            name="world_yaw",
            type=JointType.REVOLUTE,
            parent_link="world",
            child_link="base_link",
            pose=Pose(position=Vector3(0.0, 0.0, 0.1)),
            limits=JointLimits(lower=-1.0, upper=1.0),
        )
        spec = SceneInstance(instance_id=ARM_UUID, schema=schema, spawn_pose=Pose(position=Vector3(3.0, 0.0, 0.0)))
        result = export_scene_usd(spec, tmp_path, options=options())
        stage = Usd.Stage.Open(str(tmp_path / result.root_layer))
        joint = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(f"{instance_path(ARM_UUID)}/Joints/world_yaw"))
        assert joint.GetLocalPos0Attr().Get() == Gf.Vec3f(3.0, 0.0, 0.1)
        articulation = by(result.articulations, instance_id=ARM_UUID)
        assert articulation["world_anchor_usd_path"].endswith("/Joints/world_yaw")
        # No generated anchor in addition to the kept joint.
        assert not stage.GetPrimAtPath(f"{instance_path(ARM_UUID)}/Joints/world_anchor")


# --------------------------------------------------------------------------
# Joints and drives
# --------------------------------------------------------------------------


class TestJointsAndDrives:
    def test_joint_identity_has_dof_indices_in_schema_order(self, scene):
        result, _, _ = scene
        arm_joints = [e for e in result.identity_map["joints"] if e["instance_id"] == ARM_UUID]
        assert [(e["joint"], e["dof_index"]) for e in arm_joints] == [
            ("joint_1", 0),
            ("joint_2", 1),
            ("joint_3", 2),
            ("left_finger_joint", 3),
            ("right_finger_joint", 4),
        ]
        dog_joints = [e for e in result.identity_map["joints"] if e["instance_id"] == DOG_UUID]
        assert len(dog_joints) == 12

    def test_angular_drive_is_authored_per_degree(self, scene):
        _, stage, _ = scene
        prim = stage.GetPrimAtPath(f"{instance_path(ARM_UUID)}/Joints/joint_1")
        drive = UsdPhysics.DriveAPI(prim, "angular")
        assert drive.GetStiffnessAttr().Get() == pytest.approx(100.0 * math.pi / 180.0, rel=1e-6)
        assert drive.GetDampingAttr().Get() == pytest.approx(10.0 * math.pi / 180.0, rel=1e-6)
        assert drive.GetMaxForceAttr().Get() == pytest.approx(39.0)
        assert drive.GetTargetPositionAttr().Get() == pytest.approx(math.degrees(0.25), rel=1e-5)
        data = prim.GetCustomData()["cyberwave"]
        assert data["controlMode"] == "position"
        assert data["dofIndex"] == 0

    def test_limits_are_authored_in_degrees(self, scene):
        _, stage, _ = scene
        joint = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(f"{instance_path(ARM_UUID)}/Joints/joint_2"))
        assert joint.GetLowerLimitAttr().Get() == pytest.approx(math.degrees(-2.5), rel=1e-5)
        assert joint.GetUpperLimitAttr().Get() == pytest.approx(math.degrees(2.5), rel=1e-5)

    def test_articulation_drives_are_si_per_radian(self, scene):
        result, _, _ = scene
        drive = by(by(result.articulations, instance_id=ARM_UUID)["drives"], joint="joint_1")
        assert drive == {
            "joint": "joint_1",
            "usd_path": f"{instance_path(ARM_UUID)}/Joints/joint_1",
            "dof_index": 0,
            "joint_type": "revolute",
            "actuator": "joint_1_servo",
            "mode": "position",
            "stiffness": 100.0,
            "damping": 10.0,
            "max_force": 39.0,
            "max_velocity": 1.4,
            "target_position": 0.25,
            "lower": -2.5,
            "upper": 2.5,
            "joint_damping": 0.1,
            "joint_friction": 0.0,
            "armature": 0.01,
        }

    def test_mimic_is_declared_on_the_drive(self, scene):
        result, _, _ = scene
        drive = by(by(result.articulations, instance_id=ARM_UUID)["drives"], joint="right_finger_joint")
        assert drive["mimic"] == {"joint": "left_finger_joint", "multiplier": 1.0, "offset": 0.0}
        assert drive["mode"] == "none"
        codes = {w["code"] for w in result.compatibility["warnings"]}
        assert "mimic_joint_not_native" not in codes

    def test_mimic_is_a_native_newton_mimic_on_the_composed_stage(self, scene):
        _, stage, _ = scene
        joints = f"{instance_path(ARM_UUID)}/Joints"
        follower = stage.GetPrimAtPath(f"{joints}/right_finger_joint")
        assert "NewtonMimicAPI" in follower.GetMetadata("apiSchemas").GetAddedOrExplicitItems()
        # The asset-layer target is remapped through the instance's reference.
        assert follower.GetRelationship("newton:mimicJoint").GetTargets() == [
            Sdf.Path(f"{joints}/left_finger_joint")
        ]
        assert follower.GetAttribute("newton:mimicCoef1").Get() == 1.0
        assert follower.GetAttribute("newton:mimicCoef0").Get() == 0.0
        assert follower.GetAttribute("newton:mimicEnabled").Get() is True

    def test_mimic_isaac_cannot_simulate_is_reported(self, tmp_path):
        schema = arm_schema()
        follower = next(j for j in schema.joints if j.mimic is not None)
        follower.mimic.joint = "missing_joint"
        result = export_scene_usd(
            [SceneInstance(instance_id=ARM_UUID, schema=schema)],
            tmp_path,
            options=options(),
        )
        warning = by(result.compatibility["warnings"], code="mimic_joint_not_native")
        assert warning["subject"] == follower.name
        assert "missing_joint" in warning["message"]

    def test_physx_passthrough_reaches_the_physics_scene(self, tmp_path):
        physics = Physics(
            extensions={
                "physx": {
                    "physxScene:enableStabilization": True,
                    "physxScene:solverType": "PGS",
                    # Owned by the Isaac runtime profile, so refused.
                    "physxScene:timeStepsPerSecond": 240,
                }
            }
        )
        result = export_scene_usd(
            [arm_spec()], tmp_path, options=options(physics=physics)
        )
        stage = Usd.Stage.Open(str(tmp_path / result.root_layer))
        scene = stage.GetPrimAtPath(result.physics_scene["usd_path"])
        assert "PhysxSceneAPI" in scene.GetMetadata("apiSchemas").GetAddedOrExplicitItems()
        assert scene.GetAttribute("physxScene:enableStabilization").Get() is True
        assert scene.GetAttribute("physxScene:solverType").Get() == "PGS"
        assert not scene.GetAttribute("physxScene:timeStepsPerSecond")
        skipped = by(result.compatibility["warnings"], code="physx_attribute_skipped")
        assert "timeStepsPerSecond" in skipped["message"]

    def test_quadruped_primary_actuator_owns_the_drive(self, scene):
        result, _stage, _ = scene
        drives = by(result.articulations, instance_id=DOG_UUID)["drives"]
        assert {d["mode"] for d in drives} == {"position"}
        assert {d["actuator"] for d in drives} == {f"{j['joint'][:-6]}_pos" for j in drives}
        warning = by(result.compatibility["warnings"], code="secondary_actuator_not_driven")
        assert warning["instance_id"] == DOG_UUID
        assert len(warning["subject"].split(",")) == 12

    def test_reset_joint_positions_are_home_positions(self, scene):
        result, _, _ = scene
        reset = by(result.articulations, instance_id=ARM_UUID)["reset"]
        assert reset["joint_positions"]["joint_3"] == pytest.approx(0.75)
        assert reset["joint_positions"]["left_finger_joint"] == 0.0


# --------------------------------------------------------------------------
# Sensors, cameras, rotors
# --------------------------------------------------------------------------


class TestSensors:
    def test_camera_sensor_is_a_usd_camera_with_intrinsics(self, scene):
        result, stage, _ = scene
        entry = by(result.sensors, name="wrist_camera")
        assert entry["sensor_id"] == f"{ARM_UUID.replace('-', '')}__wrist_camera"
        assert entry["kind"] == "rgb" and entry["fidelity"] == "rendered"
        params = entry["params"]
        assert (params["width"], params["height"]) == (640, 480)
        assert params["vfov_rad"] == pytest.approx(math.radians(60.0))
        assert params["fy"] == pytest.approx(240.0 / math.tan(math.radians(30.0)))
        assert params["update_rate_hz"] == 30.0
        camera = UsdGeom.Camera(stage.GetPrimAtPath(entry["usd_path"]))
        assert camera
        focal = camera.GetFocalLengthAttr().Get()
        vertical = camera.GetVerticalApertureAttr().Get()
        assert 2.0 * math.atan(vertical / (2.0 * focal)) == pytest.approx(math.radians(60.0), rel=1e-5)
        data = camera.GetPrim().GetCustomData()["cyberwave"]
        assert data["sensorId"] == entry["sensor_id"]
        assert data["cameraRole"] == "sensor"
        camera_entry = by(result.identity_map["cameras"], camera_id=entry["sensor_id"])
        assert camera_entry["role"] == "sensor"
        assert camera_entry["usd_path"] == entry["usd_path"]

    def test_depth_camera_keeps_explicit_focal_length(self, scene):
        result, _, _ = scene
        params = by(result.sensors, name="wrist_depth")["params"]
        assert params["fx"] == params["fy"] == 336.0
        assert params["near_clip"] == pytest.approx(0.01)
        assert params["min_depth"] == pytest.approx(0.1)

    def test_explicit_sensor_ids_are_used(self, scene):
        result, stage, _ = scene
        entry = by(result.sensors, sensor_id="go2-front")
        assert entry["name"] == "front_camera"
        prim = stage.GetPrimAtPath(entry["usd_path"])
        assert prim.GetCustomData()["cyberwave"]["sensorId"] == "go2-front"

    def test_non_camera_sensors_are_xforms_with_params(self, scene):
        result, stage, _ = scene
        lidar = by(result.sensors, name="lidar")
        assert lidar["kind"] == "lidar" and lidar["fidelity"] == "geometric_ray"
        assert lidar["params"]["hfov_rad"] == pytest.approx(2 * math.pi)
        assert lidar["params"]["max_range"] == 30.0
        prim = stage.GetPrimAtPath(lidar["usd_path"])
        assert prim.IsA(UsdGeom.Xform) and not prim.IsA(UsdGeom.Camera)
        assert prim.GetCustomData()["cyberwave"]["sensorKind"] == "lidar"
        imu = by(result.sensors, name="imu")
        assert imu["kind"] == "imu" and imu["params"]["update_rate_hz"] == 500.0
        # Isaac sensor schemas are not authored: only core USD types.
        for entry in result.sensors:
            type_name = stage.GetPrimAtPath(entry["usd_path"]).GetTypeName()
            assert type_name in {"Camera", "Xform"}

    def test_first_person_camera_role(self, scene):
        result, stage, _ = scene
        entry = by(result.sensors, name="main_camera")
        assert entry["kind"] == "first_person_camera"
        assert by(result.identity_map["cameras"], camera_id=entry["sensor_id"])["role"] == "first_person"
        data = stage.GetPrimAtPath(entry["usd_path"]).GetCustomData()["cyberwave"]
        assert data["sensorKind"] == "first_person_camera"
        assert data["cameraRole"] == "first_person"

    def test_missing_first_person_camera_is_a_warning(self, tmp_path):
        result = export_scene_usd(drone_spec(first_person_camera="nope"), tmp_path, options=options())
        assert "first_person_camera_missing" in {w["code"] for w in result.compatibility["warnings"]}

    @pytest.mark.parametrize(
        ("extensions", "parameters", "thermal"),
        [
            ({"cyberwave": {"display_palette": "thermal"}}, {}, True),
            ({}, {"color_mode": "thermal"}, True),
            # display_palette wins over the legacy color_mode (cyberwave-sim rule).
            ({"cyberwave": {"display_palette": "rgb"}}, {"color_mode": "thermal"}, False),
            ({}, {}, False),
        ],
    )
    def test_thermal_camera_is_an_rgb_sensor_with_the_thermal_marker(self, tmp_path, extensions, parameters, thermal):
        schema = quadrotor_schema()
        camera = schema.sensors[0]
        camera.extensions = extensions
        camera.parameters = {
            **camera.parameters,
            **parameters,
            "thermal_width": 160,
            "thermal_height": 120,
            "min_temperature_c": -20.0,
            "max_temperature_c": 120.0,
        }
        spec = SceneInstance(instance_id=DRONE_UUID, schema=schema, fixed_base=False)
        result = export_scene_usd(spec, tmp_path, options=options())
        entry = by(result.sensors, name="main_camera")
        assert entry["kind"] == "rgb"
        params = entry["params"]
        assert (params["thermal_width"], params["thermal_height"]) == (160, 120)
        assert (params["min_temperature_c"], params["max_temperature_c"]) == (-20.0, 120.0)
        if thermal:
            assert entry["fidelity"] == "heuristic"
            assert params["display_palette"] == "thermal"
        else:
            assert entry["fidelity"] == "rendered"
            assert "display_palette" not in params
        assert list(params) == sorted(params)

    def test_thermal_first_person_camera_keeps_its_role(self, tmp_path):
        schema = quadrotor_schema()
        schema.sensors[0].extensions = {"cyberwave": {"display_palette": "thermal"}}
        spec = SceneInstance(
            instance_id=DRONE_UUID,
            schema=schema,
            fixed_base=False,
            first_person_camera="main_camera",
        )
        entry = by(export_scene_usd(spec, tmp_path, options=options()).sensors, name="main_camera")
        assert entry["kind"] == "first_person_camera"
        assert entry["fidelity"] == "heuristic" and entry["params"]["display_palette"] == "thermal"

    def test_is_thermal_sensor_matches_cyberwave_sim(self):
        def sensor(type_, extensions=None, parameters=None):
            return Sensor(
                name="s", type=type_, parent_link="body", extensions=extensions or {}, parameters=parameters or {}
            )

        palette = {"cyberwave": {"display_palette": "thermal"}}
        assert sem.is_thermal_sensor(sensor("rgb", palette))
        assert sem.is_thermal_sensor(sensor("camera", parameters={"color_mode": "thermal"}))
        # Only rgb/camera types qualify, exactly as cyberwave-sim compares them.
        assert not sem.is_thermal_sensor(sensor("depth", palette))
        assert not sem.is_thermal_sensor(sensor("RGB", palette))
        assert not sem.is_thermal_sensor(
            sensor("rgb", {"cyberwave": {"display_palette": None}}, {"color_mode": "thermal"})
        )

    def test_unsupported_sensor_kind_is_rejected_not_dropped(self, scene):
        result, stage, _ = scene
        rejected = by(result.compatibility["rejected_features"], code="sensor_kind_unsupported")
        assert rejected["subject"] == "gnss"
        assert not [s for s in result.sensors if s.get("name") == "gnss"]
        # The prim is still there with its raw attributes.
        prim = stage.GetPrimAtPath(f"{instance_path(DRONE_UUID)}/Links/body/Sensors/gnss")
        assert prim and prim.GetAttribute(t.A_SENSOR_TYPE).Get() == "gps"

    def test_rotors_are_frames_with_thrust_custom_data(self, scene):
        result, stage, _ = scene
        rotors = by(result.articulations, instance_id=DRONE_UUID)["rotors"]
        assert [r["actuator"] for r in rotors] == ["thrust_fl", "thrust_fr", "thrust_rl", "thrust_rr"]
        front_left = rotors[0]
        assert front_left["thrust_axis"] == [0.0, 0.0, 1.0]
        assert front_left["thrust_gain"] == 13.0
        assert front_left["max_thrust"] == 13.0
        assert front_left["torque_to_thrust"] == pytest.approx(0.02)
        assert rotors[1]["torque_to_thrust"] == pytest.approx(-0.02)
        prim = stage.GetPrimAtPath(front_left["usd_path"])
        assert str(prim.GetPath()) == f"{instance_path(DRONE_UUID)}/Links/body/Rotors/thrust_fl"
        assert tuple(world_translation(stage, front_left["usd_path"])) == pytest.approx((0.12, -1.88, 1.54))
        data = prim.GetCustomData()["cyberwave"]
        assert data["rotor"]["site"] == "rotor_fl"
        assert list(data["rotor"]["thrustAxis"]) == [0.0, 0.0, 1.0]

    def test_reaction_wheel_site_motor_is_rejected(self, tmp_path):
        schema = quadrotor_schema()
        schema.actuators[0].extensions["mjcf_site_transmission"]["gear"] = "0 0 0 0 0 1"
        result = export_scene_usd(
            SceneInstance(instance_id=DRONE_UUID, schema=schema), tmp_path, options=options()
        )
        assert by(result.compatibility["rejected_features"], code="site_motor_not_rotor")["subject"] == "thrust_fl"
        rotors = by(result.articulations, instance_id=DRONE_UUID)["rotors"]
        assert len(rotors) == 3


class TestSimViewport:
    def test_viewport_camera_matches_mujoco_default_view(self, scene):
        result, stage, _ = scene
        entry = by(result.sensors, sensor_id="sim_viewport")
        assert entry["instance_id"] is None and entry["kind"] == "sim_viewport"
        assert (entry["params"]["width"], entry["params"]["height"]) == (960, 540)
        assert entry["params"]["update_rate_hz"] == 15.0
        camera_entry = by(result.identity_map["cameras"], camera_id="sim_viewport")
        assert camera_entry["role"] == "sim_viewport" and camera_entry["instance_id"] is None
        prim = stage.GetPrimAtPath(entry["usd_path"])
        assert str(prim.GetPath()).endswith("/Cameras/sim_viewport")
        matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        eye = matrix.ExtractTranslation()
        assert tuple(eye) == pytest.approx((0.0, -5.638, 2.052), abs=1e-3)
        # The camera looks down its -Z at the origin, with +Y up-ish.
        forward = -Gf.Vec3d(matrix.TransformDir(Gf.Vec3d(0.0, 0.0, 1.0)))
        to_origin = (Gf.Vec3d(0.0, 0.0, 0.0) - eye).GetNormalized()
        assert Gf.Dot(forward.GetNormalized(), to_origin) == pytest.approx(1.0, abs=1e-9)
        up = matrix.TransformDir(Gf.Vec3d(0.0, 1.0, 0.0))
        assert up[2] > 0.9
        camera = UsdGeom.Camera(prim)
        vfov = 2.0 * math.atan(camera.GetVerticalApertureAttr().Get() / (2.0 * camera.GetFocalLengthAttr().Get()))
        assert vfov == pytest.approx(math.radians(45.0), rel=1e-5)

    def test_viewport_matrix_is_orthonormal(self):
        matrix = viewport_matrix(ViewportOptions(azimuth_deg=30.0, elevation_deg=-45.0))
        rotation = matrix[:3, :3]
        assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-9)
        assert np.linalg.det(rotation) == pytest.approx(1.0)

    def test_viewport_looking_straight_down(self):
        matrix = viewport_matrix(ViewportOptions(elevation_deg=-90.0))
        assert np.linalg.det(matrix[:3, :3]) == pytest.approx(1.0)
        assert matrix[:3, 3] == pytest.approx([0.0, 0.0, 6.0])

    def test_viewport_can_be_disabled(self, tmp_path):
        result = export_scene_usd(arm_spec(), tmp_path, options=options(viewport=ViewportOptions(enabled=False)))
        assert not [s for s in result.sensors if s["sensor_id"] == "sim_viewport"]
        assert not [c for c in result.identity_map["cameras"] if c["role"] == "sim_viewport"]


class TestSceneExtras:
    def test_generated_lights(self, scene):
        _, stage, _ = scene
        key = UsdLux.DistantLight(stage.GetPrimAtPath(f"{SCENE}/Lights/key_light"))
        dome = UsdLux.DomeLight(stage.GetPrimAtPath(f"{SCENE}/Lights/ambient_light"))
        assert key and dome
        assert key.GetIntensityAttr().Get() == 1000.0
        assert key.GetPrim().GetCustomData()["cyberwave"]["lightRole"] == "key"

    def test_lights_can_be_disabled(self, tmp_path):
        result = export_scene_usd(arm_spec(), tmp_path, options=options(lights=False))
        stage = Usd.Stage.Open(str(tmp_path / result.root_layer))
        assert not [p for p in stage.Traverse() if p.HasAPI(UsdLux.LightAPI)]

    def test_ground_plane_is_added_by_default(self, scene):
        _, stage, _ = scene
        plane = stage.GetPrimAtPath(f"{SCENE}/GroundPlane")
        assert plane.IsA(UsdGeom.Plane)
        assert plane.HasAPI(UsdPhysics.CollisionAPI)

    def test_ground_plane_is_skipped_when_an_instance_has_one(self, tmp_path):
        schema = arm_schema()
        schema.links[0].collisions.append(Collision(geometry=Geometry(type=GeometryType.PLANE)))
        result = export_scene_usd(SceneInstance(instance_id=ARM_UUID, schema=schema), tmp_path, options=options())
        stage = Usd.Stage.Open(str(tmp_path / result.root_layer))
        assert not stage.GetPrimAtPath(f"{SCENE}/GroundPlane")

    def test_static_fixtures_are_static_box_colliders(self, tmp_path):
        rail = StaticBoxFixture(
            name="procedural__rail-left",
            position=Vector3(0.0, 0.66, 0.05),
            size=Vector3(10.0, 0.05, 0.05),
            color=(0.96, 0.62, 0.04),
            mu_static=0.55,
            mu_dynamic=0.4,
            restitution=0.05,
            source_id="rail-left",
        )
        ghost = StaticBoxFixture(name="ghost", position=Vector3(1.0, 0.0, 0.5), size=Vector3(1.0, 1.0, 1.0), collision=False)
        result = export_scene_usd(arm_spec(), tmp_path, options=options(static_fixtures=[rail, ghost]))
        stage = Usd.Stage.Open(str(tmp_path / result.root_layer))
        fixtures = f"{SCENE}/Fixtures"
        box = stage.GetPrimAtPath(f"{fixtures}/procedural__rail_left/box")
        assert box.IsA(UsdGeom.Cube)
        assert box.HasAPI(UsdPhysics.CollisionAPI)
        assert not box.HasAPI(UsdPhysics.RigidBodyAPI)
        bound = UsdGeom.Imageable(box).ComputeWorldBound(Usd.TimeCode.Default(), UsdGeom.Tokens.default_).ComputeAlignedRange()
        assert tuple(bound.GetMin()) == pytest.approx((-5.0, 0.635, 0.025))
        assert tuple(bound.GetMax()) == pytest.approx((5.0, 0.685, 0.075))
        material, _ = UsdShade.MaterialBindingAPI(box).ComputeBoundMaterial("physics")
        assert UsdPhysics.MaterialAPI(material.GetPrim()).GetStaticFrictionAttr().Get() == pytest.approx(0.55)
        assert not stage.GetPrimAtPath(f"{fixtures}/ghost/box").HasAPI(UsdPhysics.CollisionAPI)
        assert result.ok
        assert {t["instance_id"] for t in result.identity_map["instances"]} == {ARM_UUID}

    def test_inert_collision_group_is_disabled_across_instances(self, tmp_path):
        # A group that collides with nothing must stay inert against *other* instances:
        # each asset layer authors its own CollisionGroups, so the asset-local
        # physics:filteredGroups alone would let PhysX collide it with them.
        track = arm_schema()
        track.links[0].collisions[0].group = "disabled"
        track.collision_config = CollisionConfig(
            groups={
                "default": CollisionGroup(name="default"),
                "disabled": CollisionGroup(name="disabled", contype=0, conaffinity=0),
            }
        )
        specs = [
            SceneInstance(instance_id=ARM_UUID, schema=track),
            arm_spec(ARM2_UUID),
        ]
        result = export_scene_usd(specs, tmp_path, options=options())
        stage = Usd.Stage.Open(str(tmp_path / result.root_layer))

        def colliders(instance_id):
            instance = next(t for t in result.identity_map["instances"] if t["instance_id"] == instance_id)
            return {
                str(prim.GetPath()): UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
                for prim in Usd.PrimRange(stage.GetPrimAtPath(instance["usd_path"]))
                if prim.HasAPI(UsdPhysics.CollisionAPI)
            }

        track_colliders = colliders(ARM_UUID)
        base = next(path for path in track_colliders if "/base_link/" in path)
        assert track_colliders.pop(base) is False
        assert track_colliders and all(track_colliders.values())
        assert all(colliders(ARM2_UUID).values())
        # The override lives in the scene layer; the asset layer stays round-trippable.
        asset_layer = next(f for f in result.files if f["path"].startswith("assets/"))
        assert "collisionEnabled" not in (tmp_path / asset_layer["path"]).read_text()

    def test_self_collisions_override_lives_on_the_articulation_root(self, tmp_path):
        specs = [dog_spec(self_collisions=False), arm_spec(self_collisions=True), drone_spec()]
        result = export_scene_usd(specs, tmp_path, options=options())
        stage = Usd.Stage.Open(str(tmp_path / result.root_layer))
        name = "physxArticulation:enabledSelfCollisions"

        def setting(instance_id):
            prim = stage.GetPrimAtPath(instance_path(instance_id))
            assert prim.HasAPI(UsdPhysics.ArticulationRootAPI)
            attribute = prim.GetAttribute(name)
            if not attribute.HasAuthoredValue():
                return None
            assert "PhysxArticulationAPI" in prim.GetPrimTypeInfo().GetAppliedAPISchemas()
            assert not attribute.IsCustom()
            return attribute.Get()

        assert setting(DOG_UUID) is False
        assert setting(ARM_UUID) is True
        assert setting(DRONE_UUID) is None  # unset: the asset's (PhysX default) stands
        # The override lives in the scene layer; the asset layers stay untouched.
        for entry in result.files:
            if entry["path"].startswith("assets/"):
                assert name not in (tmp_path / entry["path"]).read_text()

    def test_asset_authored_self_collisions_wins(self, tmp_path):
        dog = quadruped_schema()
        dog.extensions = {
            **(dog.extensions or {}),
            "physx": {"physxArticulation:enabledSelfCollisions": True},
        }
        spec = SceneInstance(instance_id=DOG_UUID, schema=dog, self_collisions=False)
        result = export_scene_usd([spec], tmp_path, options=options())
        stage = Usd.Stage.Open(str(tmp_path / result.root_layer))
        prim = stage.GetPrimAtPath(instance_path(DOG_UUID))
        assert prim.GetAttribute("physxArticulation:enabledSelfCollisions").Get() is True

    @staticmethod
    def _scenery_spec(z: float, group: str = "disabled") -> SceneInstance:
        """A static 2 x 1 x 0.4 m box instance, yawed 90 degrees, at ``(3, -1, z)``."""
        track = CommonSchema(
            metadata=Metadata(name="track"),
            links=[_body("track", 1.0, _box(2.0, 1.0, 0.4))],
            collision_config=CollisionConfig(
                groups={
                    "default": CollisionGroup(name="default"),
                    "disabled": CollisionGroup(name="disabled", contype=0, conaffinity=0),
                }
            ),
        )
        track.links[0].collisions[0].group = group
        yaw = math.radians(90.0) / 2.0
        return SceneInstance(
            instance_id=ARM2_UUID,
            schema=track,
            fixed_base=True,
            spawn_pose=Pose(
                position=Vector3(3.0, -1.0, z),
                orientation=Quaternion(w=math.cos(yaw), x=0.0, y=0.0, z=math.sin(yaw)),
            ),
        )

    def test_ground_visual_opens_over_buried_visual_only_scenery(self, tmp_path):
        result = export_scene_usd([arm_spec(), self._scenery_spec(-0.3)], tmp_path, options=options())
        stage = Usd.Stage.Open(str(tmp_path / result.root_layer))
        plane = stage.GetPrimAtPath(f"{SCENE}/GroundPlane")
        # The collider is unchanged, only hidden from rendering.
        assert plane.HasAPI(UsdPhysics.CollisionAPI)
        assert UsdGeom.Plane(plane).GetWidthAttr().Get() == 100.0
        assert UsdGeom.Imageable(plane).GetPurposeAttr().Get() == UsdGeom.Tokens.guide

        visual = list(stage.GetPrimAtPath(f"{SCENE}/GroundVisual").GetChildren())
        tiles = [p for p in visual if p.GetName().startswith("tile_")]
        pits = [p for p in visual if p.GetName().startswith("pit_")]
        assert len(tiles) == 4 and len(pits) == 1
        assert all(p.IsA(UsdGeom.Plane) and not p.HasAPI(UsdPhysics.CollisionAPI) for p in visual)
        area = sum(UsdGeom.Plane(p).GetWidthAttr().Get() * UsdGeom.Plane(p).GetLengthAttr().Get() for p in tiles)
        assert area == pytest.approx(100.0 * 100.0 - 2.0 * 1.0)
        # The footprint follows the instance's yaw: 1 m along X, 2 m along Y.
        bound = (
            UsdGeom.Imageable(pits[0])
            .ComputeWorldBound(Usd.TimeCode.Default(), UsdGeom.Tokens.default_)
            .ComputeAlignedRange()
        )
        assert tuple(bound.GetMin()) == pytest.approx((2.5, -2.0, -0.505))
        assert tuple(bound.GetMax()) == pytest.approx((3.5, 0.0, -0.505))

    def test_ground_visual_layout_tiles_around_overlapping_and_clipped_pits(self):
        from cyberwave_robot_format.usd.ground import GroundRect, ground_visual_layout

        pits = [
            GroundRect(-1.0, -1.0, 1.0, 1.0, z=-0.2),
            GroundRect(0.5, 0.0, 2.0, 3.0, z=-0.4),
            GroundRect(49.0, 0.0, 60.0, 1.0),
        ]
        tiles, floors = ground_visual_layout(pits, half_extent=50.0)
        assert floors[2] == GroundRect(49.0, 0.0, 50.0, 1.0)
        assert [f.z for f in floors] == [-0.2, -0.4, 0.0]
        holes = 2.0 * 2.0 + 1.5 * 3.0 - 0.5 * 1.0 + 1.0
        assert sum(r.size[0] * r.size[1] for r in tiles) == pytest.approx(100.0 * 100.0 - holes)
        for tile in tiles:
            for pit in floors:
                overlap_x = min(tile.max_x, pit.max_x) - max(tile.min_x, pit.min_x)
                overlap_y = min(tile.max_y, pit.max_y) - max(tile.min_y, pit.min_y)
                assert overlap_x <= 0 or overlap_y <= 0

    def test_ground_visual_square_grows_to_cover_far_pits(self):
        from cyberwave_robot_format.usd.ground import (
            GROUND_VISUAL_HALF_EXTENT_M,
            PIT_GROUND_MARGIN_M,
            GroundRect,
            ground_visual_half_extent,
            ground_visual_layout,
        )

        near = GroundRect(20.0, 20.0, 22.0, 22.0, z=-0.2)
        assert ground_visual_half_extent([near]) == GROUND_VISUAL_HALF_EXTENT_M
        # A 200 m track reaches past the 100 m square: keep it whole, with ground around it.
        far = GroundRect(-5.0, -120.0, 5.0, 80.0, z=-0.3)
        tiles, floors = ground_visual_layout([near, far])
        half = 120.0 + PIT_GROUND_MARGIN_M
        assert floors[1] == far
        assert min(t.min_y for t in tiles) == -half
        assert max(t.max_x for t in tiles) == half
        holes = 2.0 * 2.0 + 10.0 * 200.0
        assert sum(r.size[0] * r.size[1] for r in tiles) == pytest.approx((2 * half) ** 2 - holes)

    @pytest.mark.parametrize("z, group", [(0.25, "disabled"), (-0.3, "default")], ids=["not_buried", "colliding"])
    def test_ground_visual_is_untouched_otherwise(self, tmp_path, z, group):
        result = export_scene_usd([arm_spec(), self._scenery_spec(z, group)], tmp_path, options=options())
        stage = Usd.Stage.Open(str(tmp_path / result.root_layer))
        assert not stage.GetPrimAtPath(f"{SCENE}/GroundVisual")
        assert not UsdGeom.Imageable(stage.GetPrimAtPath(f"{SCENE}/GroundPlane")).GetPurposeAttr().HasAuthoredValue()

    def test_bare_schema_input_synthesizes_an_instance(self, tmp_path):
        result = export_scene_usd(arm_schema(), tmp_path)
        assert len(result.identity_map["instances"]) == 1
        assert result.identity_map["instances"][0]["usd_path"].startswith(f"{SCENE}/instance_")

    def test_duplicate_instance_id_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="Duplicate instance_id"):
            export_scene_usd([arm_spec(), arm_spec()], tmp_path)

    def test_collision_mesh_without_points_is_an_error(self, tmp_path):
        schema = arm_schema()
        schema.links[0].collisions = [Collision(geometry=Geometry(type=GeometryType.MESH, filename="meshes/base.stl"))]
        result = export_scene_usd(
            SceneInstance(instance_id=ARM_UUID, schema=schema),
            tmp_path,
            options=options(),
        )
        error = by(result.compatibility["errors"], code="mesh_not_embedded")
        assert error["instance_id"] == ARM_UUID
        assert not result.ok


# --------------------------------------------------------------------------
# Instancing, determinism, round trip
# --------------------------------------------------------------------------


class TestInstancing:
    def test_two_instances_share_one_asset_layer(self, tmp_path):
        schema = arm_schema()
        # An asset-level finding, to check every instance sharing the layer gets it.
        schema.joints[-1].extensions["physx"] = {"physxJoint:notAnAttribute": 1.0}

        def spec(instance_id: str, position: Vector3) -> SceneInstance:
            return SceneInstance(
                instance_id=instance_id,
                schema=copy.deepcopy(schema),
                spawn_pose=Pose(position=position),
            )

        result = export_scene_usd(
            [spec(ARM_UUID, Vector3(1.0, 2.0, 0.0)), spec(ARM2_UUID, Vector3(-1.0, 0.0, 0.0))],
            tmp_path,
            options=options(),
        )
        assets = [f["path"] for f in result.files if f["path"].startswith("assets/")]
        assert len(assets) == 1
        assert assets[0].startswith("assets/kinova_gen3_lite_")
        stage = Usd.Stage.Open(str(tmp_path / result.root_layer))
        first = world_translation(stage, f"{instance_path(ARM_UUID)}/Links/base_link")
        second = world_translation(stage, f"{instance_path(ARM2_UUID)}/Links/base_link")
        assert tuple(first) == pytest.approx((1.0, 2.0, 0.0))
        assert tuple(second) == pytest.approx((-1.0, 0.0, 0.0))
        ids = [s["sensor_id"] for s in result.sensors]
        assert len(ids) == len(set(ids))
        joints = [j["usd_path"] for j in result.identity_map["joints"]]
        assert len(joints) == len(set(joints)) == 10
        # Asset findings are checked once per asset but reported per instance, so
        # a caller grouping findings by instance sees both.
        skipped = [w for w in result.compatibility["warnings"] if w["code"] == "physx_attribute_skipped"]
        assert [w["instance_id"] for w in skipped] == [ARM_UUID, ARM2_UUID]

    def test_asset_error_is_reported_for_every_instance_sharing_the_layer(self, tmp_path):
        schema = arm_schema()
        schema.links[0].collisions = [Collision(geometry=Geometry(type=GeometryType.MESH, filename="meshes/base.stl"))]
        result = export_scene_usd(
            [SceneInstance(instance_id=i, schema=copy.deepcopy(schema)) for i in (ARM_UUID, ARM2_UUID)],
            tmp_path,
            options=options(),
        )
        assert len([f for f in result.files if f["path"].startswith("assets/")]) == 1
        errors = [e for e in result.compatibility["errors"] if e["code"] == "mesh_not_embedded"]
        assert [e["instance_id"] for e in errors] == [ARM_UUID, ARM2_UUID]
        assert errors[0]["subject"] == errors[1]["subject"]


class TestDeterminism:
    def test_output_is_byte_identical_across_directories_and_runs(self, tmp_path):
        specs = lambda: [dog_spec(), drone_spec(), arm_spec()]
        first = export_scene_usd(specs(), tmp_path / "a", options=options())
        second = export_scene_usd(list(reversed(specs())), tmp_path / "b", options=options())
        third = export_scene_usd(specs(), tmp_path / "a", options=options())
        assert first.to_dict() == second.to_dict() == third.to_dict()
        for entry in first.files:
            assert (tmp_path / "a" / entry["path"]).read_bytes() == (tmp_path / "b" / entry["path"]).read_bytes()

    def test_root_layer_has_no_absolute_paths_or_timestamps(self, scene):
        result, _, out = scene
        text = (out / result.root_layer).read_text()
        assert str(out) not in text
        assert "@./assets/" in text
        for entry in result.files:
            assert str(out) not in (out / entry["path"]).read_text()


class TestRoundTrip:
    def test_asset_layer_parses_back_to_the_schema(self, scene):
        result, _, out = scene
        arm_asset = next(f["path"] for f in result.files if f["path"].startswith("assets/kinova"))
        parsed = USDParser().parse(out / arm_asset)
        source = arm_schema()
        assert [link.name for link in parsed.links] == [link.name for link in source.links]
        # The world joint is placement, so it lives in the root layer instead.
        assert [j.name for j in parsed.joints] == [j.name for j in source.joints[1:]]
        assert parsed.actuators[0].kp == pytest.approx(100.0)
        assert parsed.actuators[0].kd == pytest.approx(10.0)
        assert [s.name for s in parsed.sensors] == ["wrist_camera", "wrist_depth"]
        assert parsed.get_joint("right_finger_joint").mimic.joint == "left_finger_joint"

    def test_each_instance_parses_from_the_composed_stage(self, scene):
        result, _, out = scene
        for instance_id, schema in ((DOG_UUID, quadruped_schema()), (DRONE_UUID, quadrotor_schema())):
            mask = Usd.StagePopulationMask([instance_path(instance_id)])
            stage = Usd.Stage.OpenMasked(str(out / result.root_layer), mask)
            parsed = USDParser().parse_stage(stage)
            assert sorted(link.name for link in parsed.links) == sorted(link.name for link in schema.links)
            assert sorted(a.name for a in parsed.actuators) == sorted(a.name for a in schema.actuators)
            assert sorted(s.name for s in parsed.sensors) == sorted(s.name for s in schema.sensors)

    def test_quadrotor_site_motors_survive(self, scene):
        result, _, out = scene
        asset = next(f["path"] for f in result.files if f["path"].startswith("assets/dji"))
        parsed = USDParser().parse(out / asset)
        transmission = parsed.actuators[0].get_mjcf_site_transmission()
        assert transmission["site"] == "rotor_fl"
        assert [s["name"] for s in parsed.links[0].get_mjcf_sites()] == list(ROTOR_POSITIONS)


# --------------------------------------------------------------------------
# Isolation guarantees
# --------------------------------------------------------------------------


class TestNoSimulatorDependency:
    def test_export_imports_no_isaac_modules(self, scene):
        loaded = [name for name in sys.modules if name.split(".")[0] in {"isaacsim", "omni", "carb"}]
        assert loaded == []

    def test_package_source_never_imports_isaac(self):
        import cyberwave_robot_format

        root = Path(cyberwave_robot_format.__file__).parent
        offenders = []
        for path in root.rglob("*.py"):
            for line in path.read_text().splitlines():
                stripped = line.strip()
                if stripped.startswith(("import ", "from ")) and any(
                    token in stripped.split()[1].split(".")[0] for token in ("isaacsim", "omni", "carb")
                ):
                    offenders.append(f"{path.name}: {stripped}")
        assert offenders == []

    def test_only_known_nvidia_schemas_are_authored(self, scene):
        # usd-core cannot see unregistered schemas through GetAppliedSchemas(), so
        # read the raw apiSchemas list. PhysX/Newton API tokens are allowed (they
        # are plain USD, authored without the SDK); Isaac prim types are not.
        from cyberwave_robot_format.usd import physx

        known = {a.api for a in physx.PHYSX_ATTRIBUTES.values()} | {physx.NEWTON_MIMIC_API}
        _, stage, _ = scene
        authored = set()
        for prim in stage.Traverse():
            list_op = prim.GetMetadata("apiSchemas")
            names = list_op.GetAddedOrExplicitItems() if list_op else []
            authored |= {n for n in names if n.startswith(("Physx", "Newton", "Isaac"))}
            assert not prim.GetTypeName().startswith("Isaac")
        assert authored <= known
        assert {"PhysxJointAPI", "NewtonMimicAPI"} <= authored

    def test_usd_package_knows_no_platform_identity(self):
        # Twins, environments and assets belong to the caller (the backend maps
        # them onto instances), exactly as ``merge_in`` leaves them to it.
        from cyberwave_robot_format import usd

        root = Path(usd.__file__).parent
        for path in sorted(root.glob("*.py")):
            source = path.read_text()
            assert "cyberwave_contracts" not in source, path.name
            for word in ("twin_uuid", "environment_uuid", "asset_uuid", "twinUuid", "environmentUuid"):
                assert word not in source, (path.name, word)


# --------------------------------------------------------------------------
# Structural golden
# --------------------------------------------------------------------------

GOLDEN = Path(__file__).parent / "fixtures" / "scene_export_golden.json"


def golden_view(result) -> dict:
    """The result without what depends on usd-core's writer or the package version.

    File hashes/sizes change with USD's ASCII formatting; asset file names embed a
    content hash for the same reason. Everything else — identity map, drives,
    sensors, findings — is pinned exactly.
    """
    data = json.loads(json.dumps(result.to_dict()))
    data["files"] = sorted(
        entry["path"].rsplit("_", 1)[0] if entry["path"].startswith("assets/") else entry["path"]
        for entry in data["files"]
    )
    data["exporter"].pop("version")
    return data


def test_matches_structural_golden(scene):
    """Regenerate with ``CW_REGENERATE_GOLDENS=1 pytest tests/test_usd_scene_export.py``."""
    import os

    result, _, _ = scene
    view = golden_view(result)
    if os.environ.get("CW_REGENERATE_GOLDENS"):
        GOLDEN.write_text(json.dumps(view, indent=2, sort_keys=True) + "\n")
    assert view == json.loads(GOLDEN.read_text())
