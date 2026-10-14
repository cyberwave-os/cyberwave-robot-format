# Copyright [2025] Cyberwave
# Licensed under the Apache License, Version 2.0

"""NVIDIA PhysX / Newton schema authoring (``cyberwave_robot_format.usd.physx``).

Isaac Sim reads velocity limits, armature, mimic couplings and most solver tuning
from PhysX/Newton API schemas that ``usd-core`` does not ship. These tests pin the
attributes the exporter writes for them without the SDK, that Isaac-only values in
``extensions["physx"]`` survive a round trip, and that a foreign stage authored by
Kit reads back into the schema.
"""

from __future__ import annotations

import logging
import math

import pytest

pytest.importorskip("pxr")

from pxr import Sdf, Usd  # noqa: E402

from cyberwave_robot_format.schema import (  # noqa: E402
    Collision,
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
    Vector3,
)
from cyberwave_robot_format.usd import USDExporter, USDParser  # noqa: E402
from cyberwave_robot_format.usd import physx  # noqa: E402
from cyberwave_robot_format.usd import tokens as t  # noqa: E402
from tests.usd_helpers import assert_schema_equal  # noqa: E402


def _link(name: str, **kwargs) -> Link:
    return Link(
        name=name,
        mass=1.0,
        inertia=Inertia(ixx=0.01, iyy=0.01, izz=0.01),
        **kwargs,
    )


def gripper_schema(**joint_overrides) -> CommonSchema:
    """A base with a revolute arm joint and two coupled finger joints."""
    joints = [
        Joint(
            name="shoulder",
            type=JointType.REVOLUTE,
            parent_link="base",
            child_link="arm",
            limits=JointLimits(lower=-1.0, upper=1.0, effort=10.0, velocity=2.0),
            dynamics=JointDynamics(damping=0.1, armature=0.02),
        ),
        Joint(
            name="finger_left",
            type=JointType.REVOLUTE,
            parent_link="arm",
            child_link="left",
            limits=JointLimits(lower=0.0, upper=0.8, velocity=0.0),
        ),
        Joint(
            name="finger_right",
            type=JointType.REVOLUTE,
            parent_link="arm",
            child_link="right",
            limits=JointLimits(lower=-0.8, upper=0.0),
            mimic=MimicJoint(joint="finger_left", multiplier=-1.0, offset=0.1),
        ),
    ]
    for joint in joints:
        for field, value in joint_overrides.get(joint.name, {}).items():
            setattr(joint, field, value)
    return CommonSchema(
        metadata=Metadata(name="gripper"),
        links=[_link("base"), _link("arm"), _link("left"), _link("right")],
        joints=joints,
    )


def exported_stage(schema: CommonSchema) -> Usd.Stage:
    layer = Sdf.Layer.CreateAnonymous(".usda")
    layer.ImportFromString(USDExporter().export_to_string(schema))
    return Usd.Stage.Open(layer)


def api_schemas(prim) -> list[str]:
    # usd-core hides unregistered schemas from GetAppliedSchemas().
    list_op = prim.GetMetadata("apiSchemas")
    return list(list_op.GetAddedOrExplicitItems()) if list_op else []


def joint_prim(stage, name: str):
    return stage.GetPrimAtPath(f"/gripper/Joints/{name}")


class TestDerivedJointAttributes:
    def test_velocity_limit_is_degrees_per_second_for_an_angular_joint(self):
        stage = exported_stage(gripper_schema())
        shoulder = joint_prim(stage, "shoulder")
        assert "PhysxJointAPI" in api_schemas(shoulder)
        velocity = shoulder.GetAttribute("physxJoint:maxJointVelocity").Get()
        assert velocity == pytest.approx(math.degrees(2.0), rel=1e-6)
        assert shoulder.GetAttribute("physxJoint:armature").Get() == pytest.approx(0.02)

    def test_velocity_limit_is_meters_per_second_for_a_prismatic_joint(self):
        joint = Joint(
            name="slide",
            type=JointType.PRISMATIC,
            parent_link="a",
            child_link="b",
            limits=JointLimits(lower=0.0, upper=0.1, velocity=0.25),
        )
        assert physx.derived_joint_physx(joint) == {"physxJoint:maxJointVelocity": 0.25}

    def test_zero_velocity_is_unspecified_not_a_frozen_joint(self):
        # URDF writes velocity="0" when it has no limit; authoring 0 deg/s would
        # lock the joint in Isaac.
        stage = exported_stage(gripper_schema())
        left = joint_prim(stage, "finger_left")
        assert not left.GetAttribute("physxJoint:maxJointVelocity")
        assert "PhysxJointAPI" not in api_schemas(left)

    def test_multi_dof_and_fixed_joints_derive_nothing(self):
        for joint_type in (JointType.FIXED, JointType.SPHERICAL, JointType.FLOATING):
            joint = Joint(
                name="j",
                type=joint_type,
                parent_link="a",
                child_link="b",
                limits=JointLimits(velocity=1.0),
                dynamics=JointDynamics(armature=0.1),
            )
            assert physx.derived_joint_physx(joint) == {}

    def test_friction_is_not_mapped_to_joint_friction(self):
        # physxJoint:jointFriction is a dimensionless coefficient, not a torque.
        joint = Joint(
            name="j",
            type=JointType.REVOLUTE,
            parent_link="a",
            child_link="b",
            dynamics=JointDynamics(friction=0.5),
        )
        assert physx.derived_joint_physx(joint) == {}

    def test_explicit_extension_overrides_the_derived_value(self):
        schema = gripper_schema(
            shoulder={"extensions": {"physx": {"physxJoint:maxJointVelocity": 30.0}}}
        )
        stage = exported_stage(schema)
        velocity = joint_prim(stage, "shoulder").GetAttribute(
            "physxJoint:maxJointVelocity"
        )
        assert velocity.Get() == 30.0


class TestNativeMimic:
    def test_mimic_is_authored_as_newton_mimic_api(self):
        stage = exported_stage(gripper_schema())
        follower = joint_prim(stage, "finger_right")
        assert "NewtonMimicAPI" in api_schemas(follower)
        assert follower.GetRelationship("newton:mimicJoint").GetTargets() == [
            Sdf.Path("/gripper/Joints/finger_left")
        ]
        assert follower.GetAttribute("newton:mimicCoef1").Get() == -1.0
        # Offsets are degrees for an angular joint.
        assert follower.GetAttribute("newton:mimicCoef0").Get() == pytest.approx(
            math.degrees(0.1), rel=1e-6
        )
        assert follower.GetAttribute("newton:mimicEnabled").Get() is True

    def test_leader_written_after_its_follower_is_still_targeted(self):
        schema = gripper_schema()
        schema.joints.reverse()
        stage = exported_stage(schema)
        follower = joint_prim(stage, "finger_right")
        assert follower.GetRelationship("newton:mimicJoint").GetTargets()

    @pytest.mark.parametrize(
        ("overrides", "reason"),
        [
            ({"finger_right": {"type": JointType.CONTINUOUS}}, "continuous"),
            ({"finger_left": {"type": JointType.PRISMATIC}}, "one joint type"),
            ({"finger_left": {"limits": None}}, "finite limits"),
            (
                {"finger_right": {"mimic": MimicJoint(joint="nowhere")}},
                "does not exist",
            ),
        ],
    )
    def test_mimic_nvidia_cannot_simulate_stays_sidecar_only(
        self, overrides, reason, caplog
    ):
        schema = gripper_schema(**overrides)
        follower = next(j for j in schema.joints if j.name == "finger_right")
        leader = next(
            (j for j in schema.joints if j.name == follower.mimic.joint), None
        )
        assert reason in physx.native_mimic_problem(follower, leader)

        with caplog.at_level(logging.WARNING):
            stage = exported_stage(schema)
        prim = joint_prim(stage, "finger_right")
        assert "NewtonMimicAPI" not in api_schemas(prim)
        assert prim.GetAttribute(t.A_JOINT_MIMIC_JOINT)
        assert "not authored natively" in caplog.text

    def test_round_trip_reads_the_sidecar_not_the_float32_native(self):
        schema = gripper_schema(
            finger_right={
                "mimic": MimicJoint(joint="finger_left", multiplier=-1.1, offset=0.123)
            }
        )
        back = USDParser().parse_string(USDExporter().export_to_string(schema))
        assert back.joints[2].mimic == MimicJoint(
            joint="finger_left", multiplier=-1.1, offset=0.123
        )


class TestPassthrough:
    def schema(self) -> CommonSchema:
        schema = gripper_schema()
        schema.extensions = {
            "physx": {
                "physxArticulation:solverPositionIterationCount": 32,
                "physxArticulation:enabledSelfCollisions": False,
            }
        }
        schema.links[1].extensions = {
            "physx": {
                "physxRigidBody:linearDamping": 0.1,
                "physxRigidBody:enableCCD": True,
            }
        }
        schema.links[1].collisions = [
            Collision(
                name="hull",
                geometry=Geometry(
                    type=GeometryType.BOX, size=Vector3(x=0.1, y=0.1, z=0.1)
                ),
                extensions={
                    "physx": {
                        "physxCollision:contactOffset": 0.002,
                        "physxSDFMeshCollision:sdfResolution": 256,
                    }
                },
            )
        ]
        schema.joints[0].extensions = {"physx": {"physxJoint:jointFriction": 0.05}}
        schema.physics = Physics(
            extensions={
                "physx": {
                    "physxScene:solverType": "TGS",
                    "physxScene:gpuMaxRigidContactCount": 1048576,
                }
            }
        )
        return schema

    def test_attributes_and_schemas_land_where_isaac_reads_them(self):
        stage = exported_stage(self.schema())
        root = stage.GetPrimAtPath("/gripper")
        assert "PhysxArticulationAPI" in api_schemas(root)
        assert (
            root.GetAttribute("physxArticulation:solverPositionIterationCount").Get()
            == 32
        )
        arm = stage.GetPrimAtPath("/gripper/Links/arm")
        assert "PhysxRigidBodyAPI" in api_schemas(arm)
        assert arm.GetAttribute("physxRigidBody:enableCCD").Get() is True
        hull = stage.GetPrimAtPath("/gripper/Links/arm/Collisions/hull")
        assert {"PhysxCollisionAPI", "PhysxSDFMeshCollisionAPI"} <= set(
            api_schemas(hull)
        )
        resolution = hull.GetAttribute("physxSDFMeshCollision:sdfResolution")
        assert resolution.GetVariability() == Sdf.VariabilityUniform
        scene = next(p for p in stage.Traverse() if p.GetTypeName() == "PhysicsScene")
        assert scene.GetAttribute("physxScene:solverType").Get() == "TGS"
        assert scene.GetAttribute("physxScene:gpuMaxRigidContactCount").Get() == 1048576
        # Schema-declared, not custom: this is what Kit itself writes.
        assert not scene.GetAttribute("physxScene:solverType").IsCustom()

    def test_round_trip_is_exact(self):
        schema = self.schema()
        back = USDParser().parse_string(USDExporter().export_to_string(schema))
        assert_schema_equal(schema, back)

    def test_derived_values_do_not_leak_into_extensions(self):
        back = USDParser().parse_string(
            USDExporter().export_to_string(gripper_schema())
        )
        assert all("physx" not in joint.extensions for joint in back.joints)

    @pytest.mark.parametrize(
        ("entry", "message"),
        [
            ({"physxJoint:notAThing": 1.0}, "not a supported"),
            ({"physxRigidBody:linearDamping": 0.1}, "belongs to PhysxRigidBodyAPI"),
            ({"physxJoint:armature": "heavy"}, "finite number"),
            ({"physxJoint:armature": float("nan")}, "finite number"),
        ],
    )
    def test_invalid_entries_are_skipped_not_fatal(self, entry, message, caplog):
        schema = gripper_schema(shoulder={"extensions": {"physx": entry}})
        with caplog.at_level(logging.WARNING):
            stage = exported_stage(schema)
        assert message in caplog.text
        shoulder = joint_prim(stage, "shoulder")
        for name in entry:
            if name != "physxJoint:armature":
                assert not shoulder.GetAttribute(name)
        # The derived armature is unaffected by a bad override.
        assert shoulder.GetAttribute("physxJoint:armature").Get() == pytest.approx(0.02)


class TestCoerceValue:
    @pytest.mark.parametrize(
        ("name", "value", "expected"),
        [
            ("physxRigidBody:enableCCD", True, True),
            ("physxRigidBody:solverPositionIterationCount", 8, 8),
            ("physxRigidBody:solverPositionIterationCount", 8.0, 8),
            ("physxScene:gpuTempBufferCapacity", 2**40, 2**40),
            ("physxRigidBody:linearDamping", 1, 1.0),
            ("physxScene:solverType", "PGS", "PGS"),
        ],
    )
    def test_accepts(self, name, value, expected):
        coerced = physx.coerce_value(physx.PHYSX_ATTRIBUTES[name], value)
        assert coerced == expected and type(coerced) is type(expected)

    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("physxRigidBody:enableCCD", 1),
            ("physxRigidBody:solverPositionIterationCount", True),
            ("physxRigidBody:solverPositionIterationCount", 2.5),
            ("physxScene:gpuMaxRigidContactCount", -1),
            ("physxRigidBody:linearDamping", float("inf")),
            ("physxScene:solverType", "Jacobi"),
        ],
    )
    def test_rejects(self, name, value):
        with pytest.raises(ValueError):
            physx.coerce_value(physx.PHYSX_ATTRIBUTES[name], value)

    def test_runtime_owned_scene_attributes_are_not_in_the_table(self):
        for name in (
            "physxScene:timeStepsPerSecond",
            "physxScene:broadphaseType",
            "physxScene:enableGPUDynamics",
            "physxScene:updateType",
            "physxScene:minPositionIterationCount",
        ):
            assert name not in physx.PHYSX_ATTRIBUTES


FOREIGN_STAGE = """#usda 1.0
(
    defaultPrim = "robot"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "robot" (
    prepend apiSchemas = ["PhysicsArticulationRootAPI", "PhysxArticulationAPI"]
)
{
    int physxArticulation:solverPositionIterationCount = 16

    def Xform "base" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysxRigidBodyAPI"]
    )
    {
        float physxRigidBody:angularDamping = 0.5
    }

    def Xform "a" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
    def Xform "b" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
    def Xform "c" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}

    def PhysicsRevoluteJoint "leader" (
        prepend apiSchemas = ["PhysxJointAPI"]
    )
    {
        rel physics:body0 = </robot/base>
        rel physics:body1 = </robot/a>
        uniform token physics:axis = "Z"
        float physics:lowerLimit = -45
        float physics:upperLimit = 45
        float physxJoint:maxJointVelocity = 180
    }

    def PhysicsRevoluteJoint "newton_follower" (
        prepend apiSchemas = ["NewtonMimicAPI"]
    )
    {
        rel physics:body0 = </robot/base>
        rel physics:body1 = </robot/b>
        uniform token physics:axis = "Z"
        float physics:lowerLimit = -45
        float physics:upperLimit = 45
        rel newton:mimicJoint = </robot/leader>
        float newton:mimicCoef0 = 90
        float newton:mimicCoef1 = 0.5
    }

    def PhysicsRevoluteJoint "physx_follower" (
        prepend apiSchemas = ["PhysxMimicJointAPI:rotZ"]
    )
    {
        rel physics:body0 = </robot/base>
        rel physics:body1 = </robot/c>
        uniform token physics:axis = "Z"
        float physics:lowerLimit = -45
        float physics:upperLimit = 45
        rel physxMimicJoint:rotZ:referenceJoint = </robot/leader>
        float physxMimicJoint:rotZ:gearing = 2
        float physxMimicJoint:rotZ:offset = -90
    }
}
"""


class TestForeignStage:
    @pytest.fixture
    def schema(self) -> CommonSchema:
        return USDParser().parse_string(FOREIGN_STAGE)

    def joint(self, schema, name) -> Joint:
        return next(j for j in schema.joints if j.name == name)

    def test_physx_attributes_are_harvested(self, schema):
        assert schema.extensions["physx"] == {
            "physxArticulation:solverPositionIterationCount": 16
        }
        base = next(link for link in schema.links if link.name == "base")
        assert base.extensions["physx"] == {"physxRigidBody:angularDamping": 0.5}
        # No Cyberwave sidecar, so the velocity limit is kept as passthrough.
        assert self.joint(schema, "leader").extensions["physx"] == {
            "physxJoint:maxJointVelocity": 180.0
        }

    def test_newton_mimic_is_imported(self, schema):
        mimic = self.joint(schema, "newton_follower").mimic
        assert mimic.joint == "leader"
        assert mimic.multiplier == pytest.approx(0.5)
        assert mimic.offset == pytest.approx(math.pi / 2)

    def test_deprecated_physx_mimic_is_imported_with_its_sign_convention(self, schema):
        # joint + gearing * reference + offset = 0
        mimic = self.joint(schema, "physx_follower").mimic
        assert mimic.joint == "leader"
        assert mimic.multiplier == pytest.approx(-2.0)
        assert mimic.offset == pytest.approx(math.pi / 2)

    def test_disabled_newton_mimic_is_not_a_mimic(self):
        text = FOREIGN_STAGE.replace(
            "float newton:mimicCoef1 = 0.5",
            "float newton:mimicCoef1 = 0.5\n        bool newton:mimicEnabled = 0",
        )
        schema = USDParser().parse_string(text)
        assert self.joint(schema, "newton_follower").mimic is None

    def test_re_export_keeps_the_native_attributes(self, schema):
        stage = exported_stage(schema)
        root = stage.GetDefaultPrim()
        assert (
            root.GetAttribute("physxArticulation:solverPositionIterationCount").Get()
            == 16
        )
        follower = stage.GetPrimAtPath(f"{root.GetPath()}/Joints/newton_follower")
        assert "NewtonMimicAPI" in api_schemas(follower)
