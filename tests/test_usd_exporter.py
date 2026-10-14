# Copyright [2025] Cyberwave
# Licensed under the Apache License, Version 2.0

"""Unit tests for ``USDExporter``.

Golden-text assertions here are deliberately **targeted** substring or structural
checks, never whole-file comparisons. USD's ASCII writer formatting (float
precision, blank lines, metadata ordering) shifts between ``usd-core`` releases, so
a whole-file golden would be a version tripwire rather than a test. What is worth
pinning down is what a human must be able to read in the ``.usda`` and what a
physics consumer must be able to rely on — so most assertions go through the pxr
API against the exported stage.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pxr")

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402

from cyberwave_robot_format.schema import (  # noqa: E402
    Actuator,
    ActuatorType,
    Collision,
    CollisionConfig,
    CollisionExclude,
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
    Material,
    Metadata,
    MimicJoint,
    Physics,
    Pose,
    Quaternion,
    Sensor,
    Vector3,
    Visual,
)
from cyberwave_robot_format.usd import USDExporter, USDParser  # noqa: E402
from cyberwave_robot_format.usd import tokens as t  # noqa: E402


def export(schema: CommonSchema) -> str:
    return USDExporter().export_to_string(schema)


def open_exported(schema: CommonSchema) -> Usd.Stage:
    """Export ``schema`` and reopen it as a stage for structural assertions."""
    layer = Sdf.Layer.CreateAnonymous(".usda")
    assert layer.ImportFromString(export(schema))
    stage = Usd.Stage.Open(layer)
    assert stage is not None
    # The layer must stay referenced for as long as the stage is used, hence the
    # stage being returned rather than the caller re-deriving it.
    return stage


def sphere(radius: float = 1.0) -> Geometry:
    return Geometry(type=GeometryType.SPHERE, radius=radius)


def two_link_schema(**joint_kwargs) -> CommonSchema:
    kwargs = {
        "name": "j",
        "type": JointType.REVOLUTE,
        "parent_link": "a",
        "child_link": "b",
    }
    kwargs.update(joint_kwargs)
    return CommonSchema(
        metadata=Metadata(name="robot"),
        links=[Link(name="a"), Link(name="b")],
        joints=[Joint(**kwargs)],
    )


def frame_matrix(position, quaternion) -> np.ndarray:
    """Build a column-vector 4x4 from a ``Gf.Vec3`` and a real-first ``Gf.Quat``."""
    matrix = np.eye(4)
    matrix[:3, :3] = np.array(Gf.Matrix3d(Gf.Rotation(Gf.Quatd(quaternion)))).T
    matrix[:3, 3] = [position[0], position[1], position[2]]
    return matrix


def world_transform(prim: Usd.Prim) -> np.ndarray:
    return np.array(
        UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()),
        dtype=np.float64,
    ).T


class TestStageMetadata:
    def test_units_and_up_axis_are_authored_explicitly(self):
        text = export(CommonSchema(metadata=Metadata(name="robot")))
        assert 'upAxis = "Z"' in text
        assert "metersPerUnit = 1" in text
        assert "kilogramsPerUnit = 1" in text

    def test_default_prim_is_the_sanitized_root_name(self):
        text = export(CommonSchema(metadata=Metadata(name="my robot-2.0")))
        assert 'defaultPrim = "my_robot_2_0"' in text

    def test_generator_comment_is_present(self):
        assert "cyberwave-robot-format" in export(
            CommonSchema(metadata=Metadata(name="robot"))
        )

    def test_articulation_root_is_applied_to_the_root_prim(self):
        stage = open_exported(CommonSchema(metadata=Metadata(name="robot")))
        root = stage.GetDefaultPrim()
        assert root.HasAPI(UsdPhysics.ArticulationRootAPI)


class TestLinkLayout:
    def test_no_rigid_body_has_a_rigid_body_ancestor(self):
        # UsdPhysics ignores a RigidBodyAPI nested under another one, which is why
        # link prims are flat siblings under /Links rather than a kinematic tree.
        stage = open_exported(
            CommonSchema(
                metadata=Metadata(name="chain"),
                links=[Link(name=f"l{index}") for index in range(4)],
                joints=[
                    Joint(
                        name=f"j{index}",
                        type=JointType.REVOLUTE,
                        parent_link=f"l{index}",
                        child_link=f"l{index + 1}",
                        pose=Pose(position=Vector3(z=0.1)),
                    )
                    for index in range(3)
                ],
            )
        )
        bodies = [
            prim
            for prim in stage.Traverse()
            if prim.HasAPI(UsdPhysics.RigidBodyAPI)
        ]
        assert len(bodies) == 4
        for prim in bodies:
            ancestor = prim.GetParent()
            while ancestor and ancestor.IsValid():
                assert not ancestor.HasAPI(UsdPhysics.RigidBodyAPI)
                ancestor = ancestor.GetParent()

    def test_link_world_transforms_accumulate_along_the_chain(self):
        pitch = Quaternion.from_rpy(0.0, math.pi / 2, 0.0)
        schema = CommonSchema(
            metadata=Metadata(name="chain"),
            links=[Link(name="a"), Link(name="b"), Link(name="c")],
            joints=[
                Joint(
                    name="ab",
                    type=JointType.REVOLUTE,
                    parent_link="a",
                    child_link="b",
                    pose=Pose(position=Vector3(z=0.5), orientation=pitch),
                ),
                Joint(
                    name="bc",
                    type=JointType.REVOLUTE,
                    parent_link="b",
                    child_link="c",
                    pose=Pose(position=Vector3(x=0.25)),
                ),
            ],
        )
        stage = open_exported(schema)
        root = stage.GetDefaultPrim().GetPath()

        expected_b = np.eye(4)
        expected_b[:3, :3] = np.array(
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
        )
        expected_b[:3, 3] = [0.0, 0.0, 0.5]
        local_c = np.eye(4)
        local_c[:3, 3] = [0.25, 0.0, 0.0]

        assert np.allclose(
            world_transform(stage.GetPrimAtPath(root.AppendPath("Links/a"))), np.eye(4)
        )
        assert np.allclose(
            world_transform(stage.GetPrimAtPath(root.AppendPath("Links/b"))),
            expected_b,
            atol=1e-9,
        )
        assert np.allclose(
            world_transform(stage.GetPrimAtPath(root.AppendPath("Links/c"))),
            expected_b @ local_c,
            atol=1e-9,
        )

    def test_home_position_is_not_baked_into_link_transforms(self):
        # Baking it would have to be un-baked on import and would make the link
        # placements disagree with the joint poses the joint prims carry.
        without = export(two_link_schema(pose=Pose(position=Vector3(z=0.3))))
        with_home = export(
            two_link_schema(pose=Pose(position=Vector3(z=0.3)), home_position=1.0)
        )
        assert "xformOp:transform" in without
        matrices_without = [
            line for line in without.splitlines() if "xformOp:transform" in line
        ]
        matrices_with = [
            line for line in with_home.splitlines() if "xformOp:transform" in line
        ]
        assert matrices_without == matrices_with


class TestMass:
    def test_mass_api_is_omitted_for_a_default_inertial_with_geometry(self):
        # physics:mass = 0 asks UsdPhysics to derive mass from density and geometry,
        # the opposite of what a schema mass of 0.0 means — so MassAPI must not be
        # applied at all for a link with no inertial data but a collider to
        # derive it from.
        stage = open_exported(
            CommonSchema(
                metadata=Metadata(name="r"),
                links=[Link(name="a", collisions=[Collision(geometry=sphere(0.1))])],
            )
        )
        link = stage.GetPrimAtPath(
            stage.GetDefaultPrim().GetPath().AppendPath("Links/a")
        )
        assert link.HasAPI(UsdPhysics.RigidBodyAPI)
        assert not link.HasAPI(UsdPhysics.MassAPI)
        assert not link.GetAttribute(t.A_LINK_MASS_FALLBACK).IsValid()

    def test_zero_mass_collider_link_is_massless_once_the_schema_has_inertials(self):
        # Spot: the floating-base `body` link carries the collision mesh but no
        # inertial; its 32 kg sit on `base_link`, fixed to it. Deriving `body`'s
        # mass from its collider gave PhysX an 81 kg Spot (MuJoCo: 50 kg) that
        # the policy could not hold up. With inertials authored elsewhere, zero
        # means massless, as the MJCF exporter already treats it.
        schema = CommonSchema(
            metadata=Metadata(name="dog"),
            links=[
                Link(name="body", collisions=[Collision(geometry=sphere(0.3))]),
                Link(
                    name="base_link",
                    mass=32.86,
                    inertia=Inertia(ixx=0.1, iyy=0.9, izz=0.9),
                ),
            ],
            joints=[
                Joint(
                    name="base_joint",
                    type=JointType.FIXED,
                    parent_link="body",
                    child_link="base_link",
                )
            ],
        )
        stage = open_exported(schema)
        body = stage.GetPrimAtPath(
            stage.GetDefaultPrim().GetPath().AppendPath("Links/body")
        )
        assert body.HasAPI(UsdPhysics.MassAPI)
        assert UsdPhysics.MassAPI(body).GetMassAttr().Get() == pytest.approx(
            t.MASSLESS_LINK_FALLBACK_MASS_KG
        )
        assert body.GetAttribute(t.A_LINK_MASS_FALLBACK).Get() is True

        parsed = USDParser().parse_string(export(schema))
        by_name = {link.name: link for link in parsed.links}
        assert by_name["body"].mass == 0.0
        assert by_name["body"].inertia == Inertia()
        assert by_name["base_link"].mass == pytest.approx(32.86)
        assert export(parsed) == export(schema)

    def test_massless_link_without_geometry_gets_a_tiny_explicit_inertial(self):
        # A bare frame (camera optical frame, tool tip) is still a rigid body,
        # and with nothing to derive a mass from PhysX invents one ("negative
        # mass, small sphere approximated inertia") that moved the Kinova wrist
        # camera off the arm. The exporter authors a gram-scale stand-in.
        stage = open_exported(
            CommonSchema(
                metadata=Metadata(name="r"),
                links=[Link(name="a", visuals=[Visual(geometry=sphere(0.01))])],
            )
        )
        link = stage.GetPrimAtPath(
            stage.GetDefaultPrim().GetPath().AppendPath("Links/a")
        )
        assert link.HasAPI(UsdPhysics.RigidBodyAPI)
        assert link.HasAPI(UsdPhysics.MassAPI)
        mass_api = UsdPhysics.MassAPI(link)
        assert mass_api.GetMassAttr().Get() == pytest.approx(
            t.MASSLESS_LINK_FALLBACK_MASS_KG
        )
        assert tuple(mass_api.GetDiagonalInertiaAttr().Get()) == pytest.approx(
            (t.MASSLESS_LINK_FALLBACK_INERTIA_KG_M2,) * 3
        )
        assert tuple(mass_api.GetCenterOfMassAttr().Get()) == (0.0, 0.0, 0.0)
        assert link.GetAttribute(t.A_LINK_MASS_FALLBACK).Get() is True
        # The sidecars keep the schema's own values.
        assert link.GetAttribute(t.A_LINK_MASS).Get() == 0.0
        assert list(link.GetAttribute(t.A_LINK_INERTIA).Get()) == [0.0] * 6

    def test_massless_frame_chain_round_trips_to_zero_mass(self):
        # The Kinova wrist: bracelet -> end_effector_link -> camera frames, all
        # fixed, none with mass, inertia or geometry.
        frames = ["end_effector_link", "camera_link", "camera_depth_frame"]
        schema = CommonSchema(
            metadata=Metadata(name="wrist"),
            links=[
                Link(
                    name="bracelet_link",
                    mass=0.5,
                    inertia=Inertia(ixx=1e-3, iyy=1e-3, izz=1e-3),
                ),
                *(Link(name=name) for name in frames),
            ],
            joints=[
                Joint(
                    name=f"{child}_joint",
                    type=JointType.FIXED,
                    parent_link=parent,
                    child_link=child,
                )
                for parent, child in zip(
                    ["bracelet_link", "end_effector_link", "end_effector_link"],
                    frames,
                    strict=True,
                )
            ],
        )
        stage = open_exported(schema)
        root = stage.GetDefaultPrim().GetPath()
        for name in frames:
            prim = stage.GetPrimAtPath(root.AppendPath(f"Links/{name}"))
            assert UsdPhysics.MassAPI(prim).GetMassAttr().Get() > 0.0, name
        bracelet = stage.GetPrimAtPath(root.AppendPath("Links/bracelet_link"))
        assert not bracelet.GetAttribute(t.A_LINK_MASS_FALLBACK).IsValid()
        assert UsdPhysics.MassAPI(bracelet).GetMassAttr().Get() == pytest.approx(0.5)

        parsed = USDParser().parse_string(export(schema))
        by_name = {link.name: link for link in parsed.links}
        for name in frames:
            assert by_name[name].mass == 0.0
            assert by_name[name].inertia == Inertia()
        assert export(parsed) == export(schema)

    def test_fallback_marker_alone_still_reads_as_massless(self):
        # A consumer that strips the double sidecars must not turn the stand-in
        # into the link's real mass on import.
        text = export(CommonSchema(metadata=Metadata(name="r"), links=[Link(name="a")]))
        layer = Sdf.Layer.CreateAnonymous(".usda")
        assert layer.ImportFromString(text)
        stage = Usd.Stage.Open(layer)
        link = stage.GetPrimAtPath(stage.GetDefaultPrim().GetPath().AppendPath("Links/a"))
        link.RemoveProperty(t.A_LINK_MASS)
        link.RemoveProperty(t.A_LINK_INERTIA)
        (parsed_link,) = USDParser().parse_string(layer.ExportToString()).links
        assert parsed_link.mass == 0.0
        assert parsed_link.inertia == Inertia()

    def test_mass_and_inertia_are_authored_natively_and_as_sidecars(self):
        schema = CommonSchema(
            metadata=Metadata(name="r"),
            links=[
                Link(
                    name="a",
                    mass=2.5,
                    center_of_mass=Vector3(x=0.1),
                    inertia=Inertia(ixx=1.0, iyy=2.0, izz=3.0),
                )
            ],
        )
        text = export(schema)
        assert "float physics:mass = 2.5" in text
        assert "float3 physics:diagonalInertia = (1, 2, 3)" in text
        assert t.A_LINK_MASS in text
        assert t.A_LINK_CENTER_OF_MASS in text
        assert t.A_LINK_INERTIA in text

    def test_off_diagonal_inertia_authors_principal_axes(self):
        schema = CommonSchema(
            metadata=Metadata(name="r"),
            links=[
                Link(name="a", mass=1.0, inertia=Inertia(ixx=1.0, iyy=2.0, izz=3.0, ixy=0.4))
            ],
        )
        stage = open_exported(schema)
        link = stage.GetPrimAtPath(
            stage.GetDefaultPrim().GetPath().AppendPath("Links/a")
        )
        principal = UsdPhysics.MassAPI(link).GetPrincipalAxesAttr()
        assert principal.HasAuthoredValue()
        assert Gf.Quatf(principal.Get()) != Gf.Quatf(1.0, 0.0, 0.0, 0.0)


class TestJoints:
    @pytest.mark.parametrize(
        ("joint_type", "prim_type"),
        [
            (JointType.REVOLUTE, "PhysicsRevoluteJoint"),
            (JointType.CONTINUOUS, "PhysicsRevoluteJoint"),
            (JointType.PRISMATIC, "PhysicsPrismaticJoint"),
            (JointType.FIXED, "PhysicsFixedJoint"),
            (JointType.SPHERICAL, "PhysicsSphericalJoint"),
            (JointType.FLOATING, "PhysicsJoint"),
            (JointType.PLANAR, "PhysicsJoint"),
            (JointType.UNIVERSAL, "PhysicsJoint"),
        ],
    )
    def test_joint_prim_types(self, joint_type, prim_type):
        stage = open_exported(two_link_schema(type=joint_type))
        joint = stage.GetPrimAtPath(
            stage.GetDefaultPrim().GetPath().AppendPath("Joints/j")
        )
        assert str(joint.GetTypeName()) == prim_type
        # cyberwave:joint:type is what tells CONTINUOUS from REVOLUTE and
        # PLANAR/UNIVERSAL from FLOATING, all of which share a prim type.
        assert joint.GetAttribute(t.A_JOINT_TYPE).Get() == joint_type.value

    def test_axis_token_is_always_x_with_the_axis_baked_into_the_frame(self):
        stage = open_exported(two_link_schema(axis=Vector3(y=1.0)))
        joint = UsdPhysics.RevoluteJoint(
            stage.GetPrimAtPath(
                stage.GetDefaultPrim().GetPath().AppendPath("Joints/j")
            )
        )
        assert joint.GetAxisAttr().Get() == "X"
        rotation = Gf.Matrix3d(Gf.Rotation(Gf.Quatd(joint.GetLocalRot1Attr().Get())))
        assert np.allclose(
            np.array(rotation).T @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-6
        )

    def test_revolute_limits_are_degrees_natively_and_radians_in_the_sidecar(self):
        stage = open_exported(
            two_link_schema(limits=JointLimits(lower=-math.pi / 2, upper=math.pi / 2))
        )
        joint = stage.GetPrimAtPath(
            stage.GetDefaultPrim().GetPath().AppendPath("Joints/j")
        )
        assert math.isclose(
            joint.GetAttribute("physics:lowerLimit").Get(), -90.0, abs_tol=1e-4
        )
        assert joint.GetAttribute(t.A_JOINT_LIMIT_LOWER).Get() == -math.pi / 2

    def test_prismatic_limits_are_not_converted(self):
        stage = open_exported(
            two_link_schema(
                type=JointType.PRISMATIC, limits=JointLimits(lower=-0.05, upper=0.15)
            )
        )
        joint = stage.GetPrimAtPath(
            stage.GetDefaultPrim().GetPath().AppendPath("Joints/j")
        )
        assert math.isclose(
            joint.GetAttribute("physics:lowerLimit").Get(), -0.05, abs_tol=1e-7
        )

    def test_continuous_joint_authors_no_native_limits(self):
        # An unauthored -inf/+inf pair *is* "unlimited" in UsdPhysics.
        stage = open_exported(
            two_link_schema(type=JointType.CONTINUOUS, limits=JointLimits(effort=5.0))
        )
        joint = stage.GetPrimAtPath(
            stage.GetDefaultPrim().GetPath().AppendPath("Joints/j")
        )
        assert not joint.GetAttribute("physics:lowerLimit").HasAuthoredValue()
        assert not joint.GetAttribute("physics:upperLimit").HasAuthoredValue()
        assert joint.GetAttribute(t.A_JOINT_TYPE).Get() == "continuous"

    def test_floating_joint_excludes_from_articulation_and_locks_nothing(self):
        stage = open_exported(two_link_schema(type=JointType.FLOATING))
        joint = stage.GetPrimAtPath(
            stage.GetDefaultPrim().GetPath().AppendPath("Joints/j")
        )
        assert joint.GetAttribute("physics:excludeFromArticulation").Get() is True
        assert not [
            schema
            for schema in joint.GetAppliedSchemas()
            if schema.startswith("PhysicsLimitAPI")
        ]

    @pytest.mark.parametrize(
        ("joint_type", "locked"),
        [
            (JointType.PLANAR, {"transX", "rotY", "rotZ"}),
            (JointType.UNIVERSAL, {"transX", "transY", "transZ", "rotZ"}),
        ],
    )
    def test_reduced_dof_joints_lock_the_right_axes(self, joint_type, locked):
        stage = open_exported(two_link_schema(type=joint_type))
        joint = stage.GetPrimAtPath(
            stage.GetDefaultPrim().GetPath().AppendPath("Joints/j")
        )
        instances = {
            schema.split(":", 1)[1]
            for schema in joint.GetAppliedSchemas()
            if schema.startswith("PhysicsLimitAPI:")
        }
        assert instances == locked
        for instance in instances:
            limit = UsdPhysics.LimitAPI(joint, instance)
            # low > high is UsdPhysics' encoding for a locked DOF.
            assert limit.GetLowAttr().Get() > limit.GetHighAttr().Get()

    def test_world_parent_leaves_body0_empty(self):
        schema = CommonSchema(
            metadata=Metadata(name="r"),
            links=[Link(name="a")],
            joints=[
                Joint(name="j", type=JointType.FIXED, parent_link="world", child_link="a")
            ],
        )
        stage = open_exported(schema)
        joint = UsdPhysics.Joint(
            stage.GetPrimAtPath(
                stage.GetDefaultPrim().GetPath().AppendPath("Joints/j")
            )
        )
        assert joint.GetBody0Rel().GetTargets() == []
        assert len(joint.GetBody1Rel().GetTargets()) == 1

    def test_a_real_world_link_is_bound_to_body0(self):
        schema = CommonSchema(
            metadata=Metadata(name="r"),
            links=[Link(name="world"), Link(name="a")],
            joints=[
                Joint(name="j", type=JointType.FIXED, parent_link="world", child_link="a")
            ],
        )
        stage = open_exported(schema)
        joint = UsdPhysics.Joint(
            stage.GetPrimAtPath(
                stage.GetDefaultPrim().GetPath().AppendPath("Joints/j")
            )
        )
        targets = joint.GetBody0Rel().GetTargets()
        assert len(targets) == 1
        assert targets[0].name == "world"

    def test_joint_frames_coincide_at_the_zero_configuration(self):
        # The UsdPhysics consistency requirement, and the single most important
        # structural property of the export: if the two joint frames do not land in
        # the same place in world space, the stage is not simulatable.
        schema = CommonSchema(
            metadata=Metadata(name="arm"),
            links=[Link(name=f"l{index}") for index in range(4)],
            joints=[
                Joint(
                    name="j0",
                    type=JointType.REVOLUTE,
                    parent_link="l0",
                    child_link="l1",
                    pose=Pose(
                        position=Vector3(x=0.1, z=0.2),
                        orientation=Quaternion.from_rpy(0.3, -0.2, 0.5),
                    ),
                    axis=Vector3(x=0.0, y=0.6, z=0.8),
                ),
                Joint(
                    name="j1",
                    type=JointType.PRISMATIC,
                    parent_link="l1",
                    child_link="l2",
                    pose=Pose(
                        position=Vector3(y=-0.3),
                        orientation=Quaternion.from_rpy(-0.7, 0.1, 0.0),
                    ),
                    axis=Vector3(x=1.0, y=1.0, z=0.0),
                ),
                Joint(
                    name="j2",
                    type=JointType.FIXED,
                    parent_link="l2",
                    child_link="l3",
                    pose=Pose(
                        position=Vector3(z=0.05),
                        orientation=Quaternion.from_rpy(0.0, 1.2, -0.4),
                    ),
                ),
            ],
        )
        stage = open_exported(schema)
        root = stage.GetDefaultPrim().GetPath()

        for name in ("j0", "j1", "j2"):
            joint = UsdPhysics.Joint(
                stage.GetPrimAtPath(root.AppendPath(f"Joints/{name}"))
            )
            body0 = stage.GetPrimAtPath(joint.GetBody0Rel().GetTargets()[0])
            body1 = stage.GetPrimAtPath(joint.GetBody1Rel().GetTargets()[0])
            frame0 = world_transform(body0) @ frame_matrix(
                joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get()
            )
            frame1 = world_transform(body1) @ frame_matrix(
                joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get()
            )
            assert np.allclose(frame0, frame1, atol=1e-5), name

    def test_mimic_and_extra_limits_are_sidecar_only(self):
        schema = two_link_schema(
            limits=JointLimits(acceleration=1.5, jerkmax=9.0),
            dynamics=JointDynamics(armature=0.01, backlash=0.002),
        )
        schema.joints[0].mimic = MimicJoint(joint="other", multiplier=-1.0)
        text = export(schema)
        for attribute in (
            t.A_JOINT_LIMIT_ACCELERATION,
            t.A_JOINT_LIMIT_JERKMAX,
            t.A_JOINT_DYN_ARMATURE,
            t.A_JOINT_DYN_BACKLASH,
            t.A_JOINT_MIMIC_JOINT,
        ):
            assert attribute in text


class TestGeometry:
    def test_box_is_a_unit_cube_plus_scale(self):
        schema = CommonSchema(
            metadata=Metadata(name="r"),
            links=[
                Link(
                    name="a",
                    visuals=[
                        Visual(
                            name="v",
                            geometry=Geometry(
                                type=GeometryType.BOX, size=Vector3(x=0.1, y=0.2, z=0.3)
                            ),
                        )
                    ],
                )
            ],
        )
        text = export(schema)
        assert "double size = 1" in text
        assert "double3 xformOp:scale = (0.1, 0.2, 0.3)" in text

    def test_gprim_scale_is_applied_before_the_pose(self):
        # With ["xformOp:transform", "xformOp:scale"] USD scales the geometry first
        # and then places it. The reverse order would scale the offset too, so a
        # box 1 m out would end up at 0.1 m.
        schema = CommonSchema(
            metadata=Metadata(name="r"),
            links=[
                Link(
                    name="a",
                    visuals=[
                        Visual(
                            name="v",
                            pose=Pose(position=Vector3(x=1.0)),
                            geometry=Geometry(
                                type=GeometryType.BOX, size=Vector3(x=0.1, y=0.1, z=0.1)
                            ),
                        )
                    ],
                )
            ],
        )
        stage = open_exported(schema)
        prim = stage.GetPrimAtPath(
            stage.GetDefaultPrim().GetPath().AppendPath("Links/a/Visuals/v")
        )
        # A unit cube spans -0.5..0.5, so its +X corner scales to 0.05 and then
        # translates to 1.05. Reversed ops would give 0.1 * (1 + 0.5) = 0.15.
        point = world_transform(prim) @ np.array([0.5, 0.0, 0.0, 1.0])
        assert np.allclose(point[:3], [1.05, 0.0, 0.0], atol=1e-9)

    @pytest.mark.parametrize(
        ("geometry_type", "prim_type"),
        [
            (GeometryType.SPHERE, "Sphere"),
            (GeometryType.ELLIPSOID, "Sphere"),
            (GeometryType.CYLINDER, "Cylinder"),
            (GeometryType.CAPSULE, "Capsule"),
            (GeometryType.PLANE, "Plane"),
            (GeometryType.MESH, "Mesh"),
            (GeometryType.BOX, "Cube"),
            (GeometryType.UNKNOWN, "Xform"),
        ],
    )
    def test_geometry_prim_types(self, geometry_type, prim_type):
        geometry = Geometry(
            type=geometry_type,
            radius=0.2,
            length=0.4,
            size=Vector3(x=1.0, y=2.0, z=3.0),
            filename="m.obj" if geometry_type == GeometryType.MESH else None,
        )
        schema = CommonSchema(
            metadata=Metadata(name="r"),
            links=[Link(name="a", visuals=[Visual(name="v", geometry=geometry)])],
        )
        stage = open_exported(schema)
        prim = stage.GetPrimAtPath(
            stage.GetDefaultPrim().GetPath().AppendPath("Links/a/Visuals/v")
        )
        assert str(prim.GetTypeName()) == prim_type
        assert prim.GetAttribute(t.A_GEOM_TYPE).Get() == geometry_type.value

    def test_cylinder_and_capsule_are_z_aligned(self):
        # Matches the URDF and MJCF convention, so no corrective rotation is needed.
        for geometry_type in (GeometryType.CYLINDER, GeometryType.CAPSULE):
            schema = CommonSchema(
                metadata=Metadata(name="r"),
                links=[
                    Link(
                        name="a",
                        visuals=[
                            Visual(
                                name="v",
                                geometry=Geometry(
                                    type=geometry_type, radius=0.2, length=0.4
                                ),
                            )
                        ],
                    )
                ],
            )
            assert 'uniform token axis = "Z"' in export(schema)

    def test_ellipsoid_is_a_scaled_unit_sphere(self):
        schema = CommonSchema(
            metadata=Metadata(name="r"),
            links=[
                Link(
                    name="a",
                    visuals=[
                        Visual(
                            name="v",
                            geometry=Geometry(
                                type=GeometryType.ELLIPSOID,
                                size=Vector3(x=0.1, y=0.2, z=0.05),
                            ),
                        )
                    ],
                )
            ],
        )
        text = export(schema)
        assert "double radius = 1" in text
        assert "double3 xformOp:scale = (0.1, 0.2, 0.05)" in text
        assert f'{t.A_GEOM_TYPE} = "ellipsoid"' in text

    def test_plane_authors_width_and_length(self):
        schema = CommonSchema(
            metadata=Metadata(name="r"),
            links=[
                Link(
                    name="a",
                    visuals=[
                        Visual(
                            name="v",
                            geometry=Geometry(
                                type=GeometryType.PLANE,
                                size=Vector3(x=4.0, y=5.0, z=0.5),
                            ),
                        )
                    ],
                )
            ],
        )
        text = export(schema)
        assert "double width = 4" in text
        assert "double length = 5" in text

    def test_mesh_filename_survives_as_a_plain_string(self):
        # A string rather than an asset path: usdz packaging *rewrites* asset paths
        # during localization, which would turn package:// URIs into "0/arm.STL".
        schema = CommonSchema(
            metadata=Metadata(name="r"),
            links=[
                Link(
                    name="a",
                    visuals=[
                        Visual(
                            name="v",
                            geometry=Geometry(
                                type=GeometryType.MESH,
                                filename="package://demo/meshes/arm.STL",
                                scale=Vector3(x=0.001, y=0.001, z=0.001),
                            ),
                        )
                    ],
                )
            ],
        )
        text = export(schema)
        assert 'package://demo/meshes/arm.STL"' in text
        assert "@package://demo/meshes/arm.STL@" not in text
        assert "double3 xformOp:scale = (0.001, 0.001, 0.001)" in text

    def test_mesh_baking_is_off_by_default(self, tmp_path):
        obj = tmp_path / "tri.obj"
        obj.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
        schema = CommonSchema(
            metadata=Metadata(name="r"),
            links=[
                Link(
                    name="a",
                    visuals=[
                        Visual(
                            name="v",
                            geometry=Geometry(type=GeometryType.MESH, filename=str(obj)),
                        )
                    ],
                )
            ],
        )
        assert "point3f[] points" not in USDExporter().export_to_string(schema)
        baked = USDExporter(bake_meshes=True).export_to_string(schema)
        assert "point3f[] points" in baked
        assert 'uniform token subdivisionScheme = "none"' in baked

    @pytest.mark.parametrize(
        "filename",
        [
            "meshes/tri.obj",
            "package://pkg/meshes/tri.obj",
        ],
        ids=["relative", "package-uri"],
    )
    def test_mesh_search_paths_resolve_a_non_absolute_reference(
        self, tmp_path, filename
    ):
        # resolve_mesh_uri takes one base directory, not a list, so passing
        # mesh_search_paths straight through raised inside the best-effort `except`
        # and every relative or package:// mesh silently failed to bake.
        meshes = tmp_path / "pkg" / "meshes"
        meshes.mkdir(parents=True)
        (meshes / "tri.obj").write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
        exporter = USDExporter(
            bake_meshes=True, mesh_search_paths=[str(tmp_path / "pkg"), str(tmp_path)]
        )
        schema = CommonSchema(
            metadata=Metadata(name="r"),
            links=[
                Link(
                    name="a",
                    visuals=[
                        Visual(
                            name="v",
                            geometry=Geometry(
                                type=GeometryType.MESH, filename=filename
                            ),
                        )
                    ],
                )
            ],
        )
        text = exporter.export_to_string(schema)
        assert "point3f[] points" in text
        # The reference itself is never replaced by the resolved path.
        assert f'cyberwave:geometry:meshFilename = "{filename}"' in text

    def test_unbakeable_mesh_degrades_to_a_reference_only_prim(self, caplog):
        schema = CommonSchema(
            metadata=Metadata(name="r"),
            links=[
                Link(
                    name="a",
                    visuals=[
                        Visual(
                            name="v",
                            geometry=Geometry(
                                type=GeometryType.MESH,
                                filename="package://nope/missing.stl",
                            ),
                        )
                    ],
                )
            ],
        )
        text = USDExporter(bake_meshes=True).export_to_string(schema)
        assert "point3f[] points" not in text
        assert "package://nope/missing.stl" in text


class TestCollisions:
    @staticmethod
    def _schema(**collision_kwargs) -> CommonSchema:
        kwargs = {"name": "c", "geometry": sphere(0.2)}
        kwargs.update(collision_kwargs)
        return CommonSchema(
            metadata=Metadata(name="r"),
            links=[Link(name="a", collisions=[Collision(**kwargs)])],
        )

    def test_collision_prims_are_guide_purpose_with_collision_api(self):
        stage = open_exported(self._schema())
        prim = stage.GetPrimAtPath(
            stage.GetDefaultPrim().GetPath().AppendPath("Links/a/Collisions/c")
        )
        assert prim.HasAPI(UsdPhysics.CollisionAPI)
        assert UsdGeom.Imageable(prim).GetPurposeAttr().Get() == "guide"

    def test_mesh_collisions_get_an_approximation(self):
        schema = self._schema(
            geometry=Geometry(type=GeometryType.MESH, filename="m.obj")
        )
        stage = open_exported(schema)
        prim = stage.GetPrimAtPath(
            stage.GetDefaultPrim().GetPath().AppendPath("Links/a/Collisions/c")
        )
        assert prim.HasAPI(UsdPhysics.MeshCollisionAPI)
        assert (
            UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
            == "convexHull"
        )

    def test_approximation_can_be_overridden_via_extensions(self):
        schema = self._schema(
            geometry=Geometry(type=GeometryType.MESH, filename="m.obj"),
            extensions={"approximation": "convexDecomposition"},
        )
        assert 'physics:approximation = "convexDecomposition"' in export(schema)

    def test_none_friction_is_not_authored(self):
        text = export(self._schema())
        assert "physics:staticFriction" not in text
        assert t.A_COLLISION_MU_STATIC not in text

    def test_physics_materials_are_deduplicated(self):
        schema = CommonSchema(
            metadata=Metadata(name="r"),
            links=[
                Link(
                    name="a",
                    collisions=[
                        Collision(name="c1", geometry=sphere(), mu_static=0.8),
                        Collision(name="c2", geometry=sphere(), mu_static=0.8),
                        Collision(name="c3", geometry=sphere(), mu_static=0.5),
                    ],
                )
            ],
        )
        stage = open_exported(schema)
        scope = stage.GetPrimAtPath(
            stage.GetDefaultPrim().GetPath().AppendPath("PhysicsMaterials")
        )
        assert len(list(scope.GetChildren())) == 2

    def test_collision_material_is_bound_with_the_physics_purpose(self):
        stage = open_exported(self._schema(mu_static=0.8, mu_dynamic=0.7))
        prim = stage.GetPrimAtPath(
            stage.GetDefaultPrim().GetPath().AppendPath("Links/a/Collisions/c")
        )
        bound = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")[0]
        assert bound
        api = UsdPhysics.MaterialAPI(bound.GetPrim())
        assert math.isclose(api.GetStaticFrictionAttr().Get(), 0.8, abs_tol=1e-6)


class TestCollisionConfig:
    def test_filtered_groups_are_derived_from_the_bitmasks(self):
        # ROBOT and ENV collide with each other but not with themselves; SOLO shares
        # no bits with either, so it must be filtered against both.
        schema = CommonSchema(
            metadata=Metadata(name="r"),
            links=[Link(name="a")],
            collision_config=CollisionConfig(
                groups={
                    "ROBOT": CollisionGroup(name="ROBOT", contype=2, conaffinity=4),
                    "ENV": CollisionGroup(name="ENV", contype=4, conaffinity=2),
                    "SOLO": CollisionGroup(name="SOLO", contype=8, conaffinity=8),
                }
            ),
        )
        stage = open_exported(schema)
        groups = stage.GetDefaultPrim().GetPath().AppendPath("CollisionGroups")

        def filtered(name: str) -> set[str]:
            group = UsdPhysics.CollisionGroup(
                stage.GetPrimAtPath(groups.AppendChild(name))
            )
            return {path.name for path in group.GetFilteredGroupsRel().GetTargets()}

        assert filtered("ROBOT") == {"ROBOT", "SOLO"}
        assert filtered("ENV") == {"ENV", "SOLO"}
        assert filtered("SOLO") == {"ROBOT", "ENV"}

    def test_group_membership_is_recorded_in_the_collection(self):
        schema = CommonSchema(
            metadata=Metadata(name="r"),
            links=[
                Link(
                    name="a",
                    collisions=[Collision(name="c", geometry=sphere(), group="ROBOT")],
                )
            ],
            collision_config=CollisionConfig(
                groups={"ROBOT": CollisionGroup(name="ROBOT", contype=2, conaffinity=2)}
            ),
        )
        stage = open_exported(schema)
        group = UsdPhysics.CollisionGroup(
            stage.GetPrimAtPath(
                stage.GetDefaultPrim()
                .GetPath()
                .AppendPath("CollisionGroups/ROBOT")
            )
        )
        targets = group.GetCollidersCollectionAPI().GetIncludesRel().GetTargets()
        assert [path.name for path in targets] == ["c"]

    def test_excludes_apply_filtered_pairs_to_the_first_body(self):
        schema = CommonSchema(
            metadata=Metadata(name="r"),
            links=[Link(name="a"), Link(name="b")],
            collision_config=CollisionConfig(
                excludes=[CollisionExclude(body1="a", body2="b")]
            ),
        )
        stage = open_exported(schema)
        link = stage.GetPrimAtPath(
            stage.GetDefaultPrim().GetPath().AppendPath("Links/a")
        )
        assert link.HasAPI(UsdPhysics.FilteredPairsAPI)
        targets = UsdPhysics.FilteredPairsAPI(link).GetFilteredPairsRel().GetTargets()
        assert [path.name for path in targets] == ["b"]


class TestActuatorsAndDrives:
    def test_actuator_writes_a_drive_and_a_record_prim(self):
        schema = two_link_schema(home_position=math.pi / 4)
        schema.actuators = [
            Actuator(
                name="m1",
                joint="j",
                type=ActuatorType.SERVO,
                kp=100.0,
                kd=5.0,
                max_torque=20.0,
                gear_ratio=101.0,
                ki=0.5,
            )
        ]
        stage = open_exported(schema)
        root = stage.GetDefaultPrim().GetPath()
        joint = stage.GetPrimAtPath(root.AppendPath("Joints/j"))
        drive = UsdPhysics.DriveAPI(joint, "angular")
        # UsdPhysics angular gains are per degree; the schema's are per radian.
        assert math.isclose(
            drive.GetStiffnessAttr().Get(), math.radians(100.0), rel_tol=1e-6
        )
        assert math.isclose(drive.GetDampingAttr().Get(), math.radians(5.0), rel_tol=1e-6)
        assert math.isclose(drive.GetMaxForceAttr().Get(), 20.0, abs_tol=1e-4)
        assert math.isclose(drive.GetTargetPositionAttr().Get(), 45.0, abs_tol=1e-3)
        assert drive.GetTypeAttr().Get() == "force"

        # UsdPhysicsDriveAPI has no gear ratio or integral gain, so the full record
        # prim is what makes the actuator round-trip.
        record = stage.GetPrimAtPath(root.AppendPath("Actuators/m1"))
        assert record.IsValid()
        assert record.GetAttribute(t.A_ACTUATOR_GEAR_RATIO).Get() == 101.0
        assert record.GetAttribute(t.A_ACTUATOR_KI).Get() == 0.5

    def test_prismatic_actuator_uses_the_linear_drive_instance(self):
        schema = two_link_schema(type=JointType.PRISMATIC)
        schema.actuators = [
            Actuator(name="m1", joint="j", type=ActuatorType.POSITION, kp=10.0)
        ]
        stage = open_exported(schema)
        joint = stage.GetPrimAtPath(
            stage.GetDefaultPrim().GetPath().AppendPath("Joints/j")
        )
        assert "PhysicsDriveAPI:linear" in joint.GetAppliedSchemas()

    def test_passive_spring_becomes_a_drive_when_there_is_no_actuator(self):
        schema = two_link_schema(
            dynamics=JointDynamics(damping=0.5, spring_stiffness=7.5, spring_reference=0.25)
        )
        stage = open_exported(schema)
        joint = stage.GetPrimAtPath(
            stage.GetDefaultPrim().GetPath().AppendPath("Joints/j")
        )
        drive = UsdPhysics.DriveAPI(joint, "angular")
        assert math.isclose(drive.GetStiffnessAttr().Get(), math.radians(7.5), rel_tol=1e-6)
        assert math.isclose(drive.GetDampingAttr().Get(), math.radians(0.5), rel_tol=1e-6)

    def test_an_actuator_wins_over_the_passive_spring(self):
        schema = two_link_schema(dynamics=JointDynamics(spring_stiffness=7.5))
        schema.actuators = [
            Actuator(name="m1", joint="j", type=ActuatorType.SERVO, kp=100.0)
        ]
        stage = open_exported(schema)
        joint = stage.GetPrimAtPath(
            stage.GetDefaultPrim().GetPath().AppendPath("Joints/j")
        )
        assert math.isclose(
            UsdPhysics.DriveAPI(joint, "angular").GetStiffnessAttr().Get(),
            math.radians(100.0),
            rel_tol=1e-6,
        )

    def test_prismatic_gains_are_already_si(self):
        schema = two_link_schema(type=JointType.PRISMATIC)
        schema.actuators = [
            Actuator(name="m1", joint="j", type=ActuatorType.POSITION, kp=10.0, kd=2.0)
        ]
        stage = open_exported(schema)
        joint = stage.GetPrimAtPath(
            stage.GetDefaultPrim().GetPath().AppendPath("Joints/j")
        )
        drive = UsdPhysics.DriveAPI(joint, "linear")
        assert math.isclose(drive.GetStiffnessAttr().Get(), 10.0, abs_tol=1e-6)
        assert math.isclose(drive.GetDampingAttr().Get(), 2.0, abs_tol=1e-6)

    def test_velocity_actuator_is_a_pure_damping_drive(self):
        schema = two_link_schema(home_position=0.5)
        schema.actuators = [
            Actuator(name="m1", joint="j", type=ActuatorType.VELOCITY, kp=3.0, kd=4.0)
        ]
        stage = open_exported(schema)
        joint = stage.GetPrimAtPath(
            stage.GetDefaultPrim().GetPath().AppendPath("Joints/j")
        )
        drive = UsdPhysics.DriveAPI(joint, "angular")
        assert not drive.GetStiffnessAttr().HasAuthoredValue()
        assert not drive.GetTargetPositionAttr().HasAuthoredValue()
        assert math.isclose(drive.GetDampingAttr().Get(), math.radians(4.0), rel_tol=1e-6)
        assert drive.GetTargetVelocityAttr().Get() == 0.0
        assert joint.GetCustomDataByKey(t.CD_CONTROL_MODE) == "velocity"

    def test_motor_is_an_effort_only_drive(self):
        schema = two_link_schema()
        schema.actuators = [
            Actuator(
                name="m1", joint="j", type=ActuatorType.TORQUE, kp=3.0, max_torque=9.0
            )
        ]
        stage = open_exported(schema)
        joint = stage.GetPrimAtPath(
            stage.GetDefaultPrim().GetPath().AppendPath("Joints/j")
        )
        drive = UsdPhysics.DriveAPI(joint, "angular")
        assert not drive.GetStiffnessAttr().HasAuthoredValue()
        assert not drive.GetDampingAttr().HasAuthoredValue()
        assert math.isclose(drive.GetMaxForceAttr().Get(), 9.0, abs_tol=1e-6)
        assert joint.GetCustomDataByKey(t.CD_CONTROL_MODE) == "effort"
        assert joint.GetCustomDataByKey(t.CD_ACTUATOR_NAME) == "m1"

    def test_zero_dynamics_does_not_author_a_drive(self):
        schema = two_link_schema(dynamics=JointDynamics())
        joint_text = export(schema)
        assert "drive:angular" not in joint_text


class TestMaterials:
    def test_named_material_becomes_a_shared_preview_surface(self):
        schema = CommonSchema(
            metadata=Metadata(name="r"),
            links=[
                Link(
                    name="a",
                    visuals=[
                        Visual(
                            name="v",
                            geometry=sphere(),
                            material=Material(
                                name="red",
                                color=[1.0, 0.0, 0.0, 0.5],
                                specular=[0.2, 0.2, 0.2],
                                emissive=[0.01, 0.0, 0.0],
                                shininess=0.75,
                            ),
                        )
                    ],
                )
            ],
        )
        text = export(schema)
        assert 'info:id = "UsdPreviewSurface"' in text
        assert "inputs:diffuseColor = (1, 0, 0)" in text
        assert "float inputs:opacity = 0.5" in text
        assert "inputs:specularColor" in text
        assert "inputs:emissiveColor" in text
        # UsdPreviewSurface has roughness, not shininess; 1 - 0.75 = 0.25.
        assert "float inputs:roughness = 0.25" in text

        stage = open_exported(schema)
        prim = stage.GetPrimAtPath(
            stage.GetDefaultPrim().GetPath().AppendPath("Links/a/Visuals/v")
        )
        bound = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
        assert bound.GetPrim().GetName() == "red"

    def test_texture_authors_a_uv_reader_chain(self):
        schema = CommonSchema(
            metadata=Metadata(name="r"),
            links=[
                Link(
                    name="a",
                    visuals=[
                        Visual(
                            name="v",
                            geometry=sphere(),
                            material=Material(name="m", texture="tex/wood.png"),
                        )
                    ],
                )
            ],
        )
        text = export(schema)
        assert 'info:id = "UsdUVTexture"' in text
        assert 'info:id = "UsdPrimvarReader_float2"' in text
        # The native shader input stays an asset — there, usdz localization
        # rewriting the path to the bundled copy is exactly right.
        assert "inputs:file = @tex/wood.png@" in text

    def test_redefined_material_name_is_authored_inline_and_warns(self, caplog):
        import logging

        schema = CommonSchema(
            metadata=Metadata(name="r"),
            links=[
                Link(
                    name="a",
                    visuals=[
                        Visual(
                            name="v1",
                            geometry=sphere(),
                            material=Material(name="paint", color=[1, 0, 0, 1]),
                        ),
                        Visual(
                            name="v2",
                            geometry=sphere(),
                            material=Material(name="paint", color=[0, 1, 0, 1]),
                        ),
                        Visual(
                            name="v3",
                            geometry=sphere(),
                            material=Material(name="paint", color=[0, 0, 1, 1]),
                        ),
                    ],
                )
            ],
        )
        with caplog.at_level(logging.WARNING):
            stage = open_exported(schema)
        # Once per name, not once per visual: a merged scene collides names in bulk.
        warnings = [r for r in caplog.records if "defined more than once" in r.message]
        assert len(warnings) == 1
        assert "paint" in warnings[0].getMessage()

        root = stage.GetDefaultPrim().GetPath()
        # The first definition still owns the shared prim, so the common case is
        # unchanged; only the redefinition goes inline, with a name sidecar so it
        # still imports as "paint".
        first = stage.GetPrimAtPath(root.AppendPath("Links/a/Visuals/v1"))
        second = stage.GetPrimAtPath(root.AppendPath("Links/a/Visuals/v2"))
        assert (
            UsdShade.MaterialBindingAPI(first).ComputeBoundMaterial()[0].GetPath()
            == root.AppendPath("Materials/paint")
        )
        third = stage.GetPrimAtPath(root.AppendPath("Links/a/Visuals/v3"))
        for shadowed in (second, third):
            binding = UsdShade.MaterialBindingAPI(shadowed)
            inline = binding.ComputeBoundMaterial()[0].GetPrim()
            assert inline.GetPath() == shadowed.GetPath().AppendChild("Material")
            assert inline.GetAttribute("cyberwave:name").Get() == "paint"

    def test_unnamed_material_is_authored_inline_under_its_gprim(self):
        schema = CommonSchema(
            metadata=Metadata(name="r"),
            links=[
                Link(
                    name="a",
                    visuals=[
                        Visual(
                            name="v", geometry=sphere(), material=Material(color=[0, 1, 0, 1])
                        )
                    ],
                )
            ],
        )
        stage = open_exported(schema)
        root = stage.GetDefaultPrim().GetPath()
        assert stage.GetPrimAtPath(root.AppendPath("Links/a/Visuals/v/Material")).IsValid()
        assert not stage.GetPrimAtPath(root.AppendPath("Materials")).IsValid()


class TestSensorsAndScopes:
    def test_sensors_live_under_their_link_in_a_dedicated_scope(self):
        schema = CommonSchema(
            metadata=Metadata(name="r"),
            links=[Link(name="a")],
            sensors=[Sensor(name="imu", type="IMU", parent_link="a")],
        )
        stage = open_exported(schema)
        root = stage.GetDefaultPrim().GetPath()
        assert stage.GetPrimAtPath(root.AppendPath("Links/a/Sensors/imu")).IsValid()

    def test_world_parented_sensors_get_a_top_level_scope(self):
        schema = CommonSchema(
            metadata=Metadata(name="r"),
            links=[Link(name="a")],
            sensors=[Sensor(name="cam", type="camera", parent_link="world")],
        )
        stage = open_exported(schema)
        root = stage.GetDefaultPrim().GetPath()
        assert stage.GetPrimAtPath(root.AppendPath("Sensors/cam")).IsValid()

    def test_a_sensor_named_like_a_scope_does_not_collide(self):
        # The dedicated Sensors scope is what keeps this from clobbering the
        # Visuals scope on the same link.
        schema = CommonSchema(
            metadata=Metadata(name="r"),
            links=[Link(name="a", visuals=[Visual(name="v", geometry=sphere())])],
            sensors=[Sensor(name="Visuals", type="IMU", parent_link="a")],
        )
        stage = open_exported(schema)
        root = stage.GetDefaultPrim().GetPath()
        assert stage.GetPrimAtPath(root.AppendPath("Links/a/Visuals/v")).IsValid()
        assert stage.GetPrimAtPath(root.AppendPath("Links/a/Sensors/Visuals")).IsValid()

    def test_sensor_noise_and_parameters_are_json_strings(self):
        schema = CommonSchema(
            metadata=Metadata(name="r"),
            links=[Link(name="a")],
            sensors=[
                Sensor(
                    name="imu",
                    type="IMU",
                    parent_link="a",
                    noise={"stddev": 0.001, "bias": [0.1, 0.2]},
                    parameters={"frame": "body"},
                )
            ],
        )
        text = export(schema)
        assert t.A_SENSOR_NOISE in text
        assert '"stddev":0.001' in text


class TestPhysicsScene:
    def test_physics_scene_is_a_sibling_of_the_robot_prim(self):
        # Keeping it outside defaultPrim means referencing the asset into a larger
        # stage does not drag a second gravity definition along.
        stage = open_exported(
            CommonSchema(
                metadata=Metadata(name="r"), links=[Link(name="a")], physics=Physics()
            )
        )
        scene = stage.GetPrimAtPath("/PhysicsScene")
        assert scene.IsValid()
        assert scene.GetParent() == stage.GetPseudoRoot()

    def test_gravity_is_split_into_direction_and_magnitude(self):
        stage = open_exported(
            CommonSchema(
                metadata=Metadata(name="r"),
                physics=Physics(gravity=Vector3(z=-9.81)),
            )
        )
        scene = UsdPhysics.Scene(stage.GetPrimAtPath("/PhysicsScene"))
        assert math.isclose(scene.GetGravityMagnitudeAttr().Get(), 9.81, abs_tol=1e-5)
        assert np.allclose(list(scene.GetGravityDirectionAttr().Get()), [0, 0, -1])

    def test_a_robot_named_physicsscene_keeps_its_own_prim_type(self):
        # Defining the scene at the canonical path would *redefine* the articulation
        # root as a UsdPhysicsScene, corrupting the root's type and merging
        # world-level physics into the referenceable subtree.
        stage = open_exported(
            CommonSchema(
                metadata=Metadata(name="PhysicsScene"),
                links=[Link(name="a")],
                physics=Physics(),
            )
        )
        root = stage.GetPrimAtPath("/PhysicsScene")
        assert root == stage.GetDefaultPrim()
        assert root.GetTypeName() == "Xform"
        scene = stage.GetPrimAtPath("/PhysicsScene_2")
        assert scene.IsValid()
        assert UsdPhysics.Scene(scene)
        # And the parser still finds it, so the round trip is unaffected.
        schema = USDParser().parse_string(export(
            CommonSchema(
                metadata=Metadata(name="PhysicsScene"),
                links=[Link(name="a")],
                physics=Physics(),
            )
        ))
        assert schema.physics is not None
        assert schema.metadata.name == "PhysicsScene"

    def test_zero_gravity_does_not_author_the_native_sentinels(self):
        # (0,0,0) means "use the stage up axis" and -inf means "earth gravity", so a
        # genuine zero-g schema must rely on the sidecar instead.
        stage = open_exported(
            CommonSchema(metadata=Metadata(name="r"), physics=Physics(gravity=Vector3()))
        )
        scene = UsdPhysics.Scene(stage.GetPrimAtPath("/PhysicsScene"))
        assert not scene.GetGravityMagnitudeAttr().HasAuthoredValue()
        assert (
            stage.GetPrimAtPath("/PhysicsScene").GetAttribute(t.A_PHYSICS_GRAVITY).Get()
            is not None
        )


class TestDeterminismAndPurity:
    def test_export_is_byte_identical_across_calls(self):
        schema = two_link_schema()
        assert export(schema) == export(schema)

    def test_export_does_not_mutate_the_schema(self):
        schema = two_link_schema(limits=JointLimits(lower=-1.0, upper=1.0))
        before = schema.to_dict()
        export(schema)
        assert schema.to_dict() == before

    def test_exporter_instance_is_reusable(self):
        exporter = USDExporter()
        first = exporter.export_to_string(two_link_schema())
        second = exporter.export_to_string(two_link_schema())
        assert first == second

    def test_dataclass_in_extensions_is_serialized_via_cattrs(self):
        schema = two_link_schema()
        schema.extensions["parse_context"] = {
            "materials": {"red": Material(name="red", color=[1.0, 0, 0, 1])}
        }
        assert '"name":"red"' in export(schema)

    def test_writes_usda_usdc_and_usdz_files(self, tmp_path):
        schema = two_link_schema()
        for suffix in (".usda", ".usdc", ".usd", ".usdz"):
            path = tmp_path / f"r{suffix}"
            USDExporter().export(schema, path)
            assert path.stat().st_size > 0
            assert Usd.Stage.Open(str(path)) is not None

    def test_export_creates_missing_parent_directories(self, tmp_path):
        path = tmp_path / "nested" / "deeper" / "r.usda"
        USDExporter().export(two_link_schema(), path)
        assert path.exists()

    def test_get_extension_is_the_human_readable_encoding(self):
        assert USDExporter().get_extension() == "usda"


class TestUsdPhysicsReferenceInterpretation:
    """What USD's *own* physics loader makes of the stage.

    The strongest available check that the export is valid, simulatable USD rather
    than merely something our parser can read back: ``UsdPhysicsLoadStageFromPrimRange``
    is the reference implementation a physics engine uses to interpret a stage, so
    its counts have to match the source schema exactly.
    """

    def test_reference_parser_sees_the_whole_articulation(self):
        from collections import Counter

        from cyberwave_robot_format.urdf import URDFParser

        schema = URDFParser().parse(
            Path(__file__).parent / "fixtures" / "piper_description.urdf"
        )
        schema.extensions.pop("parse_context", None)
        schema.physics = Physics()

        stage = open_exported(schema)
        # A single root path: passing every prim path would report each body once per
        # ancestor subtree that contains it.
        loaded = UsdPhysics.UsdPhysicsLoadStageFromPrimRange(
            stage, [Sdf.Path("/")]
        )

        def count(object_type: UsdPhysics.ObjectType) -> int:
            # Keys are ObjectType enum members, not strings, and each value is a
            # (paths, descriptors) pair.
            entry = loaded.get(object_type)
            return 0 if entry is None else len(entry[0])

        joint_counts = Counter(joint.type.value for joint in schema.joints)
        assert count(UsdPhysics.ObjectType.RigidBody) == len(schema.links)
        assert count(UsdPhysics.ObjectType.RevoluteJoint) == joint_counts["revolute"]
        assert count(UsdPhysics.ObjectType.PrismaticJoint) == joint_counts["prismatic"]
        assert count(UsdPhysics.ObjectType.FixedJoint) == joint_counts["fixed"]
        assert count(UsdPhysics.ObjectType.Articulation) == 1
        assert count(UsdPhysics.ObjectType.Scene) == 1
        # Only collision geometry carries PhysicsCollisionAPI, so the visual meshes
        # must not show up as collision shapes.
        collision_meshes = sum(
            1
            for link in schema.links
            for collision in link.collisions
            if collision.geometry and collision.geometry.type == GeometryType.MESH
        )
        assert count(UsdPhysics.ObjectType.MeshShape) == collision_meshes


class TestStructurallyInvalidInput:
    def test_dangling_parent_link_warns_and_still_exports(self, caplog):
        import logging

        schema = CommonSchema(
            metadata=Metadata(name="r"),
            links=[Link(name="a")],
            joints=[
                Joint(name="j", type=JointType.FIXED, parent_link="ghost", child_link="a")
            ],
        )
        with caplog.at_level(logging.WARNING):
            text = export(schema)
        assert "unknown parent link" in caplog.text
        assert 'ghost"' in text

    def test_empty_schema_exports(self):
        text = export(CommonSchema(metadata=Metadata(name="nothing")))
        assert 'defaultPrim = "nothing"' in text
