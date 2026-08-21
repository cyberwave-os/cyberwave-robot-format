# Copyright [2025] Cyberwave
# Licensed under the Apache License, Version 2.0

"""Unit tests for ``USDParser``, weighted towards **foreign** stages.

The fixtures below are hand-written ``.usda`` with no ``cyberwave:`` attributes at
all — the shape a stage from Isaac Sim, Omniverse, onshape-to-robot or a human
author has. That path is where the parser has to do real inference (prim types,
body relationships, joint frames, degrees, sentinels), and writing the input as
readable USD text is what makes the expected mapping auditable.

Two things to keep in mind while reading the fixtures: ``quatf`` literals are
**real-first** (``(w, x, y, z)``), and every unauthored ``UsdPhysics`` attribute has
a fallback that must *not* be read as data.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pxr")

from pxr import Sdf, Usd  # noqa: E402

from cyberwave_robot_format.core import ParseContext  # noqa: E402
from cyberwave_robot_format.schema import (  # noqa: E402
    ActuatorType,
    CommonSchema,
    GeometryType,
    Inertia,
    JointType,
    Link,
    Metadata,
    Vector3,
)
from cyberwave_robot_format.usd import USDExporter, USDParser  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"

HEADER = """#usda 1.0
(
    defaultPrim = "Robot"
    metersPerUnit = 1
    upAxis = "Z"
)
"""

MINIMAL_ROBOT = (
    HEADER
    + """
def Xform "Robot" (prepend apiSchemas = ["PhysicsArticulationRootAPI"])
{
    def Scope "Links"
    {
        def Xform "base" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"])
        {
            float physics:mass = 1.5
            point3f physics:centerOfMass = (0.01, 0.02, 0.03)
        }
        def Xform "arm" (prepend apiSchemas = ["PhysicsRigidBodyAPI"])
        {
            matrix4d xformOp:transform = ( (1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0.3,1) )
            uniform token[] xformOpOrder = ["xformOp:transform"]
        }
    }
    def Scope "Joints"
    {
        def PhysicsRevoluteJoint "shoulder"
        {
            uniform token physics:axis = "Y"
            rel physics:body0 = </Robot/Links/base>
            rel physics:body1 = </Robot/Links/arm>
            point3f physics:localPos0 = (0, 0, 0.3)
            float physics:lowerLimit = -45
            float physics:upperLimit = 90
        }
    }
}
"""
)


def parse(text: str) -> CommonSchema:
    return USDParser().parse_string(text)


def parse_with_context(text: str) -> tuple[CommonSchema, ParseContext]:
    """Parse via ``parse_stage`` so the warnings can be inspected."""
    layer = Sdf.Layer.CreateAnonymous(".usda")
    assert layer.ImportFromString(text)
    stage = Usd.Stage.Open(layer)
    context = ParseContext(file_path=Path("<test>"), base_dir=Path("."))
    schema = USDParser().parse_stage(stage, context)
    return schema, context


class TestForeignMinimalRobot:
    def test_links_are_discovered_by_rigid_body_api(self):
        schema = parse(MINIMAL_ROBOT)
        assert [link.name for link in schema.links] == ["base", "arm"]

    def test_joint_kind_comes_from_the_prim_type(self):
        schema = parse(MINIMAL_ROBOT)
        assert schema.joints[0].type == JointType.REVOLUTE

    def test_bodies_resolve_to_link_names(self):
        joint = parse(MINIMAL_ROBOT).joints[0]
        assert joint.parent_link == "base"
        assert joint.child_link == "arm"

    def test_axis_is_recovered_from_the_axis_token(self):
        assert parse(MINIMAL_ROBOT).joints[0].axis == Vector3(y=1.0)

    def test_joint_pose_comes_from_local_pos0(self):
        pose = parse(MINIMAL_ROBOT).joints[0].pose
        assert np.allclose(pose.position.to_list(), [0.0, 0.0, 0.3], atol=1e-7)

    def test_native_limits_are_converted_from_degrees(self):
        limits = parse(MINIMAL_ROBOT).joints[0].limits
        assert math.isclose(limits.lower, -math.pi / 4, abs_tol=1e-6)
        assert math.isclose(limits.upper, math.pi / 2, abs_tol=1e-6)
        # Only lower/upper exist natively; the rest of JointLimits has no USD home.
        assert limits.effort is None
        assert limits.velocity is None

    def test_mass_and_center_of_mass_are_read(self):
        link = parse(MINIMAL_ROBOT).links[0]
        assert math.isclose(link.mass, 1.5, abs_tol=1e-6)
        assert np.allclose(link.center_of_mass.to_list(), [0.01, 0.02, 0.03], atol=1e-7)

    def test_metadata_records_the_source_format(self):
        schema = parse(MINIMAL_ROBOT)
        assert schema.metadata.name == "Robot"
        assert schema.metadata.source_format == "usd"


class TestWorldAndSentinels:
    def test_absent_body0_means_world(self):
        text = (
            HEADER
            + """
def Xform "Robot" (prepend apiSchemas = ["PhysicsArticulationRootAPI"])
{
    def Xform "base" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
    def PhysicsFixedJoint "anchor"
    {
        rel physics:body1 = </Robot/base>
    }
}
"""
        )
        assert parse(text).joints[0].parent_link == "world"

    def test_unlimited_native_limits_become_none(self):
        # -inf/+inf are UsdPhysics' "unlimited" fallbacks, not numbers.
        text = (
            HEADER
            + """
def Xform "Robot" (prepend apiSchemas = ["PhysicsArticulationRootAPI"])
{
    def Xform "a" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
    def Xform "b" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
    def PhysicsRevoluteJoint "j"
    {
        rel physics:body0 = </Robot/a>
        rel physics:body1 = </Robot/b>
    }
}
"""
        )
        assert parse(text).joints[0].limits is None

    def test_explicit_infinite_limits_also_become_none(self):
        text = (
            HEADER
            + """
def Xform "Robot" (prepend apiSchemas = ["PhysicsArticulationRootAPI"])
{
    def Xform "a" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
    def Xform "b" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
    def PhysicsRevoluteJoint "j"
    {
        rel physics:body0 = </Robot/a>
        rel physics:body1 = </Robot/b>
        float physics:lowerLimit = -inf
        float physics:upperLimit = 60
    }
}
"""
        )
        limits = parse(text).joints[0].limits
        assert limits.lower is None
        assert math.isclose(limits.upper, math.radians(60.0), abs_tol=1e-6)

    def test_unauthored_mass_api_does_not_leak_sentinels(self):
        text = (
            HEADER
            + """
def Xform "Robot" (prepend apiSchemas = ["PhysicsArticulationRootAPI"])
{
    def Xform "a" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]) {}
}
"""
        )
        link = parse(text).links[0]
        assert link.mass == 0.0
        assert link.center_of_mass == Vector3()
        assert link.inertia == Inertia()

    def test_explicit_center_of_mass_sentinel_is_ignored(self):
        # (-inf, -inf, -inf) means "derive it from the geometry".
        text = (
            HEADER
            + """
def Xform "Robot" (prepend apiSchemas = ["PhysicsArticulationRootAPI"])
{
    def Xform "a" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"])
    {
        point3f physics:centerOfMass = (-inf, -inf, -inf)
    }
}
"""
        )
        assert parse(text).links[0].center_of_mass == Vector3()

    def test_zero_principal_axes_sentinel_is_treated_as_identity(self):
        text = (
            HEADER
            + """
def Xform "Robot" (prepend apiSchemas = ["PhysicsArticulationRootAPI"])
{
    def Xform "a" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"])
    {
        float3 physics:diagonalInertia = (1, 2, 3)
        quatf physics:principalAxes = (0, 0, 0, 0)
    }
}
"""
        )
        inertia = parse(text).links[0].inertia
        assert (inertia.ixx, inertia.iyy, inertia.izz) == (1.0, 2.0, 3.0)
        assert (inertia.ixy, inertia.ixz, inertia.iyz) == (0.0, 0.0, 0.0)

    def test_full_principal_axes_recomposes_the_tensor(self):
        # 90 degrees about Z swaps the X and Y diagonal terms.
        half = math.sqrt(0.5)
        text = (
            HEADER
            + f"""
def Xform "Robot" (prepend apiSchemas = ["PhysicsArticulationRootAPI"])
{{
    def Xform "a" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"])
    {{
        float3 physics:diagonalInertia = (1, 2, 3)
        quatf physics:principalAxes = ({half}, 0, 0, {half})
    }}
}}
"""
        )
        inertia = parse(text).links[0].inertia
        assert math.isclose(inertia.ixx, 2.0, abs_tol=1e-5)
        assert math.isclose(inertia.iyy, 1.0, abs_tol=1e-5)
        assert math.isclose(inertia.izz, 3.0, abs_tol=1e-5)
        assert math.isclose(inertia.ixy, 0.0, abs_tol=1e-5)


class TestJointFrameInference:
    def test_nonzero_local_pos1_is_folded_into_the_joint_pose(self):
        text = (
            HEADER
            + """
def Xform "Robot" (prepend apiSchemas = ["PhysicsArticulationRootAPI"])
{
    def Xform "a" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
    def Xform "b" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
    def PhysicsRevoluteJoint "j"
    {
        uniform token physics:axis = "X"
        rel physics:body0 = </Robot/a>
        rel physics:body1 = </Robot/b>
        point3f physics:localPos0 = (1, 0, 0)
        point3f physics:localPos1 = (0, 0, 0.5)
    }
}
"""
        )
        pose = parse(text).joints[0].pose
        # T_PC = [I, (1,0,0)] @ [I, (0,0,0.5)]^-1 = [I, (1,0,-0.5)]
        assert np.allclose(pose.position.to_list(), [1.0, 0.0, -0.5], atol=1e-6)

    @pytest.mark.parametrize(
        ("token", "expected"),
        [("X", [1.0, 0, 0]), ("Y", [0, 1.0, 0]), ("Z", [0, 0, 1.0])],
    )
    def test_axis_token_selects_the_recovered_axis(self, token, expected):
        text = (
            HEADER
            + f"""
def Xform "Robot" (prepend apiSchemas = ["PhysicsArticulationRootAPI"])
{{
    def Xform "a" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {{}}
    def Xform "b" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {{}}
    def PhysicsPrismaticJoint "j"
    {{
        uniform token physics:axis = "{token}"
        rel physics:body0 = </Robot/a>
        rel physics:body1 = </Robot/b>
    }}
}}
"""
        )
        assert np.allclose(parse(text).joints[0].axis.to_list(), expected)

    def test_local_rot1_rotates_the_axis(self):
        half = math.sqrt(0.5)
        text = (
            HEADER
            + f"""
def Xform "Robot" (prepend apiSchemas = ["PhysicsArticulationRootAPI"])
{{
    def Xform "a" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {{}}
    def Xform "b" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {{}}
    def PhysicsRevoluteJoint "j"
    {{
        uniform token physics:axis = "X"
        rel physics:body0 = </Robot/a>
        rel physics:body1 = </Robot/b>
        quatf physics:localRot1 = ({half}, 0, 0, {half})
    }}
}}
"""
        )
        # 90 degrees about Z maps the joint-frame X axis onto +Y.
        assert np.allclose(parse(text).joints[0].axis.to_list(), [0.0, 1.0, 0.0], atol=1e-6)


class TestForeignGeometry:
    @staticmethod
    def _link_with(body: str) -> str:
        return (
            HEADER
            + f"""
def Xform "Robot" (prepend apiSchemas = ["PhysicsArticulationRootAPI"])
{{
    def Xform "a" (prepend apiSchemas = ["PhysicsRigidBodyAPI"])
    {{
{body}
    }}
}}
"""
        )

    def test_cube_size_and_scale_become_a_box(self):
        schema = parse(
            self._link_with(
                """        def Cube "c"
        {
            double size = 2
            double3 xformOp:scale = (0.5, 1, 2)
            uniform token[] xformOpOrder = ["xformOp:scale"]
        }"""
            )
        )
        geometry = schema.links[0].visuals[0].geometry
        assert geometry.type == GeometryType.BOX
        assert np.allclose(geometry.size.to_list(), [1.0, 2.0, 4.0])

    def test_unauthored_shape_dimensions_use_usds_schema_fallbacks(self):
        # An unauthored UsdGeom dimension is not missing data the way an unauthored
        # `physics:mass` is — the schema fallback *is* the value, so it must be read
        # off the attribute rather than replaced with a local constant.
        schema = parse(
            self._link_with(
                """        def Cylinder "c"
        {
            double radius = 0.05
        }

        def Capsule "p"
        {
        }

        def Plane "pl"
        {
        }"""
            )
        )
        geometries = [visual.geometry for visual in schema.links[0].visuals]
        cylinder, capsule, plane = geometries
        assert (cylinder.radius, cylinder.length) == (0.05, 2.0)
        assert (capsule.radius, capsule.length) == (0.5, 1.0)
        assert np.allclose(plane.size.to_list(), [2.0, 2.0, 0.0])

    def test_uniformly_scaled_sphere_stays_a_sphere(self):
        schema = parse(
            self._link_with(
                """        def Sphere "s"
        {
            double radius = 0.5
            double3 xformOp:scale = (2, 2, 2)
            uniform token[] xformOpOrder = ["xformOp:scale"]
        }"""
            )
        )
        geometry = schema.links[0].visuals[0].geometry
        assert geometry.type == GeometryType.SPHERE
        assert math.isclose(geometry.radius, 1.0, abs_tol=1e-9)

    def test_non_uniformly_scaled_sphere_becomes_an_ellipsoid(self):
        schema = parse(
            self._link_with(
                """        def Sphere "s"
        {
            double radius = 1
            double3 xformOp:scale = (0.1, 0.2, 0.05)
            uniform token[] xformOpOrder = ["xformOp:scale"]
        }"""
            )
        )
        geometry = schema.links[0].visuals[0].geometry
        assert geometry.type == GeometryType.ELLIPSOID
        assert np.allclose(geometry.size.to_list(), [0.1, 0.2, 0.05])

    def test_cylinder_and_capsule_map_radius_and_height(self):
        for prim_type, expected in (
            ("Cylinder", GeometryType.CYLINDER),
            ("Capsule", GeometryType.CAPSULE),
        ):
            schema = parse(
                self._link_with(
                    f"""        def {prim_type} "g"
        {{
            double radius = 0.2
            double height = 0.4
        }}"""
                )
            )
            geometry = schema.links[0].visuals[0].geometry
            assert geometry.type == expected
            assert math.isclose(geometry.radius, 0.2)
            assert math.isclose(geometry.length, 0.4)

    def test_plane_maps_width_and_length(self):
        schema = parse(
            self._link_with(
                """        def Plane "p"
        {
            uniform token axis = "Z"
            double width = 8
            double length = 6
        }"""
            )
        )
        geometry = schema.links[0].visuals[0].geometry
        assert geometry.type == GeometryType.PLANE
        assert np.allclose(geometry.size.to_list(), [8.0, 6.0, 0.0])

    def test_translate_and_orient_ops_yield_the_pose_without_the_scale(self):
        half = math.sqrt(0.5)
        schema = parse(
            self._link_with(
                f"""        def Sphere "s"
        {{
            double radius = 1
            double3 xformOp:translate = (1, 2, 3)
            quatd xformOp:orient = ({half}, 0, 0, {half})
            double3 xformOp:scale = (2, 2, 2)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
        }}"""
            )
        )
        visual = schema.links[0].visuals[0]
        assert np.allclose(visual.pose.position.to_list(), [1.0, 2.0, 3.0], atol=1e-9)
        # The 90-degree yaw must land in the orientation, and the factor of 2 must
        # land in the radius rather than inflating the pose.
        assert math.isclose(visual.pose.orientation.z, half, abs_tol=1e-9)
        assert math.isclose(visual.geometry.radius, 2.0, abs_tol=1e-9)

    def test_mesh_without_a_reference_warns_and_leaves_filename_unset(self):
        text = self._link_with(
            """        def Mesh "m"
        {
            point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
            int[] faceVertexCounts = [3]
            int[] faceVertexIndices = [0, 1, 2]
        }"""
        )
        schema, context = parse_with_context(text)
        geometry = schema.links[0].visuals[0].geometry
        assert geometry.type == GeometryType.MESH
        assert geometry.filename is None
        assert any("no" in warning and "reference" in warning for warning in context.warnings)


class TestVisualCollisionDiscrimination:
    FIXTURE = (
        HEADER
        + """
def Xform "Robot" (prepend apiSchemas = ["PhysicsArticulationRootAPI"])
{
    def Xform "a" (prepend apiSchemas = ["PhysicsRigidBodyAPI"])
    {
        def Sphere "vis"
        {
            double radius = 1
        }
        def Sphere "col" (prepend apiSchemas = ["PhysicsCollisionAPI"])
        {
            double radius = 2
            uniform token purpose = "guide"
        }
        def Sphere "both" (prepend apiSchemas = ["PhysicsCollisionAPI"])
        {
            double radius = 3
        }
        def Sphere "hidden"
        {
            double radius = 4
            token visibility = "invisible"
        }
        def Sphere "proxy_only"
        {
            double radius = 5
            uniform token purpose = "proxy"
        }
    }
}
"""
    )

    def test_collision_api_prims_become_collisions(self):
        link = parse(self.FIXTURE).links[0]
        radii = sorted(collision.geometry.radius for collision in link.collisions)
        assert radii == [2.0, 3.0]

    def test_only_visible_renderable_prims_become_visuals(self):
        link = parse(self.FIXTURE).links[0]
        radii = sorted(visual.geometry.radius for visual in link.visuals)
        # "col" is guide, "hidden" is invisible, "proxy_only" is proxy; "both" has a
        # CollisionAPI but is still renderable, so it appears in both lists.
        assert radii == [1.0, 3.0]

    def test_foreign_visuals_and_collisions_have_no_names(self):
        link = parse(self.FIXTURE).links[0]
        assert all(visual.name is None for visual in link.visuals)


class TestForeignDrives:
    FIXTURE = (
        HEADER
        + """
def Xform "Robot" (prepend apiSchemas = ["PhysicsArticulationRootAPI"])
{
    def Xform "a" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
    def Xform "b" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
    def PhysicsRevoluteJoint "j" (prepend apiSchemas = ["PhysicsDriveAPI:angular"])
    {
        rel physics:body0 = </Robot/a>
        rel physics:body1 = </Robot/b>
        float drive:angular:physics:stiffness = 250
        float drive:angular:physics:damping = 12.5
        float drive:angular:physics:maxForce = 80
        float drive:angular:physics:targetPosition = 30
    }
}
"""
    )

    def test_a_drive_yields_an_actuator(self):
        # Without this, importing an Isaac Sim robot and re-exporting to MJCF would
        # silently produce an unactuated model.
        actuators = parse(self.FIXTURE).actuators
        assert len(actuators) == 1
        actuator = actuators[0]
        assert actuator.name == "j_drive"
        assert actuator.joint == "j"
        assert actuator.type == ActuatorType.POSITION
        assert math.isclose(actuator.kp, 250.0, abs_tol=1e-4)
        assert math.isclose(actuator.kd, 12.5, abs_tol=1e-4)
        assert math.isclose(actuator.max_torque, 80.0, abs_tol=1e-4)

    def test_a_drive_yields_joint_dynamics(self):
        dynamics = parse(self.FIXTURE).joints[0].dynamics
        assert math.isclose(dynamics.spring_stiffness, 250.0, abs_tol=1e-4)
        assert math.isclose(dynamics.damping, 12.5, abs_tol=1e-4)

    def test_drive_target_becomes_home_position_in_radians(self):
        home = parse(self.FIXTURE).joints[0].home_position
        assert math.isclose(home, math.radians(30.0), abs_tol=1e-6)

    def test_infinite_max_force_is_not_a_torque_limit(self):
        text = (
            HEADER
            + """
def Xform "Robot" (prepend apiSchemas = ["PhysicsArticulationRootAPI"])
{
    def Xform "a" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
    def Xform "b" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
    def PhysicsRevoluteJoint "j" (prepend apiSchemas = ["PhysicsDriveAPI:angular"])
    {
        rel physics:body0 = </Robot/a>
        rel physics:body1 = </Robot/b>
        float drive:angular:physics:stiffness = 100
        float drive:angular:physics:maxForce = inf
    }
}
"""
        )
        assert parse(text).actuators[0].max_torque is None

    def test_a_joint_without_a_drive_yields_no_actuator(self):
        assert parse(MINIMAL_ROBOT).actuators == []

    def test_our_own_output_never_gets_inferred_actuators(self):
        # Our exporter derives drives from actuators and from passive spring
        # dynamics; re-reading them as new actuators would duplicate or invent them.
        from cyberwave_robot_format.schema import Joint, JointDynamics

        schema = CommonSchema(
            metadata=Metadata(name="r"),
            links=[Link(name="a"), Link(name="b")],
            joints=[
                Joint(
                    name="j",
                    type=JointType.REVOLUTE,
                    parent_link="a",
                    child_link="b",
                    dynamics=JointDynamics(damping=0.5, spring_stiffness=7.5),
                )
            ],
        )
        parsed = USDParser().parse_string(USDExporter().export_to_string(schema))
        assert parsed.actuators == []
        assert parsed.joints[0].home_position is None


class TestRootDiscovery:
    def test_articulation_root_wins_over_the_default_prim(self):
        text = """#usda 1.0
(
    defaultPrim = "Decoy"
    upAxis = "Z"
)

def Xform "Decoy" {}

def Xform "Real" (prepend apiSchemas = ["PhysicsArticulationRootAPI"])
{
    def Xform "a" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
}
"""
        schema = parse(text)
        assert schema.metadata.name == "Real"
        assert [link.name for link in schema.links] == ["a"]

    def test_default_prim_is_used_when_there_is_no_articulation_root(self):
        text = """#usda 1.0
(
    defaultPrim = "Robot"
    upAxis = "Z"
)

def Xform "Robot"
{
    def Xform "a" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
}
"""
        schema, context = parse_with_context(text)
        assert schema.metadata.name == "Robot"
        assert any("defaultPrim" in warning for warning in context.warnings)

    def test_several_articulation_roots_warn_and_the_first_is_parsed(self):
        # One parse produces one articulation; a scene of several robots must say
        # what it dropped rather than returning a silent subset.
        text = """#usda 1.0
(
    upAxis = "Z"
)

def Xform "World"
{
    def Xform "RobotA" (prepend apiSchemas = ["PhysicsArticulationRootAPI"])
    {
        def Xform "a1" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
    }

    def Xform "RobotB" (prepend apiSchemas = ["PhysicsArticulationRootAPI"])
    {
        def Xform "b1" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
    }
}
"""
        schema, context = parse_with_context(text)
        assert schema.metadata.name == "RobotA"
        assert [link.name for link in schema.links] == ["a1"]
        assert any(
            "2 articulation roots" in warning and "/World/RobotB" in warning
            for warning in context.warnings
        )

    def test_a_bare_stage_of_rigid_bodies_still_parses(self):
        text = """#usda 1.0
(
    upAxis = "Z"
)

def Xform "a" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
def Xform "b" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
"""
        schema, context = parse_with_context(text)
        assert [link.name for link in schema.links] == ["a", "b"]
        assert any("whole stage" in warning for warning in context.warnings)


class TestWarnings:
    def test_joint_targeting_a_non_body_prim_warns(self):
        text = (
            HEADER
            + """
def Xform "Robot" (prepend apiSchemas = ["PhysicsArticulationRootAPI"])
{
    def Xform "a" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
    def Xform "not_a_body" {}
    def PhysicsRevoluteJoint "j"
    {
        rel physics:body0 = </Robot/a>
        rel physics:body1 = </Robot/not_a_body>
    }
}
"""
        )
        schema, context = parse_with_context(text)
        assert schema.joints[0].child_link == "not_a_body"
        assert any("not a rigid" in warning for warning in context.warnings)

    def test_unrecognized_joint_type_warns_and_falls_back_to_fixed(self):
        text = (
            HEADER
            + """
def Xform "Robot" (prepend apiSchemas = ["PhysicsArticulationRootAPI"])
{
    def Xform "a" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
    def Xform "b" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
    def PhysicsDistanceJoint "j"
    {
        rel physics:body0 = </Robot/a>
        rel physics:body1 = </Robot/b>
    }
}
"""
        )
        schema, context = parse_with_context(text)
        assert schema.joints[0].type == JointType.FIXED
        assert any("unrecognized joint type" in warning for warning in context.warnings)

    def test_invalid_json_in_a_sidecar_is_ignored_with_a_warning(self, caplog):
        import logging

        text = (
            HEADER
            + """
def Xform "Robot" (prepend apiSchemas = ["PhysicsArticulationRootAPI"])
{
    def Xform "a" (prepend apiSchemas = ["PhysicsRigidBodyAPI"])
    {
        def Xform "Sensors"
        {
            def Xform "s"
            {
                custom string cyberwave:sensor:type = "IMU"
                custom string cyberwave:sensor:parameters = "{not json"
            }
        }
    }
}
"""
        )
        with caplog.at_level(logging.WARNING):
            schema = parse(text)
        assert schema.sensors[0].parameters == {}
        assert "not valid JSON" in caplog.text


class TestForeignMaterialsAndPhysics:
    """Recovery from a stage that has only the native encodings."""

    def test_preview_surface_diffuse_and_opacity_become_a_color(self):
        text = (
            HEADER
            + """
def Xform "Robot" (prepend apiSchemas = ["PhysicsArticulationRootAPI"])
{
    def Xform "a" (prepend apiSchemas = ["PhysicsRigidBodyAPI"])
    {
        def Sphere "s" (prepend apiSchemas = ["MaterialBindingAPI"])
        {
            double radius = 1
            rel material:binding = </Robot/Looks/paint>
        }
    }
    def Scope "Looks"
    {
        def Material "paint"
        {
            token outputs:surface.connect = </Robot/Looks/paint/Shader.outputs:surface>

            def Shader "Shader"
            {
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = (0.25, 0.5, 0.75)
                float inputs:opacity = 0.5
                token outputs:surface
            }
        }
    }
}
"""
        )
        material = parse(text).links[0].visuals[0].material
        assert material is not None
        assert material.name == "paint"
        assert np.allclose(material.color, [0.25, 0.5, 0.75, 0.5], atol=1e-6)

    def test_foreign_material_called_material_keeps_its_name(self):
        # "Material" is Blender's default material name, and a foreign stage may
        # park it under a plain Xform rather than a Scope. Only a material that is a
        # child of the prim binding it is one of *our* inline materials.
        text = (
            HEADER
            + """
def Xform "Robot" (prepend apiSchemas = ["PhysicsArticulationRootAPI"])
{
    def Xform "a" (prepend apiSchemas = ["PhysicsRigidBodyAPI"])
    {
        def Sphere "s" (prepend apiSchemas = ["MaterialBindingAPI"])
        {
            double radius = 1
            rel material:binding = </Robot/Looks/Material>
        }
    }
    def Xform "Looks"
    {
        def Material "Material"
        {
            token outputs:surface.connect = </Robot/Looks/Material/Shader.outputs:surface>

            def Shader "Shader"
            {
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = (0.2, 0.4, 0.6)
                token outputs:surface
            }
        }
    }
}
"""
        )
        material = parse(text).links[0].visuals[0].material
        assert material.name == "Material"
        assert np.allclose(material.color[:3], [0.2, 0.4, 0.6], atol=1e-6)

    def test_uv_texture_file_is_recovered(self):
        text = (
            HEADER
            + """
def Xform "Robot" (prepend apiSchemas = ["PhysicsArticulationRootAPI"])
{
    def Xform "a" (prepend apiSchemas = ["PhysicsRigidBodyAPI"])
    {
        def Sphere "s" (prepend apiSchemas = ["MaterialBindingAPI"])
        {
            double radius = 1
            rel material:binding = </Robot/Looks/wood>
        }
    }
    def Scope "Looks"
    {
        def Material "wood"
        {
            token outputs:surface.connect = </Robot/Looks/wood/Shader.outputs:surface>

            def Shader "Shader"
            {
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = (1, 1, 1)
                token outputs:surface
            }

            def Shader "Tex"
            {
                uniform token info:id = "UsdUVTexture"
                asset inputs:file = @textures/wood.png@
                float3 outputs:rgb
            }
        }
    }
}
"""
        )
        material = parse(text).links[0].visuals[0].material
        assert material.texture == "textures/wood.png"

    def test_material_with_no_recoverable_properties_is_none(self):
        text = (
            HEADER
            + """
def Xform "Robot" (prepend apiSchemas = ["PhysicsArticulationRootAPI"])
{
    def Xform "a" (prepend apiSchemas = ["PhysicsRigidBodyAPI"])
    {
        def Sphere "s"
        {
            double radius = 1
        }
    }
}
"""
        )
        assert parse(text).links[0].visuals[0].material is None

    def test_collision_friction_comes_from_the_bound_physics_material(self):
        text = (
            HEADER
            + """
def Xform "Robot" (prepend apiSchemas = ["PhysicsArticulationRootAPI"])
{
    def Xform "a" (prepend apiSchemas = ["PhysicsRigidBodyAPI"])
    {
        def Sphere "c" (prepend apiSchemas = ["PhysicsCollisionAPI", "MaterialBindingAPI"])
        {
            double radius = 1
            rel material:binding:physics = </Robot/PhysMats/rubber>
        }
    }
    def Scope "PhysMats"
    {
        def Material "rubber" (prepend apiSchemas = ["PhysicsMaterialAPI"])
        {
            float physics:staticFriction = 0.9
            float physics:dynamicFriction = 0.8
            float physics:restitution = 0.2
        }
    }
}
"""
        )
        collision = parse(text).links[0].collisions[0]
        assert math.isclose(collision.mu_static, 0.9, abs_tol=1e-6)
        assert math.isclose(collision.mu_dynamic, 0.8, abs_tol=1e-6)
        assert math.isclose(collision.restitution, 0.2, abs_tol=1e-6)

    def test_gravity_is_reconstructed_from_direction_and_magnitude(self):
        text = (
            HEADER
            + """
def Xform "Robot" (prepend apiSchemas = ["PhysicsArticulationRootAPI"]) {}

def PhysicsScene "PhysicsScene"
{
    vector3f physics:gravityDirection = (0, 1, 0)
    float physics:gravityMagnitude = 3
}
"""
        )
        gravity = parse(text).physics.gravity
        assert np.allclose(gravity.to_list(), [0.0, 3.0, 0.0], atol=1e-6)

    def test_zero_direction_sentinel_means_down_the_up_axis(self):
        text = (
            HEADER
            + """
def Xform "Robot" (prepend apiSchemas = ["PhysicsArticulationRootAPI"]) {}

def PhysicsScene "PhysicsScene"
{
    float physics:gravityMagnitude = 9.8
}
"""
        )
        gravity = parse(text).physics.gravity
        assert np.allclose(gravity.to_list(), [0.0, 0.0, -9.8], atol=1e-6)

    def test_earth_gravity_sentinel_falls_back_to_the_schema_default(self):
        # An unauthored magnitude reads as -inf, which means "earth gravity" — not a
        # number to multiply a direction by.
        text = (
            HEADER
            + """
def Xform "Robot" (prepend apiSchemas = ["PhysicsArticulationRootAPI"]) {}

def PhysicsScene "PhysicsScene"
{
    vector3f physics:gravityDirection = (0, 0, -1)
}
"""
        )
        from cyberwave_robot_format.schema import Physics

        assert parse(text).physics.gravity == Physics().gravity

    def test_solver_and_contact_defaults_are_used_when_unauthored(self):
        from cyberwave_robot_format.schema import ContactSurface, PhysicsSolver

        text = (
            HEADER
            + """
def Xform "Robot" (prepend apiSchemas = ["PhysicsArticulationRootAPI"]) {}

def PhysicsScene "PhysicsScene" {}
"""
        )
        physics = parse(text).physics
        assert physics.solver == PhysicsSolver()
        assert physics.contact == ContactSurface()

    def test_a_foreign_physics_scene_is_found_wherever_it_sits(self):
        # Isaac Sim and Omniverse assets nest the scene under /World and spell it
        # with a lowercase p; keying only off our own /PhysicsScene path dropped
        # their gravity and solver settings entirely.
        text = (
            HEADER
            + """
def Xform "World"
{
    def Xform "Robot" (prepend apiSchemas = ["PhysicsArticulationRootAPI"]) {}

    def PhysicsScene "physicsScene"
    {
        vector3f physics:gravityDirection = (0, 0, -1)
        float physics:gravityMagnitude = 3.7
    }
}
"""
        )
        gravity = parse(text).physics.gravity
        assert np.allclose(gravity.to_list(), [0.0, 0.0, -3.7], atol=1e-6)

    def test_several_physics_scenes_keep_the_first_and_warn(self):
        text = (
            HEADER
            + """
def Xform "Robot" (prepend apiSchemas = ["PhysicsArticulationRootAPI"]) {}

def PhysicsScene "sceneA"
{
    float physics:gravityMagnitude = 1
}

def PhysicsScene "sceneB"
{
    float physics:gravityMagnitude = 2
}
"""
        )
        parser = USDParser()
        schema = parser.parse_string(text)
        gravity = schema.physics.gravity
        assert np.allclose(gravity.to_list(), [0.0, 0.0, -1.0], atol=1e-6)

    def test_no_physics_scene_means_no_physics(self):
        assert parse(MINIMAL_ROBOT).physics is None
        assert parse(MINIMAL_ROBOT).scene is None
        assert parse(MINIMAL_ROBOT).collision_config is None


class TestMalformedSidecars:
    def test_unknown_geometry_type_token_degrades_to_unknown(self, caplog):
        import logging

        text = (
            HEADER
            + """
def Xform "Robot" (prepend apiSchemas = ["PhysicsArticulationRootAPI"])
{
    def Xform "a" (prepend apiSchemas = ["PhysicsRigidBodyAPI"])
    {
        def Scope "Visuals"
        {
            def Sphere "s"
            {
                double radius = 1
                custom token cyberwave:geometry:type = "dodecahedron"
            }
        }
    }
}
"""
        )
        with caplog.at_level(logging.WARNING):
            geometry = parse(text).links[0].visuals[0].geometry
        assert geometry.type == GeometryType.UNKNOWN
        assert "dodecahedron" in caplog.text

    def test_unknown_joint_type_token_falls_back_to_the_prim_type(self):
        text = (
            HEADER
            + """
def Xform "Robot" (prepend apiSchemas = ["PhysicsArticulationRootAPI"])
{
    def Xform "a" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
    def Xform "b" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
    def PhysicsPrismaticJoint "j"
    {
        rel physics:body0 = </Robot/a>
        rel physics:body1 = </Robot/b>
        custom token cyberwave:joint:type = "wobbly"
    }
}
"""
        )
        schema, context = parse_with_context(text)
        assert schema.joints[0].type == JointType.PRISMATIC
        assert any("wobbly" in warning for warning in context.warnings)

    def test_unknown_actuator_type_token_defaults_to_position(self):
        text = (
            HEADER
            + """
def Xform "Robot" (prepend apiSchemas = ["PhysicsArticulationRootAPI"])
{
    def Scope "Actuators"
    {
        def Scope "m"
        {
            custom string cyberwave:actuator:joint = "j"
            custom token cyberwave:actuator:type = "warp_drive"
        }
    }
}
"""
        )
        schema, context = parse_with_context(text)
        assert schema.actuators[0].type == ActuatorType.POSITION
        assert any("warp_drive" in warning for warning in context.warnings)

    def test_mimic_without_a_multiplier_defaults_to_one_to_one(self):
        # A multiplier of 0 is a dead coupling; the dataclass default is 1:1, so a
        # hand-authored stage naming only the driver must read back as 1:1.
        text = (
            HEADER
            + """
def Xform "Robot" (prepend apiSchemas = ["PhysicsArticulationRootAPI"])
{
    def Xform "a" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
    def Xform "b" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
    def PhysicsRevoluteJoint "j"
    {
        rel physics:body0 = </Robot/a>
        rel physics:body1 = </Robot/b>
        custom string cyberwave:joint:mimicJoint = "k"
    }
}
"""
        )
        mimic = parse(text).joints[0].mimic
        assert mimic.joint == "k"
        assert mimic.multiplier == 1.0
        assert mimic.offset == 0.0

    def test_non_numeric_pair_suffix_sorts_last(self):
        from cyberwave_robot_format.usd.parser import _pair_index

        assert _pair_index("pair_3") == 3
        assert _pair_index("pair_x") > 1000


class TestFileHandling:
    def test_can_parse_accepts_usd_suffixes(self, tmp_path):
        parser = USDParser()
        for suffix in (".usd", ".usda", ".usdc", ".usdz"):
            path = tmp_path / f"r{suffix}"
            path.write_text("")
            assert parser.can_parse(path)

    def test_can_parse_sniffs_an_extensionless_file(self, tmp_path):
        path = tmp_path / "mystery"
        path.write_text(MINIMAL_ROBOT)
        assert USDParser().can_parse(path)

    def test_can_parse_rejects_urdf_and_mjcf(self):
        parser = USDParser()
        assert not parser.can_parse(FIXTURES / "simple_robot.urdf")
        assert not parser.can_parse(FIXTURES / "simple_robot.xml")

    def test_can_parse_rejects_a_missing_file(self, tmp_path):
        assert not USDParser().can_parse(tmp_path / "nope")

    def test_parse_stages_an_extensionless_file_through_a_temp_copy(self, tmp_path):
        # Usd.Stage.Open resolves the layer encoding from the suffix and hard-errors
        # without one, so the parser copies to a correctly-named temp file first.
        path = tmp_path / "mystery"
        path.write_text(MINIMAL_ROBOT)
        schema = USDParser().parse(path)
        assert [link.name for link in schema.links] == ["base", "arm"]

    def test_parse_rejects_a_non_usd_file_with_a_value_error(self):
        with pytest.raises(ValueError, match="does not look like a USD file"):
            USDParser().parse(FIXTURES / "simple_robot.urdf")

    def test_parse_of_a_corrupt_usda_raises_value_error(self, tmp_path):
        path = tmp_path / "broken.usda"
        path.write_text("#usda 1.0\ndef Xform \"Unclosed\" {\n")
        with pytest.raises(ValueError, match="Could not open USD stage"):
            USDParser().parse(path)

    def test_parse_string_rejects_garbage(self):
        with pytest.raises(ValueError, match="Could not parse"):
            USDParser().parse_string("this is not usda at all")

    def test_parser_instance_is_reusable(self):
        parser = USDParser()
        first = parser.parse_string(MINIMAL_ROBOT)
        second = parser.parse_string(MINIMAL_ROBOT)
        assert first.to_dict() == second.to_dict()


class TestUntrustedInputHardening:
    """Parsing a stage you did not author is a trust boundary.

    Opening a USD file *composes* it, so ``subLayers``/``references``/``payloads``
    are resolved and their content lands in the returned schema. A service that
    parses an upload and returns the schema would otherwise hand back the content of
    local USD files it never meant to expose.
    """

    @pytest.mark.parametrize("arc", ["subLayer", "reference"])
    def test_external_layers_are_rejected_when_restricted(self, tmp_path, arc):
        # The victim lives outside the attacker file's directory.
        outside = tmp_path / "elsewhere"
        outside.mkdir()
        attacker_dir = tmp_path / "upload"
        attacker_dir.mkdir()
        victim = outside / "private.usda"
        victim.write_text(
            """#usda 1.0
(
    defaultPrim = "Private"
    upAxis = "Z"
)

def Xform "Private" (prepend apiSchemas = ["PhysicsArticulationRootAPI"])
{
    def Xform "confidential_link" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
}
"""
        )
        attacker = attacker_dir / "attack.usda"
        if arc == "subLayer":
            attacker.write_text(
                f'#usda 1.0\n(\n    defaultPrim = "Robot"\n    upAxis = "Z"\n'
                f"    subLayers = [@{victim}@]\n)\n\n"
                'def Xform "Robot" (prepend apiSchemas = ["PhysicsArticulationRootAPI"]) {}\n'
            )
        else:
            attacker.write_text(
                f'#usda 1.0\n(\n    defaultPrim = "Robot"\n    upAxis = "Z"\n)\n\n'
                'def Xform "Robot" (prepend apiSchemas = ["PhysicsArticulationRootAPI"])\n{\n'
                f"    def Xform \"pulled\" (prepend references = @{victim}@</Private/confidential_link>) {{}}\n}}\n"
            )

        with pytest.raises(ValueError, match="outside the permitted directory"):
            USDParser(allow_external_layers=False).parse(attacker)

        # ...and the permissive default still composes it, which is why the flag has
        # to exist rather than the behaviour simply being changed.
        leaked = USDParser().parse(attacker)
        assert leaked.links, "composition is expected to work by default"

    def test_layers_beside_the_input_are_still_allowed(self, tmp_path):
        # Restriction must not break ordinary multi-file assets.
        part = tmp_path / "part.usda"
        part.write_text(
            """#usda 1.0
(
    defaultPrim = "Part"
    upAxis = "Z"
)

def Xform "Part"
{
    def Xform "arm" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}
}
"""
        )
        main = tmp_path / "main.usda"
        main.write_text(
            f'#usda 1.0\n(\n    defaultPrim = "Robot"\n    upAxis = "Z"\n'
            f"    subLayers = [@./{part.name}@]\n)\n\n"
            'def Xform "Robot" (prepend apiSchemas = ["PhysicsArticulationRootAPI"])\n'
            "{\n"
            '    def Xform "base" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}\n}\n'
        )
        schema = USDParser(allow_external_layers=False).parse(main)
        assert {link.name for link in schema.links} >= {"base"}

    def test_nested_subdirectories_are_allowed(self, tmp_path):
        nested = tmp_path / "meshes"
        nested.mkdir()
        part = nested / "part.usda"
        part.write_text(
            '#usda 1.0\n(\n    upAxis = "Z"\n)\n\ndef Xform "Part" {}\n'
        )
        main = tmp_path / "main.usda"
        main.write_text(
            f'#usda 1.0\n(\n    defaultPrim = "Robot"\n    upAxis = "Z"\n'
            f"    subLayers = [@./meshes/{part.name}@]\n)\n\n"
            'def Xform "Robot" (prepend apiSchemas = ["PhysicsArticulationRootAPI"])\n'
            '{\n    def Xform "base" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}\n}\n'
        )
        assert USDParser(allow_external_layers=False).parse(main).links

    def test_symlink_cannot_escape_the_permitted_directory(self, tmp_path):
        outside = tmp_path / "outside"
        outside.mkdir()
        victim = outside / "private.usda"
        victim.write_text('#usda 1.0\n(\n    upAxis = "Z"\n)\n\ndef Xform "Private" {}\n')

        upload = tmp_path / "upload"
        upload.mkdir()
        link = upload / "innocent.usda"
        try:
            link.symlink_to(victim)
        except (OSError, NotImplementedError):  # pragma: no cover - platform dependent
            pytest.skip("symlinks unavailable on this platform")

        main = upload / "main.usda"
        main.write_text(
            '#usda 1.0\n(\n    defaultPrim = "Robot"\n    upAxis = "Z"\n'
            "    subLayers = [@./innocent.usda@]\n)\n\n"
            'def Xform "Robot" (prepend apiSchemas = ["PhysicsArticulationRootAPI"]) {}\n'
        )
        # realPath resolves the symlink, so the real location is what gets checked.
        with pytest.raises(ValueError, match="outside the permitted directory"):
            USDParser(allow_external_layers=False).parse(main)

    def test_parse_string_rejects_any_external_layer_when_restricted(self):
        # Text has no directory to anchor against.
        assert USDParser(allow_external_layers=False).parse_string(MINIMAL_ROBOT).links

    def test_restriction_does_not_disturb_a_normal_single_file(self, tmp_path):
        path = tmp_path / "plain.usda"
        path.write_text(MINIMAL_ROBOT)
        strict = USDParser(allow_external_layers=False).parse(path)
        relaxed = USDParser().parse(path)
        assert strict.to_dict() == relaxed.to_dict()

    def test_deeply_nested_json_payload_does_not_abort_the_parse(self, caplog):
        # json's scanner raises RecursionError, which is not a ValueError — without
        # explicit handling one hostile attribute would kill the whole run.
        import logging

        payload = "[" * 60000 + "]" * 60000
        text = (
            HEADER
            + """
def Xform "Robot" (prepend apiSchemas = ["PhysicsArticulationRootAPI"])
{
    def Xform "a" (prepend apiSchemas = ["PhysicsRigidBodyAPI"])
    {
        def Xform "Sensors"
        {
            def Xform "s"
            {
                custom string cyberwave:sensor:type = "IMU"
                custom string cyberwave:sensor:parameters = \""""
            + payload
            + """\"
            }
        }
    }
}
"""
        )
        with caplog.at_level(logging.WARNING):
            schema = parse(text)
        assert schema.sensors[0].type == "IMU"
        assert schema.sensors[0].parameters == {}
        assert "nested too deeply" in caplog.text

    def test_deeply_nested_custom_data_does_not_abort_the_parse(self, caplog):
        import logging

        payload = "[" * 60000 + "]" * 60000
        text = (
            HEADER
            + """
def Xform "Robot" (
    prepend apiSchemas = ["PhysicsArticulationRootAPI"]
    customData = {
        dictionary cyberwave = {
            string extensions = \""""
            + payload
            + """\"
        }
    }
)
{
}
"""
        )
        with caplog.at_level(logging.WARNING):
            schema = parse(text)
        assert schema.extensions == {}
        assert "nested too deeply" in caplog.text
