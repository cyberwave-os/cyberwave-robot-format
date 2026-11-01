# Copyright [2025] Cyberwave
# Licensed under the Apache License, Version 2.0

"""Round-trip tests: ``CommonSchema`` -> USD -> ``CommonSchema``.

The contract these tests pin down is that the USD round trip is **lossless** for
everything the IR can express, not merely close. That is only possible because the
exporter pairs every lossy or absent native ``UsdPhysics`` encoding with a
double-precision ``cyberwave:`` sidecar that the parser treats as authoritative —
so if a sidecar ever stops being written, a test here fails rather than a value
quietly losing its 8th significant digit.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

pytest.importorskip("pxr")

from cyberwave_robot_format.schema import (  # noqa: E402
    Actuator,
    ActuatorType,
    Collision,
    CollisionConfig,
    CollisionExclude,
    CollisionGroup,
    CollisionPair,
    CommonSchema,
    Contact,
    ContactSurface,
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
    PhysicsSolver,
    Pose,
    Quaternion,
    Scene,
    Sensor,
    Vector3,
    Visual,
)
from cyberwave_robot_format.urdf import URDFParser  # noqa: E402
from cyberwave_robot_format.mjcf import MJCFParser  # noqa: E402
from cyberwave_robot_format.usd import USDExporter, USDParser  # noqa: E402

from tests.usd_helpers import assert_schema_equal  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"

NESTED_EXTENSIONS = {
    "a": None,
    "b": True,
    "c": 1,
    "d": 1.5,
    "e": [1, "x", None],
    "f": {"g": []},
}


def round_trip(schema: CommonSchema) -> CommonSchema:
    """Export ``schema`` to ``.usda`` text and parse it back."""
    return USDParser().parse_string(USDExporter().export_to_string(schema))


def full_schema() -> CommonSchema:
    """A schema exercising every field of every dataclass with a non-default value.

    Deliberately built in Python rather than loaded from JSON: ``Sensor.noise``
    holds a ``dict[str, float | list[float]]`` that cattrs cannot structure, so
    ``CommonSchema.from_dict`` would reject it (see
    ``MJCFExporter._set_sensor_noise``). The USD parser does not go through cattrs,
    so it can carry the value that the JSON path cannot.
    """
    return CommonSchema(
        metadata=Metadata(
            name="kitchen-sink.robot",
            version="3.2",
            author="Cyberwave",
            description="every field populated",
            source_format="urdf",
            creation_date="2026-08-10",
            units="SI",
            extensions=dict(NESTED_EXTENSIONS),
        ),
        physics=Physics(
            gravity=Vector3(x=0.1, y=-0.2, z=-9.80665),
            timestep=0.002,
            solver=PhysicsSolver(
                type="ode",
                iterations=137,
                sor=1.7,
                min_step_size=0.00013,
                max_step_size=0.0021,
                tolerance=1e-7,
                extensions=dict(NESTED_EXTENSIONS),
            ),
            contact=ContactSurface(
                mu_static=0.91,
                mu_dynamic=0.72,
                restitution=0.31,
                stiffness=12345.5,
                damping=21.5,
                slip_compliance={"x": 0.001, "y": 0.002},
                soft_cfm=0.0001,
                soft_erp=0.21,
            ),
            extensions=dict(NESTED_EXTENSIONS),
        ),
        scene=Scene(
            ambient=[0.11, 0.12, 0.13, 1.0],
            background=[0.2, 0.3, 0.4, 0.5],
            shadows=False,
            grid=True,
            origin_visual=True,
            extensions=dict(NESTED_EXTENSIONS),
        ),
        collision_config=CollisionConfig(
            groups={
                "default": CollisionGroup(name="default", contype=1, conaffinity=1),
                "ROBOT": CollisionGroup(name="ROBOT", contype=2, conaffinity=4),
                "ENV": CollisionGroup(name="ENV", contype=4, conaffinity=2),
            },
            excludes=[
                CollisionExclude(body1="base_link", body2="arm-1"),
                CollisionExclude(body1="arm-1", body2="tool.tip"),
            ],
            pairs=[
                CollisionPair(
                    geom1="base_col",
                    geom2="arm_col",
                    friction=[1.0, 0.005, 0.0001],
                    solref=[0.02, 1.0],
                    solimp=[0.9, 0.95, 0.001, 0.5, 2.0],
                    condim=6,
                    margin=0.001,
                    gap=0.0005,
                    extensions=dict(NESTED_EXTENSIONS),
                ),
                CollisionPair(geom1="arm_col", geom2="tool_col"),
            ],
            extensions=dict(NESTED_EXTENSIONS),
        ),
        materials=[
            # Same values as the visual's copy: the document material wins on export.
            Material(
                name="metal",
                color=[0.7, 0.7, 0.75, 0.9],
                texture="textures/metal.png",
                specular=[0.4, 0.4, 0.4],
                emissive=[0.01, 0.02, 0.03],
                shininess=0.65,
            ),
            # Referenced by no visual: still part of the document.
            Material(name="rubber", color=[0.05, 0.05, 0.05, 1.0], shininess=0.1),
        ],
        links=[
            Link(
                name="base_link",
                mass=12.3456789012345,
                center_of_mass=Vector3(x=0.011, y=-0.022, z=0.033),
                inertial_orientation=Quaternion.from_rpy(0.05, -0.1, 0.785),
                inertia=Inertia(
                    ixx=0.51, iyy=0.62, izz=0.73, ixy=0.011, ixz=-0.022, iyz=0.033
                ),
                visuals=[
                    Visual(
                        name="base_box",
                        pose=Pose(
                            position=Vector3(z=0.05),
                            orientation=Quaternion.from_rpy(0.1, 0.2, 0.3),
                        ),
                        geometry=Geometry(
                            type=GeometryType.BOX,
                            size=Vector3(x=0.11, y=0.22, z=0.33),
                            extensions=dict(NESTED_EXTENSIONS),
                        ),
                        material=Material(
                            name="metal",
                            color=[0.7, 0.7, 0.75, 0.9],
                            texture="textures/metal.png",
                            specular=[0.4, 0.4, 0.4],
                            emissive=[0.01, 0.02, 0.03],
                            shininess=0.65,
                        ),
                        extensions=dict(NESTED_EXTENSIONS),
                    ),
                    Visual(
                        name=None,
                        geometry=Geometry(type=GeometryType.SPHERE, radius=0.075),
                        material=Material(color=[1.0, 0.5, 0.25]),
                    ),
                    Visual(
                        name="base_ellipsoid",
                        geometry=Geometry(
                            type=GeometryType.ELLIPSOID,
                            size=Vector3(x=0.1, y=0.2, z=0.05),
                        ),
                    ),
                    Visual(
                        name="base_plane",
                        geometry=Geometry(
                            type=GeometryType.PLANE,
                            size=Vector3(x=4.0, y=5.0, z=0.5),
                        ),
                    ),
                    Visual(
                        name="base_unknown",
                        geometry=Geometry(type=GeometryType.UNKNOWN),
                    ),
                ],
                collisions=[
                    Collision(
                        name="base_col",
                        pose=Pose(position=Vector3(x=0.01)),
                        geometry=Geometry(
                            type=GeometryType.CYLINDER, radius=0.15, length=0.4
                        ),
                        group="ROBOT",
                        mu_static=0.83,
                        mu_dynamic=0.71,
                        restitution=0.13,
                        stiffness=54321.5,
                        damping=17.25,
                        extensions=dict(NESTED_EXTENSIONS),
                    ),
                    Collision(
                        name="base_capsule",
                        geometry=Geometry(
                            type=GeometryType.CAPSULE, radius=0.05, length=0.2
                        ),
                        group="default",
                    ),
                ],
                extensions=dict(NESTED_EXTENSIONS),
            ),
            Link(
                name="arm-1",
                mass=3.5,
                collisions=[
                    Collision(
                        name="arm_col",
                        geometry=Geometry(
                            type=GeometryType.MESH,
                            filename="package://demo/meshes/arm.STL",
                            scale=Vector3(x=0.001, y=0.001, z=0.001),
                        ),
                    )
                ],
            ),
            Link(name="tool.tip"),
            Link(name="wheel_fl"),
            Link(name="wheel_fr"),
            Link(name="pan"),
            Link(name="tilt"),
            Link(name="free_body"),
        ],
        joints=[
            Joint(
                name="shoulder",
                type=JointType.REVOLUTE,
                parent_link="base_link",
                child_link="arm-1",
                pose=Pose(
                    position=Vector3(x=0.01, y=0.02, z=0.3),
                    orientation=Quaternion.from_rpy(0.11, -0.22, 0.33),
                ),
                axis=Vector3(x=0.0, y=0.7071067811865476, z=0.7071067811865476),
                limits=JointLimits(
                    lower=-1.2345678901234567,
                    upper=2.3456789012345678,
                    effort=45.5,
                    velocity=3.25,
                    acceleration=12.5,
                    jerkmax=100.5,
                ),
                dynamics=JointDynamics(
                    damping=0.55,
                    friction=0.11,
                    spring_reference=0.25,
                    spring_stiffness=7.5,
                    armature=0.0125,
                    backlash=0.002,
                ),
                safety_controller={"k_position": 10.0, "soft_lower_limit": -1.2},
                calibration={"rising": 0.15},
                home_position=0.4567890123456789,
                mimic=MimicJoint(joint="wrist", multiplier=-1.5, offset=0.125),
                extensions=dict(NESTED_EXTENSIONS),
            ),
            Joint(
                name="wrist",
                type=JointType.CONTINUOUS,
                parent_link="arm-1",
                child_link="tool.tip",
                axis=Vector3(x=1.0, y=0.0, z=0.0),
                limits=JointLimits(effort=5.0, velocity=1.0),
            ),
            Joint(
                name="slider",
                type=JointType.PRISMATIC,
                parent_link="base_link",
                child_link="wheel_fl",
                pose=Pose(position=Vector3(y=0.2)),
                axis=Vector3(x=0.0, y=0.0, z=-1.0),
                limits=JointLimits(lower=-0.05, upper=0.15),
                home_position=0.025,
            ),
            Joint(
                name="fixed_mount",
                type=JointType.FIXED,
                parent_link="base_link",
                child_link="wheel_fr",
                pose=Pose(position=Vector3(y=-0.2)),
                axis=Vector3(x=0.5, y=0.5, z=0.0),
            ),
            Joint(
                name="ball",
                type=JointType.SPHERICAL,
                parent_link="tool.tip",
                child_link="pan",
                limits=JointLimits(lower=-0.5, upper=0.5),
            ),
            Joint(
                name="hooke",
                type=JointType.UNIVERSAL,
                parent_link="pan",
                child_link="tilt",
                axis=Vector3(x=0.0, y=1.0, z=0.0),
            ),
            Joint(
                name="caster",
                type=JointType.PLANAR,
                parent_link="base_link",
                child_link="free_body",
                axis=Vector3(x=0.0, y=0.0, z=1.0),
            ),
            Joint(
                name="spawn",
                type=JointType.FLOATING,
                parent_link="world",
                child_link="base_link",
                pose=Pose(
                    position=Vector3(x=1.0, y=2.0, z=0.5),
                    orientation=Quaternion.from_rpy(0.0, 0.0, math.pi / 4),
                ),
            ),
        ],
        actuators=[
            Actuator(
                name="shoulder_motor",
                joint="shoulder",
                type=ActuatorType.DC_MOTOR,
                torque_constant=0.085,
                gear_ratio=101.5,
                max_current=12.5,
                max_torque=45.5,
                max_velocity=3.25,
                kp=125.5,
                ki=0.75,
                kd=8.25,
                control_range=(-1.5, 1.5),
                force_range=(-50.5, 50.5),
                resistance=0.55,
                inductance=0.00025,
                efficiency=0.87,
                extensions=dict(NESTED_EXTENSIONS),
            ),
            Actuator(name="wrist_servo", joint="wrist", type=ActuatorType.SERVO),
            Actuator(name="slide_vel", joint="slider", type=ActuatorType.VELOCITY),
            Actuator(name="pan_pos", joint="ball", type=ActuatorType.POSITION),
            Actuator(name="tilt_torque", joint="hooke", type=ActuatorType.TORQUE),
            Actuator(name="muscle_1", joint="caster", type=ActuatorType.MUSCLE),
        ],
        sensors=[
            Sensor(
                name="imu",
                type="IMU",
                parent_link="base_link",
                pose=Pose(
                    position=Vector3(z=0.02),
                    orientation=Quaternion.from_rpy(0.0, math.pi, 0.0),
                ),
                update_rate=200.5,
                noise={"stddev": 0.001, "bias": [0.1, 0.2, 0.3]},
                parameters={"frame": "body", "gyro_range": 2000},
                extensions=dict(NESTED_EXTENSIONS),
            ),
            Sensor(
                name="wrist-cam",
                type="camera",
                parent_link="tool.tip",
                update_rate=30.0,
                parameters={"hfov": 1.047, "width": 640, "height": 480},
            ),
            Sensor(
                name="world_lidar",
                type="lidar",
                parent_link="world",
                pose=Pose(position=Vector3(z=2.0)),
                update_rate=10.0,
            ),
        ],
        contacts=[
            Contact(
                name="foot_contact",
                link="wheel_fl",
                surface=ContactSurface(
                    mu_static=0.95,
                    mu_dynamic=0.85,
                    restitution=0.05,
                    stiffness=99999.5,
                    damping=33.25,
                    slip_compliance={"lateral": 0.0005},
                    soft_cfm=0.00002,
                    soft_erp=0.42,
                ),
                extensions=dict(NESTED_EXTENSIONS),
            ),
            Contact(name="plain_contact", link="wheel_fr"),
        ],
        extensions=dict(NESTED_EXTENSIONS),
    )


class TestFullRoundTrip:
    def test_kitchen_sink_schema_survives_the_round_trip(self):
        schema = full_schema()
        assert schema.validate() == []
        assert_schema_equal(schema, round_trip(schema))

    def test_round_trip_is_idempotent(self):
        # The strongest available check that nothing drifts: a second pass through
        # the converter must reproduce the first pass byte for byte.
        schema = full_schema()
        first = USDExporter().export_to_string(schema)
        second = USDExporter().export_to_string(USDParser().parse_string(first))
        assert first == second

    @pytest.mark.parametrize("kind", ["visual", "collision", "sensor"])
    def test_local_pose_is_exact_across_repeated_round_trips(self, kind):
        # A rotation matrix cannot recover the original quaternion bits. Preserve
        # authored poses as we already preserve joint poses and inertial values.
        pose = Pose.from_xyzrpy([0.013579, -0.24, 0.05], [0.1, 0.2, 0.3])
        schema = CommonSchema(metadata=Metadata(name="pose-round-trip"), links=[Link(name="base")])
        geometry = Geometry(type=GeometryType.BOX, size=Vector3(0.11, 0.22, 0.33))
        if kind == "visual":
            schema.links[0].visuals = [Visual(pose=pose, geometry=geometry)]
        elif kind == "collision":
            schema.links[0].collisions = [Collision(pose=pose, geometry=geometry)]
        else:
            schema.sensors = [Sensor(name="camera", type="camera", parent_link="base", pose=pose)]
        text = USDExporter().export_to_string(schema)
        for _ in range(5):
            restored = USDParser().parse_string(text)
            if kind == "visual":
                restored_pose = restored.links[0].visuals[0].pose
            elif kind == "collision":
                restored_pose = restored.links[0].collisions[0].pose
            else:
                restored_pose = restored.sensors[0].pose
            assert restored_pose == pose
            assert USDExporter().export_to_string(restored) == text

    def test_export_rejects_metadata_its_reader_would_discard(self):
        schema = full_schema()
        value = 0
        for _ in range(256):
            value = [value]
        schema.extensions = {"deep": value}
        with pytest.raises(ValueError, match="256 nesting levels"):
            USDExporter().export_to_string(schema)

    def test_export_is_deterministic(self):
        schema = full_schema()
        assert USDExporter().export_to_string(schema) == USDExporter().export_to_string(
            schema
        )

    def test_export_does_not_mutate_the_input_schema(self):
        schema = full_schema()
        before = schema.to_dict()
        USDExporter().export_to_string(schema)
        assert schema.to_dict() == before

    def test_round_trip_via_usda_usdc_and_usdz_files(self, tmp_path):
        schema = full_schema()
        for suffix in (".usda", ".usdc", ".usd", ".usdz"):
            path = tmp_path / f"robot{suffix}"
            USDExporter().export(schema, path)
            assert path.stat().st_size > 0
            assert_schema_equal(schema, USDParser().parse(path))


class TestPresenceMarkers:
    """``None`` and an all-``None`` dataclass are different things."""

    @staticmethod
    def _with_joint(**kwargs) -> CommonSchema:
        return CommonSchema(
            metadata=Metadata(name="marker"),
            links=[Link(name="a"), Link(name="b")],
            joints=[
                Joint(
                    name="j",
                    type=JointType.REVOLUTE,
                    parent_link="a",
                    child_link="b",
                    **kwargs,
                )
            ],
        )

    def test_limits_none_stays_none(self):
        parsed = round_trip(self._with_joint(limits=None))
        assert parsed.joints[0].limits is None

    def test_empty_limits_object_stays_an_object(self):
        parsed = round_trip(self._with_joint(limits=JointLimits()))
        assert parsed.joints[0].limits == JointLimits()

    def test_dynamics_none_stays_none(self):
        parsed = round_trip(self._with_joint(dynamics=None))
        assert parsed.joints[0].dynamics is None

    def test_empty_dynamics_object_stays_an_object(self):
        parsed = round_trip(self._with_joint(dynamics=JointDynamics()))
        assert parsed.joints[0].dynamics == JointDynamics()

    def test_collision_config_none_stays_none(self):
        schema = CommonSchema(metadata=Metadata(name="m"), links=[Link(name="a")])
        assert round_trip(schema).collision_config is None

    def test_empty_collision_config_stays_an_object(self):
        schema = CommonSchema(
            metadata=Metadata(name="m"),
            links=[Link(name="a")],
            collision_config=CollisionConfig(),
        )
        assert round_trip(schema).collision_config == CollisionConfig()

    def test_physics_and_scene_none_stay_none(self):
        schema = CommonSchema(metadata=Metadata(name="m"), links=[Link(name="a")])
        parsed = round_trip(schema)
        assert parsed.physics is None
        assert parsed.scene is None

    @pytest.mark.parametrize(
        "field",
        ["mu_static", "mu_dynamic", "restitution", "stiffness", "damping"],
    )
    def test_optional_collision_fields_stay_none(self, field):
        schema = CommonSchema(
            metadata=Metadata(name="m"),
            links=[
                Link(
                    name="a",
                    collisions=[
                        Collision(
                            name="c",
                            geometry=Geometry(type=GeometryType.SPHERE, radius=1.0),
                            **{field: 0.5},
                        )
                    ],
                )
            ],
        )
        collision = round_trip(schema).links[0].collisions[0]
        assert getattr(collision, field) == 0.5
        for other in ("mu_static", "mu_dynamic", "restitution", "stiffness", "damping"):
            if other != field:
                assert getattr(collision, other) is None

    def test_geometry_optional_fields_stay_none(self):
        schema = CommonSchema(
            metadata=Metadata(name="m"),
            links=[
                Link(
                    name="a",
                    visuals=[
                        Visual(geometry=Geometry(type=GeometryType.MESH, filename="a.obj"))
                    ],
                )
            ],
        )
        geometry = round_trip(schema).links[0].visuals[0].geometry
        assert geometry.filename == "a.obj"
        assert geometry.scale is None
        assert geometry.size is None
        assert geometry.radius is None
        assert geometry.length is None

    def test_sensor_and_actuator_optionals_stay_none(self):
        schema = CommonSchema(
            metadata=Metadata(name="m"),
            links=[Link(name="a"), Link(name="b")],
            joints=[
                Joint(name="j", type=JointType.REVOLUTE, parent_link="a", child_link="b")
            ],
            sensors=[Sensor(name="s", type="IMU", parent_link="a")],
            actuators=[Actuator(name="m1", joint="j", type=ActuatorType.SERVO)],
        )
        parsed = round_trip(schema)
        assert parsed.sensors[0].update_rate is None
        assert parsed.sensors[0].noise is None
        assert parsed.sensors[0].parameters == {}
        actuator = parsed.actuators[0]
        for field in ("ki", "kp", "kd", "gear_ratio", "max_current", "efficiency"):
            assert getattr(actuator, field) is None
        assert actuator.control_range is None
        assert actuator.force_range is None


class TestTypeFidelity:
    def test_actuator_ranges_stay_tuples(self):
        # to_dict() flattens tuples to lists, so assert_schema_equal cannot see
        # this; the dataclass fields are typed tuple[float, float] and
        # [a, b] != (a, b), so a list here would break equality downstream.
        schema = CommonSchema(
            metadata=Metadata(name="m"),
            links=[Link(name="a"), Link(name="b")],
            joints=[
                Joint(name="j", type=JointType.REVOLUTE, parent_link="a", child_link="b")
            ],
            actuators=[
                Actuator(
                    name="m1",
                    joint="j",
                    type=ActuatorType.SERVO,
                    control_range=(-1.0, 1.0),
                    force_range=(-2.0, 2.0),
                )
            ],
        )
        actuator = round_trip(schema).actuators[0]
        assert isinstance(actuator.control_range, tuple)
        assert isinstance(actuator.force_range, tuple)
        assert actuator.control_range == (-1.0, 1.0)
        assert actuator.force_range == (-2.0, 2.0)

    def test_collision_group_bitmasks_stay_ints(self):
        schema = CommonSchema(
            metadata=Metadata(name="m"),
            links=[Link(name="a")],
            collision_config=CollisionConfig(
                groups={"G": CollisionGroup(name="G", contype=8, conaffinity=16)}
            ),
        )
        group = round_trip(schema).collision_config.groups["G"]
        assert isinstance(group.contype, int) and not isinstance(group.contype, bool)
        assert group.contype == 8
        assert group.conaffinity == 16

    def test_scene_booleans_stay_booleans(self):
        schema = CommonSchema(
            metadata=Metadata(name="m"),
            links=[Link(name="a")],
            scene=Scene(shadows=False, grid=True, origin_visual=False),
        )
        scene = round_trip(schema).scene
        assert scene.shadows is False
        assert scene.grid is True
        assert scene.origin_visual is False


class TestNamesAndMaterials:
    @pytest.mark.parametrize(
        "names",
        [
            ["wheel-fl.001", "arm1__link-2"],
            ["3dof", "with space"],
            # Both sanitize to "a_b", so one gets a collision suffix and both must
            # still recover exactly.
            ["a-b", "a.b"],
            ["läng", "naïve"],
            ["Links", "Joints"],
        ],
    )
    def test_link_names_round_trip_through_sanitization(self, names):
        schema = CommonSchema(
            metadata=Metadata(name="names"),
            links=[Link(name=name) for name in names],
        )
        assert [link.name for link in round_trip(schema).links] == names

    def test_joint_names_round_trip_through_sanitization(self):
        schema = CommonSchema(
            metadata=Metadata(name="names"),
            links=[Link(name="a"), Link(name="b"), Link(name="c")],
            joints=[
                Joint(name="a-b", type=JointType.FIXED, parent_link="a", child_link="b"),
                Joint(name="a.b", type=JointType.FIXED, parent_link="b", child_link="c"),
            ],
        )
        assert [joint.name for joint in round_trip(schema).joints] == ["a-b", "a.b"]

    def test_metadata_name_needing_sanitization_round_trips(self):
        schema = CommonSchema(metadata=Metadata(name="my robot-2.0"), links=[Link(name="a")])
        assert round_trip(schema).metadata.name == "my robot-2.0"

    def test_visual_name_none_and_named_are_distinguished(self):
        schema = CommonSchema(
            metadata=Metadata(name="m"),
            links=[
                Link(
                    name="a",
                    visuals=[
                        Visual(name=None, geometry=Geometry(type=GeometryType.SPHERE, radius=1.0)),
                        Visual(name="named", geometry=Geometry(type=GeometryType.SPHERE, radius=2.0)),
                    ],
                    collisions=[
                        Collision(name=None, geometry=Geometry(type=GeometryType.SPHERE, radius=3.0)),
                        Collision(name="col", geometry=Geometry(type=GeometryType.SPHERE, radius=4.0)),
                    ],
                )
            ],
        )
        link = round_trip(schema).links[0]
        assert [visual.name for visual in link.visuals] == [None, "named"]
        assert [collision.name for collision in link.collisions] == [None, "col"]

    def test_two_materials_with_the_same_name_both_survive(self):
        # A named material is shared by path, so a second, *different* material
        # under the same name cannot own the shared prim — it is authored inline
        # instead. Merged multi-robot schemas hit this with names like "grey".
        schema = CommonSchema(
            metadata=Metadata(name="m"),
            links=[
                Link(
                    name="a",
                    visuals=[
                        Visual(
                            geometry=Geometry(type=GeometryType.SPHERE, radius=1.0),
                            material=Material(name="paint", color=[1.0, 0.0, 0.0, 1.0]),
                        )
                    ],
                ),
                Link(
                    name="b",
                    visuals=[
                        Visual(
                            geometry=Geometry(type=GeometryType.SPHERE, radius=1.0),
                            material=Material(name="paint", color=[0.0, 1.0, 0.0, 1.0]),
                        )
                    ],
                ),
            ],
        )
        links = round_trip(schema).links
        assert [link.visuals[0].material.name for link in links] == ["paint", "paint"]
        assert links[0].visuals[0].material.color == [1.0, 0.0, 0.0, 1.0]
        assert links[1].visuals[0].material.color == [0.0, 1.0, 0.0, 1.0]

    @pytest.mark.parametrize(
        "geometry", [None, Geometry(type=GeometryType.UNKNOWN)], ids=["none", "unknown"]
    )
    def test_unnamed_material_on_a_shapeless_visual_stays_unnamed(self, geometry):
        # A visual with no shape exports as an Xform rather than a Gprim, so the
        # inline material must still be recognised as inline and import as
        # name=None — an invented name would then be treated as a shared one.
        schema = CommonSchema(
            metadata=Metadata(name="m"),
            links=[
                Link(
                    name="a",
                    visuals=[
                        Visual(
                            geometry=geometry,
                            material=Material(color=[0.5, 0.5, 0.5, 1.0]),
                        )
                    ],
                )
            ],
        )
        material = round_trip(schema).links[0].visuals[0].material
        assert material.name is None
        assert material.color == [0.5, 0.5, 0.5, 1.0]

    def test_empty_metadata_name_keeps_the_rest_of_the_metadata(self):
        schema = CommonSchema(
            metadata=Metadata(
                name="", version="9.9", author="me", description="d", units="SI"
            ),
            links=[Link(name="a")],
        )
        parsed = round_trip(schema).metadata
        assert parsed.name == ""
        assert parsed.version == "9.9"
        assert parsed.author == "me"
        assert parsed.description == "d"

    def test_two_visuals_with_the_same_name_both_survive(self):
        schema = CommonSchema(
            metadata=Metadata(name="m"),
            links=[
                Link(
                    name="a",
                    visuals=[
                        Visual(name="dup", geometry=Geometry(type=GeometryType.SPHERE, radius=1.0)),
                        Visual(name="dup", geometry=Geometry(type=GeometryType.SPHERE, radius=2.0)),
                    ],
                )
            ],
        )
        visuals = round_trip(schema).links[0].visuals
        assert [visual.name for visual in visuals] == ["dup", "dup"]
        assert [visual.geometry.radius for visual in visuals] == [1.0, 2.0]

    def test_unnamed_material_stays_unnamed_and_named_is_shared(self):
        shared = Material(name="paint", color=[0.1, 0.2, 0.3, 1.0])
        schema = CommonSchema(
            metadata=Metadata(name="m"),
            links=[
                Link(
                    name="a",
                    visuals=[
                        Visual(name="v1", geometry=Geometry(type=GeometryType.SPHERE, radius=1.0), material=shared),
                        Visual(name="v2", geometry=Geometry(type=GeometryType.SPHERE, radius=1.0), material=shared),
                        Visual(
                            name="v3",
                            geometry=Geometry(type=GeometryType.SPHERE, radius=1.0),
                            material=Material(color=[1.0, 0.0, 0.0, 1.0]),
                        ),
                    ],
                )
            ],
        )
        visuals = round_trip(schema).links[0].visuals
        assert visuals[0].material.name == "paint"
        assert visuals[1].material.name == "paint"
        assert visuals[2].material.name is None
        assert visuals[2].material.color == [1.0, 0.0, 0.0, 1.0]

    def test_material_literally_named_material_round_trips(self):
        # The reserved prim name for an inline unnamed material is "Material", so a
        # material genuinely called that must not come back as None.
        schema = CommonSchema(
            metadata=Metadata(name="m"),
            links=[
                Link(
                    name="a",
                    visuals=[
                        Visual(
                            name="v",
                            geometry=Geometry(type=GeometryType.SPHERE, radius=1.0),
                            material=Material(name="Material", color=[0.5, 0.5, 0.5, 1.0]),
                        )
                    ],
                )
            ],
        )
        assert round_trip(schema).links[0].visuals[0].material.name == "Material"

    def test_material_with_three_component_color_keeps_its_length(self):
        schema = CommonSchema(
            metadata=Metadata(name="m"),
            links=[
                Link(
                    name="a",
                    visuals=[
                        Visual(
                            name="v",
                            geometry=Geometry(type=GeometryType.SPHERE, radius=1.0),
                            material=Material(name="rgb", color=[0.25, 0.5, 0.75]),
                        )
                    ],
                )
            ],
        )
        assert round_trip(schema).links[0].visuals[0].material.color == [0.25, 0.5, 0.75]


class TestOrdering:
    def test_link_and_joint_order_is_preserved(self):
        names = [f"link_{index}" for index in range(12)]
        schema = CommonSchema(
            metadata=Metadata(name="ordered"),
            links=[Link(name=name) for name in names],
            joints=[
                Joint(
                    name=f"j_{index}",
                    type=JointType.FIXED,
                    parent_link=names[index],
                    child_link=names[index + 1],
                )
                for index in range(11)
            ],
        )
        parsed = round_trip(schema)
        assert [link.name for link in parsed.links] == names
        assert [joint.name for joint in parsed.joints] == [
            f"j_{index}" for index in range(11)
        ]

    def test_collision_pair_order_survives_double_digits(self):
        # pair_10 must not sort before pair_2, which plain prim-name ordering would do.
        pairs = [
            CollisionPair(geom1=f"g{index}", geom2=f"h{index}", condim=index)
            for index in range(12)
        ]
        schema = CommonSchema(
            metadata=Metadata(name="pairs"),
            links=[Link(name="a")],
            collision_config=CollisionConfig(pairs=pairs),
        )
        parsed = round_trip(schema)
        assert [pair.geom1 for pair in parsed.collision_config.pairs] == [
            f"g{index}" for index in range(12)
        ]

    def test_sensor_order_is_preserved_across_links_and_world(self):
        schema = CommonSchema(
            metadata=Metadata(name="sensors"),
            links=[Link(name="a"), Link(name="b")],
            sensors=[
                Sensor(name="s_world", type="lidar", parent_link="world"),
                Sensor(name="s_b", type="IMU", parent_link="b"),
                Sensor(name="s_a", type="camera", parent_link="a"),
            ],
        )
        parsed = round_trip(schema)
        assert [sensor.name for sensor in parsed.sensors] == ["s_world", "s_b", "s_a"]
        assert [sensor.parent_link for sensor in parsed.sensors] == ["world", "b", "a"]

    def test_visual_and_collision_order_is_preserved(self):
        schema = CommonSchema(
            metadata=Metadata(name="m"),
            links=[
                Link(
                    name="a",
                    visuals=[
                        Visual(name=f"v{index}", geometry=Geometry(type=GeometryType.SPHERE, radius=float(index) + 1.0))
                        for index in range(5)
                    ],
                )
            ],
        )
        visuals = round_trip(schema).links[0].visuals
        assert [visual.name for visual in visuals] == [f"v{index}" for index in range(5)]


class TestExactFields:
    """Fields that travel through double sidecars must be bit-exact, not close."""

    def test_joint_pose_axis_and_limits_are_exact(self):
        schema = full_schema()
        parsed = round_trip(schema)
        for original, actual in zip(schema.joints, parsed.joints):
            assert actual.pose.position == original.pose.position
            assert actual.pose.orientation == original.pose.orientation
            assert actual.axis == original.axis
            assert actual.limits == original.limits
            assert actual.dynamics == original.dynamics
            assert actual.home_position == original.home_position
            assert actual.mimic == original.mimic
            assert actual.safety_controller == original.safety_controller
            assert actual.calibration == original.calibration

    def test_link_inertials_are_exact(self):
        schema = full_schema()
        parsed = round_trip(schema)
        for original, actual in zip(schema.links, parsed.links):
            assert actual.mass == original.mass
            assert actual.center_of_mass == original.center_of_mass
            assert actual.inertia == original.inertia

    def test_geometry_dimensions_are_exact(self):
        schema = full_schema()
        parsed = round_trip(schema)
        for original, actual in zip(schema.links, parsed.links):
            for source, target in zip(original.visuals, actual.visuals):
                assert target.geometry == source.geometry
            for source, target in zip(original.collisions, actual.collisions):
                assert target.geometry == source.geometry

    def test_physics_and_contact_surfaces_are_exact(self):
        schema = full_schema()
        parsed = round_trip(schema)
        assert parsed.physics == schema.physics
        assert parsed.contacts == schema.contacts
        assert parsed.scene == schema.scene


class TestExtensions:
    def test_nested_extensions_survive_on_every_object(self):
        schema = full_schema()
        parsed = round_trip(schema)
        assert parsed.extensions == NESTED_EXTENSIONS
        assert parsed.metadata.extensions == NESTED_EXTENSIONS
        assert parsed.links[0].extensions == NESTED_EXTENSIONS
        assert parsed.links[0].visuals[0].extensions == NESTED_EXTENSIONS
        assert parsed.links[0].visuals[0].geometry.extensions == NESTED_EXTENSIONS
        assert parsed.links[0].collisions[0].extensions == NESTED_EXTENSIONS
        assert parsed.joints[0].extensions == NESTED_EXTENSIONS
        assert parsed.actuators[0].extensions == NESTED_EXTENSIONS
        assert parsed.sensors[0].extensions == NESTED_EXTENSIONS
        assert parsed.contacts[0].extensions == NESTED_EXTENSIONS
        assert parsed.physics.extensions == NESTED_EXTENSIONS
        assert parsed.physics.solver.extensions == NESTED_EXTENSIONS
        assert parsed.scene.extensions == NESTED_EXTENSIONS
        assert parsed.collision_config.extensions == NESTED_EXTENSIONS
        assert parsed.collision_config.pairs[0].extensions == NESTED_EXTENSIONS

    def test_empty_extensions_stay_empty(self):
        schema = CommonSchema(metadata=Metadata(name="m"), links=[Link(name="a")])
        parsed = round_trip(schema)
        assert parsed.extensions == {}
        assert parsed.links[0].extensions == {}

    def test_dataclass_inside_extensions_is_serialized_via_cattrs(self):
        # URDFParser and MJCFParser both stash live Material instances in
        # extensions["parse_context"], so plain json.dumps would raise here.
        schema = CommonSchema(metadata=Metadata(name="m"), links=[Link(name="a")])
        schema.extensions["materials"] = {"red": Material(name="red", color=[1.0, 0, 0, 1])}
        parsed = round_trip(schema)
        assert parsed.extensions["materials"]["red"]["name"] == "red"

    def test_truly_unserializable_extension_raises_type_error(self):
        schema = CommonSchema(metadata=Metadata(name="m"), links=[Link(name="a")])
        schema.extensions["bad"] = object()
        with pytest.raises(TypeError):
            USDExporter().export_to_string(schema)


class TestComposedScenes:
    def test_merge_in_composed_scene_round_trips(self):
        arm = CommonSchema(
            metadata=Metadata(name="arm"),
            links=[
                Link(
                    name="base",
                    mass=1.0,
                    visuals=[
                        Visual(
                            name="v",
                            geometry=Geometry(
                                type=GeometryType.BOX,
                                size=Vector3(x=0.1, y=0.1, z=0.1),
                            ),
                        )
                    ],
                ),
                Link(name="tip", mass=0.5),
            ],
            joints=[
                Joint(
                    name="elbow",
                    type=JointType.REVOLUTE,
                    parent_link="base",
                    child_link="tip",
                    pose=Pose(position=Vector3(z=0.2)),
                    axis=Vector3(y=1.0),
                    limits=JointLimits(lower=-1.0, upper=1.0),
                    mimic=MimicJoint(joint="elbow", multiplier=1.0),
                )
            ],
            actuators=[Actuator(name="motor", joint="elbow", type=ActuatorType.SERVO)],
            sensors=[Sensor(name="tip_imu", type="IMU", parent_link="tip")],
        )
        rover = CommonSchema(
            metadata=Metadata(name="rover"),
            links=[Link(name="chassis", mass=10.0)],
        )

        scene = CommonSchema(metadata=Metadata(name="scene"))
        scene.merge_in(arm, "arm1", spawn_pose=Pose(position=Vector3(x=1.0)), fixed_base=True)
        scene.merge_in(rover, "rover1", spawn_pose=Pose(position=Vector3(y=2.0)), fixed_base=False)

        assert scene.validate() == []
        assert_schema_equal(scene, round_trip(scene))

        parsed = round_trip(scene)
        assert {link.name for link in parsed.links} == {
            "arm1__base",
            "arm1__tip",
            "rover1__chassis",
        }
        spawn_types = {
            joint.name: joint.type for joint in parsed.joints if joint.name.endswith("spawn")
        }
        assert spawn_types["arm1__spawn"] == JointType.FIXED
        assert spawn_types["rover1__spawn"] == JointType.FLOATING


class TestRealFixtures:
    @pytest.mark.parametrize(
        "fixture",
        [
            "simple_robot.urdf",
            "piper_description.urdf",
            "piper_description-mimic-joint.urdf",
            "arm_two_prismatic.urdf",
            "gripper_prismatic_pair.urdf",
        ],
    )
    def test_urdf_fixture_round_trips(self, fixture):
        schema = URDFParser().parse(FIXTURES / fixture)
        # parse_context holds live Material dataclass instances the parser keeps for
        # diagnostics; it is not model data and does not survive JSON verbatim.
        schema.extensions.pop("parse_context", None)
        assert_schema_equal(schema, round_trip(schema))

    def test_mjcf_fixture_round_trips(self):
        schema = MJCFParser().parse(FIXTURES / "simple_robot.xml")
        schema.extensions.pop("parse_context", None)
        assert_schema_equal(schema, round_trip(schema))

    def test_piper_fixture_export_is_idempotent(self):
        schema = URDFParser().parse(FIXTURES / "piper_description.urdf")
        schema.extensions.pop("parse_context", None)
        first = USDExporter().export_to_string(schema)
        assert USDExporter().export_to_string(USDParser().parse_string(first)) == first


class TestDegenerateInput:
    def test_zero_axis_joint_round_trips_exactly(self):
        schema = CommonSchema(
            metadata=Metadata(name="m"),
            links=[Link(name="a"), Link(name="b")],
            joints=[
                Joint(
                    name="j",
                    type=JointType.REVOLUTE,
                    parent_link="a",
                    child_link="b",
                    axis=Vector3(),
                )
            ],
        )
        assert round_trip(schema).joints[0].axis == Vector3()

    def test_antiparallel_axis_round_trips_exactly(self):
        schema = CommonSchema(
            metadata=Metadata(name="m"),
            links=[Link(name="a"), Link(name="b")],
            joints=[
                Joint(
                    name="j",
                    type=JointType.REVOLUTE,
                    parent_link="a",
                    child_link="b",
                    axis=Vector3(x=-1.0),
                )
            ],
        )
        assert round_trip(schema).joints[0].axis == Vector3(x=-1.0)

    def test_unnormalized_axis_is_preserved_verbatim(self):
        schema = CommonSchema(
            metadata=Metadata(name="m"),
            links=[Link(name="a"), Link(name="b")],
            joints=[
                Joint(
                    name="j",
                    type=JointType.REVOLUTE,
                    parent_link="a",
                    child_link="b",
                    axis=Vector3(z=2.5),
                )
            ],
        )
        assert round_trip(schema).joints[0].axis == Vector3(z=2.5)

    def test_schema_with_no_links_round_trips(self):
        schema = CommonSchema(metadata=Metadata(name="empty"))
        parsed = round_trip(schema)
        assert parsed.links == []
        assert parsed.joints == []
        assert parsed.metadata.name == "empty"

    def test_dangling_joint_references_are_preserved(self):
        schema = CommonSchema(
            metadata=Metadata(name="m"),
            links=[Link(name="a")],
            joints=[
                Joint(
                    name="j",
                    type=JointType.FIXED,
                    parent_link="ghost_parent",
                    child_link="a",
                )
            ],
        )
        # validate() only reports this, so export must not lose the name.
        assert schema.validate() != []
        assert round_trip(schema).joints[0].parent_link == "ghost_parent"

    def test_real_world_link_binds_body0(self):
        schema = CommonSchema(
            metadata=Metadata(name="m"),
            links=[Link(name="world"), Link(name="a")],
            joints=[
                Joint(
                    name="j",
                    type=JointType.FIXED,
                    parent_link="world",
                    child_link="a",
                    pose=Pose(position=Vector3(z=1.0)),
                )
            ],
        )
        parsed = round_trip(schema)
        assert [link.name for link in parsed.links] == ["world", "a"]
        assert parsed.joints[0].parent_link == "world"
