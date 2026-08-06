import io
import zipfile
import xml.etree.ElementTree as ET

import pytest

from cyberwave_robot_format.mjcf import export_mujoco_zip_cloud
from cyberwave_robot_format.schema import (
    Actuator,
    ActuatorType,
    Collision,
    CommonSchema,
    Geometry,
    GeometryType,
    Inertia,
    Joint,
    JointLimits,
    JointType,
    Link,
    Metadata,
    Pose,
    Vector3,
    Visual,
)


def _mesh_schema() -> CommonSchema:
    mesh_geometry = Geometry(
        type=GeometryType.MESH,
        filename="meshes/missing_body.dae",
        scale=Vector3(1.0, 1.0, 1.0),
    )
    return CommonSchema(
        metadata=Metadata(name="missing_mesh_scene"),
        links=[
            Link(
                name="base",
                mass=1.0,
                inertia=Inertia(ixx=0.1, iyy=0.1, izz=0.1),
                visuals=[Visual(name="body_visual", geometry=mesh_geometry)],
                collisions=[Collision(name="body_collision", geometry=mesh_geometry)],
            )
        ],
    )


def test_export_mujoco_zip_cloud_falls_back_when_mesh_is_missing() -> None:
    zip_bytes = export_mujoco_zip_cloud(_mesh_schema(), mesh_resolver=lambda _: None)

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        xml_content = zf.read("mujoco_scene.xml")
        assert not any(name.startswith("assets/") for name in zf.namelist())

    root = ET.fromstring(xml_content)
    assert root.findall(".//asset/mesh") == []

    geoms = root.findall(".//worldbody//geom")
    assert all(geom.get("mesh") is None for geom in geoms)
    collision_geoms = [
        geom
        for geom in geoms
        if geom.get("class") == "collision" or geom.get("group") == "3"
    ]
    assert len(collision_geoms) == 1
    assert collision_geoms[0].get("type") == "box"
    assert collision_geoms[0].get("size") == "0.025 0.025 0.025"


def test_export_mujoco_zip_cloud_can_require_meshes() -> None:
    with pytest.raises(FileNotFoundError, match="missing_body.dae"):
        export_mujoco_zip_cloud(
            _mesh_schema(),
            mesh_resolver=lambda _: None,
            strict_missing_meshes=True,
        )


def test_export_mujoco_zip_cloud_adds_valid_inertial_for_moving_zero_mass_link() -> (
    None
):
    schema = CommonSchema(
        metadata=Metadata(name="zero_mass_joint_scene"),
        links=[
            Link(
                name="base",
                mass=1.0,
                inertia=Inertia(ixx=0.1, iyy=0.1, izz=0.1),
            ),
            Link(name="wheel", mass=0.0),
        ],
        joints=[
            Joint(
                name="wheel_joint",
                type=JointType.CONTINUOUS,
                parent_link="base",
                child_link="wheel",
                pose=Pose(position=Vector3(0.1, 0.0, 0.0)),
            )
        ],
    )

    zip_bytes = export_mujoco_zip_cloud(schema, mesh_resolver=lambda _: None)

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        root = ET.fromstring(zf.read("mujoco_scene.xml"))

    wheel_body = root.find(".//body[@name='wheel']")
    assert wheel_body is not None
    inertial = wheel_body.find("inertial")
    assert inertial is not None
    assert inertial.get("pos") == "0 0 0"
    # Must match the exporter's fallback constants (_FALLBACK_MASS /
    # _FALLBACK_DIAGINERTIA in mjcf/exporter.py) for movable zero-mass links.
    assert inertial.get("mass") == "1e-06"
    assert inertial.get("diaginertia") == "1e-9 1e-9 1e-9"


def _arm_schema(
    *,
    continuous_limits: JointLimits | None,
    continuous_home: float | None,
) -> CommonSchema:
    """A base + two 1-DOF links: a continuous joint and a bounded revolute,
    each driven by a position actuator."""
    return CommonSchema(
        metadata=Metadata(name="continuous_joint_scene"),
        links=[
            Link(name="base", mass=1.0, inertia=Inertia(ixx=0.1, iyy=0.1, izz=0.1)),
            Link(
                name="spinner", mass=0.5, inertia=Inertia(ixx=0.01, iyy=0.01, izz=0.01)
            ),
            Link(name="elbow", mass=0.5, inertia=Inertia(ixx=0.01, iyy=0.01, izz=0.01)),
        ],
        joints=[
            Joint(
                name="spin_joint",
                type=JointType.CONTINUOUS,
                parent_link="base",
                child_link="spinner",
                pose=Pose(position=Vector3(0.0, 0.0, 0.1)),
                axis=Vector3(0.0, 0.0, 1.0),
                limits=continuous_limits,
                home_position=continuous_home,
            ),
            Joint(
                name="elbow_joint",
                type=JointType.REVOLUTE,
                parent_link="spinner",
                child_link="elbow",
                pose=Pose(position=Vector3(0.0, 0.0, 0.1)),
                axis=Vector3(0.0, 1.0, 0.0),
                limits=JointLimits(lower=-1.5, upper=1.5, effort=10.0),
            ),
        ],
        actuators=[
            Actuator(name="spin_act", joint="spin_joint", type=ActuatorType.POSITION),
            Actuator(name="elbow_act", joint="elbow_joint", type=ActuatorType.POSITION),
        ],
    )


def test_continuous_joint_exports_unlimited_even_with_limits() -> None:
    # A continuous joint that happens to carry stray positional limits (from a
    # messy source URDF) must still export as an honest limited="false" free
    # hinge with no positional range — the limits are ignored, not lied about.
    schema = _arm_schema(
        continuous_limits=JointLimits(
            lower=-62.83, upper=62.83, effort=39.0, velocity=1.0
        ),
        continuous_home=None,
    )
    zip_bytes = export_mujoco_zip_cloud(schema, mesh_resolver=lambda _: None)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        root = ET.fromstring(zf.read("mujoco_scene.xml"))

    spin = root.find(".//joint[@name='spin_joint']")
    assert spin is not None
    assert spin.get("type") == "hinge"
    assert spin.get("limited") == "false"
    assert spin.get("range") is None

    # The bounded revolute keeps its range and stays a limited hinge.
    elbow = root.find(".//joint[@name='elbow_joint']")
    assert elbow is not None
    assert elbow.get("range") == "-1.5 1.5"
    assert elbow.get("limited") != "false"


def test_continuous_joint_actuator_ctrlrange_is_home_plus_minus_pi() -> None:
    # The finite command band lives on the actuator, centered on home ± π.
    import math

    schema = _arm_schema(
        continuous_limits=JointLimits(
            lower=-62.83, upper=62.83, effort=39.0, velocity=1.0
        ),
        continuous_home=None,  # → home 0.0 → ctrlrange [-π, π]
    )
    zip_bytes = export_mujoco_zip_cloud(schema, mesh_resolver=lambda _: None)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        root = ET.fromstring(zf.read("mujoco_scene.xml"))

    spin_act = root.find(".//actuator/position[@name='spin_act']")
    assert spin_act is not None
    ctrlrange = spin_act.get("ctrlrange")
    assert ctrlrange is not None
    lo, hi = (float(v) for v in ctrlrange.split())
    assert lo == pytest.approx(-math.pi)
    assert hi == pytest.approx(math.pi)

    # A bounded joint's actuator ctrlrange still matches its joint limits.
    elbow_act = root.find(".//actuator/position[@name='elbow_act']")
    assert elbow_act is not None
    elo, ehi = (float(v) for v in elbow_act.get("ctrlrange").split())
    assert elo == pytest.approx(-1.5)
    assert ehi == pytest.approx(1.5)


def test_continuous_joint_actuator_ctrlrange_uses_home_offset() -> None:
    # A non-zero home shifts the actuator band to home ± π.
    import math

    home = -math.pi / 2.0
    schema = _arm_schema(continuous_limits=None, continuous_home=home)
    zip_bytes = export_mujoco_zip_cloud(schema, mesh_resolver=lambda _: None)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        root = ET.fromstring(zf.read("mujoco_scene.xml"))

    spin_act = root.find(".//actuator/position[@name='spin_act']")
    assert spin_act is not None
    lo, hi = (float(v) for v in spin_act.get("ctrlrange").split())
    assert lo == pytest.approx(home - math.pi)
    assert hi == pytest.approx(home + math.pi)


def _wheel_schema(*, max_velocity: float | None) -> CommonSchema:
    """A single continuous wheel joint driven by a velocity actuator.

    No ``control_range``, which is the case the exporter has to answer for
    itself — and the case a real asset hits, since a wheel's command band is a
    speed, not something the joint can supply.
    """
    return CommonSchema(
        metadata=Metadata(name="wheel_scene"),
        links=[
            Link(name="base", mass=1.0, inertia=Inertia(ixx=0.1, iyy=0.1, izz=0.1)),
            Link(name="wheel", mass=0.5, inertia=Inertia(ixx=0.01, iyy=0.01, izz=0.01)),
        ],
        joints=[
            Joint(
                name="wheel_joint",
                type=JointType.CONTINUOUS,
                parent_link="base",
                child_link="wheel",
                pose=Pose(position=Vector3(0.0, 0.2, 0.0)),
                axis=Vector3(0.0, 1.0, 0.0),
                limits=None,
                home_position=None,
            )
        ],
        actuators=[
            Actuator(
                name="wheel_act",
                joint="wheel_joint",
                type=ActuatorType.VELOCITY,
                max_torque=100.0,
                max_velocity=max_velocity,
                kd=35.0,
            )
        ],
    )


def _wheel_actuator(schema: CommonSchema) -> ET.Element:
    zip_bytes = export_mujoco_zip_cloud(schema, mesh_resolver=lambda _: None)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        root = ET.fromstring(zf.read("mujoco_scene.xml"))
    actuator = root.find(".//actuator/velocity[@name='wheel_act']")
    assert actuator is not None
    return actuator


def test_velocity_actuator_ctrlrange_is_a_speed_band() -> None:
    """A velocity servo's ``ctrl`` is rad/s, so its ctrlrange comes from max_velocity.

    Sharing the position branch handed a drive wheel its continuous joint's
    ``home ± π`` band, capping it at π rad/s — 0.17 m/s on a 53 mm wheel, a robot
    merely crawling with nothing logged anywhere.
    """
    import math

    actuator = _wheel_actuator(_wheel_schema(max_velocity=20.0))

    lo, hi = (float(v) for v in actuator.get("ctrlrange").split())
    assert lo == pytest.approx(-20.0)
    assert hi == pytest.approx(20.0)
    # The bug's signature: the joint's positional band, not the wheel's speed.
    assert hi != pytest.approx(math.pi)
    # kd is still emitted as the velocity servo's kv gain.
    assert actuator.get("kv") == "35.0"


def test_velocity_actuator_ctrlrange_omitted_without_max_velocity(caplog) -> None:
    """No declared ceiling exports a loose actuator, not a frozen one.

    ``compiler autolimits="true"`` reads an absent ctrlrange as "no limit", so
    inventing a number here would be worse than omitting it. It is warned about
    because an unbounded wheel command reaches the contact solver as a skid.
    """
    import logging

    with caplog.at_level(logging.WARNING):
        actuator = _wheel_actuator(_wheel_schema(max_velocity=None))

    assert actuator.get("ctrlrange") is None
    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING and "wheel_act" in record.getMessage()
    ]
    assert warnings, "expected a warning naming the unbounded velocity actuator"
    assert "max_velocity" in warnings[0]
