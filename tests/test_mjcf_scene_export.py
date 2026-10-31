import io
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from cyberwave_robot_format.mjcf import export_mujoco_zip_cloud
from cyberwave_robot_format.mjcf.utils import sanitize_inertia
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
from cyberwave_robot_format.urdf import URDFParser


@pytest.mark.parametrize("nested", [False, True])
def test_empty_visual_box_keeps_frames_joints_and_collision(nested, caplog) -> None:
    solid = Geometry(type=GeometryType.BOX, size=Vector3(0.1, 0.2, 0.3))
    empty = Geometry(type=GeometryType.BOX, size=Vector3(0, 0, 0))
    marker = Link(
        name="marker",
        mass=0.1,
        inertia=Inertia(ixx=0.001, iyy=0.001, izz=0.001),
        visuals=[Visual(geometry=empty), Visual(geometry=solid)],
        collisions=[Collision(name="solid_collision", geometry=solid)],
    )
    schema = CommonSchema(metadata=Metadata(name="frame_marker"), links=[marker])
    if nested:
        schema.links.insert(0, Link(name="base"))
        schema.joints.append(
            Joint(
                name="marker_joint",
                type=JointType.REVOLUTE,
                parent_link="base",
                child_link="marker",
                axis=Vector3(0, 0, 1),
                limits=JointLimits(lower=-1, upper=1),
            )
        )
    archive = export_mujoco_zip_cloud(schema, mesh_resolver=lambda _: None)
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        xml = zf.read("mujoco_scene.xml").decode()
    root = ET.fromstring(xml)
    body = root.find(".//body[@name='marker']")
    assert body is not None
    assert len(body.findall("geom")) == 2
    assert body.find("geom[@name='solid_collision']") is not None
    assert "Omitting empty visual box on body marker" in caplog.text
    assert empty.size == Vector3(0, 0, 0)
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(xml)
    assert model.body("marker").id > 0
    if nested:
        assert model.joint("marker_joint").id >= 0
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data, nstep=20)
    assert np.isfinite(data.qpos).all()


@pytest.mark.parametrize(
    "collision,size", [(True, (0, 0, 0)), (False, (0, 1, 1)), (False, (-1, 0, 0))]
)
def test_export_does_not_hide_invalid_collision_or_nonempty_box(
    collision, size
) -> None:
    geometry = Geometry(type=GeometryType.BOX, size=Vector3(*size))
    link = Link(name="invalid")
    if collision:
        link.collisions.append(Collision(geometry=geometry))
    else:
        link.visuals.append(Visual(geometry=geometry))
    archive = export_mujoco_zip_cloud(
        CommonSchema(metadata=Metadata(name="invalid"), links=[link]),
        mesh_resolver=lambda _: None,
    )
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        xml = zf.read("mujoco_scene.xml").decode()
    assert ET.fromstring(xml).find(".//worldbody//geom") is not None
    mujoco = pytest.importorskip("mujoco")
    with pytest.raises(ValueError, match="size .* must be positive"):
        mujoco.MjModel.from_xml_string(xml)


def test_export_repairs_anymal_camera_principal_inertia() -> None:
    # Public ANYmal C camera tensor: positive definite with valid diagonals,
    # but its principal moments fail MuJoCo's rigid-body triangle inequality.
    inertia = Inertia(
        ixx=0.000637051, iyy=0.000929402, izz=0.000484569, ixy=0.000484569, ixz=0.000424211, iyz=0.000005138
    )
    schema = CommonSchema(
        metadata=Metadata(name="anymal_camera"), links=[Link(name="camera", mass=0.16043, inertia=inertia)]
    )
    archive = export_mujoco_zip_cloud(schema, mesh_resolver=lambda _: None)
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        xml = zf.read("mujoco_scene.xml").decode()
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(xml)
    moments = np.sort(model.body_inertia[model.body("camera").id])
    assert moments[0] > 0
    assert moments[0] + moments[1] >= moments[2]


@pytest.mark.parametrize("moments", [(1.0, 1.0, 3.0), (1.0, 2.0, 2.5)])
def test_inertia_repair_is_frame_independent(moments) -> None:
    angle = 0.73
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    tensor = rotation @ np.diag(moments) @ rotation.T
    link = Link(
        name="rotated",
        mass=1.0,
        inertia=Inertia(
            ixx=tensor[0, 0], iyy=tensor[1, 1], izz=tensor[2, 2], ixy=tensor[0, 1], ixz=tensor[0, 2], iyz=tensor[1, 2]
        ),
    )
    result = sanitize_inertia(link)
    repaired = np.array(
        [[result[0], result[3], result[4]], [result[3], result[1], result[5]], [result[4], result[5], result[2]]]
    )
    expected = np.array(moments)
    if sum(moments[:2]) < moments[2]:
        expected[:2] += (moments[2] * 1.01 - sum(moments[:2])) / 2
    np.testing.assert_allclose(repaired, rotation @ np.diag(expected) @ rotation.T, atol=1e-12)


def test_rotated_urdf_inertial_frame_reaches_the_mjcf_tensor(tmp_path) -> None:
    """URDF ``<inertial><origin rpy>`` rotates the tensor; MJCF must see it in the body frame."""
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(
        '<robot name="r"><link name="base"><inertial><origin xyz="0 0 0" rpy="0 0 0.785"/>'
        '<mass value="2"/><inertia ixx="1" iyy="2" izz="2.5" ixy="0" ixz="0" iyz="0"/>'
        "</inertial></link></robot>"
    )
    schema = URDFParser().parse(urdf)
    archive = export_mujoco_zip_cloud(schema, mesh_resolver=lambda _: None)
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        xml = zf.read("mujoco_scene.xml").decode()

    inertial = ET.fromstring(xml).find(".//body[@name='base']/inertial")
    ixx, iyy, izz, ixy, ixz, iyz = (float(v) for v in inertial.get("fullinertia").split())
    yaw = np.array([[np.cos(0.785), -np.sin(0.785), 0], [np.sin(0.785), np.cos(0.785), 0], [0, 0, 1]])
    np.testing.assert_allclose(
        [[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]],
        yaw @ np.diag([1.0, 2.0, 2.5]) @ yaw.T,
        atol=1e-9,
    )

    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(xml)
    body = model.body("base").id
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, model.body_iquat[body])
    rotation = rotation.reshape(3, 3)
    np.testing.assert_allclose(
        rotation @ np.diag(model.body_inertia[body]) @ rotation.T,
        yaw @ np.diag([1.0, 2.0, 2.5]) @ yaw.T,
        atol=1e-6,
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
    collision_geoms = [geom for geom in geoms if geom.get("class") == "collision" or geom.get("group") == "3"]
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


def test_export_mujoco_zip_cloud_adds_valid_inertial_for_moving_zero_mass_link() -> None:
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
            Link(name="spinner", mass=0.5, inertia=Inertia(ixx=0.01, iyy=0.01, izz=0.01)),
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
        continuous_limits=JointLimits(lower=-62.83, upper=62.83, effort=39.0, velocity=1.0),
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
        continuous_limits=JointLimits(lower=-62.83, upper=62.83, effort=39.0, velocity=1.0),
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


# --------------------------------------------------------------------------- #
# The fallback <motor> (no declared actuators) bounds torque, not angle
# --------------------------------------------------------------------------- #


def _fallback_motor_xml(joint_kwargs: dict) -> str:
    """Export a one-joint schema that declares NO actuators, so the fallback runs."""
    import tempfile
    from pathlib import Path

    from cyberwave_robot_format.mjcf import MJCFExporter

    limits = (
        JointLimits(**joint_kwargs.pop("limits")) if "limits" in joint_kwargs else None
    )
    joint = Joint(
        name="j1",
        parent_link="base",
        child_link="l1",
        limits=limits,
        **joint_kwargs,
    )
    schema = CommonSchema(
        metadata=Metadata(name="r"),
        links=[Link(name="base", mass=1.0), Link(name="l1", mass=1.0)],
        joints=[joint],
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "scene.xml"
        MJCFExporter().export(schema, path)
        return path.read_text()


def _motor_attr(xml: str, attr: str) -> str | None:
    """Read an attribute off the actuator-section <motor>.

    Parsed rather than regexed: the document also carries a <motor> inside
    <default>, and a regex for the first match reads that one instead — which makes
    an "attribute is absent" assertion pass for the wrong element.
    """
    root = ET.fromstring(xml)
    motors = root.findall("./actuator/motor")
    assert len(motors) == 1, f"expected one actuator motor, got {len(motors)}"
    return motors[0].get(attr)


def test_a_fallback_motor_never_carries_a_positional_ctrlrange():
    """``ctrl`` on a <motor> is a torque. Copying the joint's angular range in means
    a joint limited to [-2.72, -0.84] could only ever be driven one way."""
    xml = _fallback_motor_xml(
        {
            "type": JointType.REVOLUTE,
            "limits": {"lower": -2.7227, "upper": -0.83776, "effort": 45.0},
        }
    )
    assert _motor_attr(xml, "ctrlrange") is None


def test_a_continuous_joint_gets_no_ctrlrange_from_stray_bounds():
    """The joint element says ``limited="false"``; a finite ctrl clamp would
    contradict it."""
    xml = _fallback_motor_xml(
        {
            "type": JointType.CONTINUOUS,
            "limits": {"lower": -3.14, "upper": 3.14, "effort": 10.0},
        }
    )
    assert _motor_attr(xml, "ctrlrange") is None


def test_a_zero_effort_on_a_fallback_motor_means_unspecified():
    """With no declared actuators, URDF ``effort="0"`` is a placeholder, not "inert".

    CAD exporters (SolidWorks among them) write ``effort="0"`` on every joint. Reading
    it as "must not be driven" froze such robots completely, so on the fallback path a
    zero effort is treated like a missing one: no ``forcerange`` and no ``gear`` pin.
    A declared actuator keeps the strict meaning; see
    ``test_a_declared_zero_torque_actuator_stays_strict_on_a_zero_effort_joint``.
    """
    xml = _fallback_motor_xml(
        {
            "type": JointType.REVOLUTE,
            "limits": {"lower": -1.0, "upper": 1.0, "effort": 0.0},
        }
    )

    assert _motor_attr(xml, "gear") is None
    assert _motor_attr(xml, "forcerange") is None


def test_a_zero_effort_fallback_motor_still_drives_the_joint():
    """The same property asserted against MuJoCo: the joint moves when commanded."""
    mujoco = pytest.importorskip("mujoco")

    xml = _fallback_motor_xml(
        {
            "type": JointType.REVOLUTE,
            "limits": {"lower": -1.0, "upper": 1.0, "effort": 0.0},
        }
    )
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    data.ctrl[0] = 1.0
    for _ in range(200):
        mujoco.mj_step(model, data)

    assert abs(float(data.qpos[0])) > 1e-3


def test_a_real_effort_bounds_the_torque_and_is_limited():
    """The counterpart: a non-zero effort must compile to an actually-limited motor."""
    mujoco = pytest.importorskip("mujoco")

    xml = _fallback_motor_xml(
        {
            "type": JointType.REVOLUTE,
            "limits": {"lower": -1.0, "upper": 1.0, "effort": 5.0},
        }
    )
    model = mujoco.MjModel.from_xml_string(xml)

    assert bool(model.actuator_forcelimited[0])
    assert model.actuator_forcerange[0].tolist() == [-5.0, 5.0]


def test_a_real_effort_still_bounds_the_torque():
    xml = _fallback_motor_xml(
        {
            "type": JointType.REVOLUTE,
            "limits": {"lower": -1.0, "upper": 1.0, "effort": 45.0},
        }
    )
    assert _motor_attr(xml, "forcerange") == "-45.0 45.0"


def _declared_actuator_xml(**actuator_kwargs) -> str:
    """Export a one-joint schema that DOES declare an actuator.

    The `_fallback_motor_xml` helper above only reaches the no-declared-actuators
    branch, which is why `_add_actuator` carried the same zero-forcerange inversion
    for as long as it did with two compiler-level tests sitting right beside it.
    """
    import tempfile
    from pathlib import Path

    from cyberwave_robot_format.schema import Actuator, ActuatorType
    from cyberwave_robot_format.mjcf import MJCFExporter

    schema = CommonSchema(
        metadata=Metadata(name="r"),
        links=[Link(name="base", mass=1.0), Link(name="l1", mass=1.0)],
        joints=[
            Joint(
                name="j1",
                type=JointType.REVOLUTE,
                parent_link="base",
                child_link="l1",
                limits=JointLimits(lower=-1.0, upper=1.0, effort=5.0),
            )
        ],
        actuators=[
            Actuator(name="a1", joint="j1", type=ActuatorType.TORQUE, **actuator_kwargs)
        ],
    )
    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory) / "r.xml"
        MJCFExporter().export(schema, out)
        return out.read_text()


def _actuator_attr(xml: str, attr: str) -> str | None:
    root = ET.fromstring(xml)
    motors = root.findall("./actuator/motor")
    assert len(motors) == 1, f"expected one actuator motor, got {len(motors)}"
    return motors[0].get(attr)


def test_a_declared_actuator_with_zero_max_torque_is_inert_not_unlimited():
    """`max_torque: 0.0` emitted `forcerange="-0.0 0.0"` — i.e. *unlimited* torque.

    A zero torque ceiling authored on a declared actuator means "do not drive", and
    the old encoding turned it into the strongest actuator in the scene.
    """
    xml = _declared_actuator_xml(max_torque=0.0)

    assert _actuator_attr(xml, "gear") == "0"
    assert _actuator_attr(xml, "forcerange") is None


def test_a_declared_actuator_with_a_zero_force_range_is_inert_too():
    """`force_range: (0, 0)` is the same statement, and was the same inversion."""
    xml = _declared_actuator_xml(force_range=(0.0, 0.0))

    assert _actuator_attr(xml, "gear") == "0"
    assert _actuator_attr(xml, "forcerange") is None


def test_a_zero_torque_ceiling_survives_a_gear_ratio():
    """`gear_ratio` is written further down and would overwrite the `gear="0"` pin,
    silently making the joint drivable again two statements after it was made inert."""
    xml = _declared_actuator_xml(max_torque=0.0, gear_ratio=50.0)

    assert _actuator_attr(xml, "gear") == "0"


def test_a_real_max_torque_still_becomes_a_symmetric_forcerange():
    """The ordinary case must be untouched."""
    xml = _declared_actuator_xml(max_torque=7.5)

    assert _actuator_attr(xml, "forcerange") == "-7.5 7.5"
    assert _actuator_attr(xml, "gear") is None


def test_a_declared_zero_torque_actuator_does_not_move_when_commanded():
    """The property behind the encoding, asserted against MuJoCo itself."""
    mujoco = pytest.importorskip("mujoco")

    model = mujoco.MjModel.from_xml_string(_declared_actuator_xml(max_torque=0.0))
    data = mujoco.MjData(model)
    data.ctrl[0] = 1000.0
    for _ in range(200):
        mujoco.mj_step(model, data)

    assert abs(float(data.qpos[0])) < 1e-9


def test_a_declared_zero_torque_actuator_stays_strict_on_a_zero_effort_joint():
    """The counterpart of the fallback rule: a *declared* zero ceiling stays inert.

    The joint carries ``effort=0.0`` as well, so this pins that the fallback's
    "zero effort means unspecified" reading does not leak into declared actuators.
    """
    import tempfile
    from pathlib import Path

    from cyberwave_robot_format.mjcf import MJCFExporter
    from cyberwave_robot_format.schema import Actuator, ActuatorType

    schema = CommonSchema(
        metadata=Metadata(name="r"),
        links=[Link(name="base", mass=1.0), Link(name="l1", mass=1.0)],
        joints=[
            Joint(
                name="j1",
                type=JointType.REVOLUTE,
                parent_link="base",
                child_link="l1",
                limits=JointLimits(lower=-1.0, upper=1.0, effort=0.0),
            )
        ],
        actuators=[
            Actuator(name="a1", joint="j1", type=ActuatorType.TORQUE, max_torque=0.0)
        ],
    )
    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory) / "r.xml"
        MJCFExporter().export(schema, out)
        xml = out.read_text()

    assert _actuator_attr(xml, "gear") == "0"
    assert _actuator_attr(xml, "forcerange") is None

    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    data.ctrl[0] = 1000.0
    for _ in range(200):
        mujoco.mj_step(model, data)
    assert abs(float(data.qpos[0])) < 1e-9


def test_a_zero_torque_ceiling_wins_over_a_declared_force_range():
    """The two conditions are not mutually exclusive, and inert must win.

    ``force_range=(-10, 10)`` with ``max_torque=0.0`` sets both ``has_real_force_range``
    and ``must_not_drive``. Testing the forcerange first wrote ``forcerange`` and left
    ``gear`` unset, and the gear-ratio guard then skipped the ratio while logging
    "keeping gear=0" — a pin that was never written. MuJoCo fell back to gear=1, so a
    schema asking for an inert joint produced a drivable one delivering 1/gear_ratio of
    the intended torque.
    """
    xml = _declared_actuator_xml(
        force_range=(-10.0, 10.0), max_torque=0.0, gear_ratio=100.0
    )

    assert _actuator_attr(xml, "gear") == "0"
    assert _actuator_attr(xml, "forcerange") is None


def test_a_gear_ratio_survives_a_real_force_range():
    """The ordinary geared actuator must keep its ratio."""
    xml = _declared_actuator_xml(force_range=(-10.0, 10.0), gear_ratio=100.0)

    assert _actuator_attr(xml, "forcerange") == "-10.0 10.0"
    assert _actuator_attr(xml, "gear") == "100.0"
