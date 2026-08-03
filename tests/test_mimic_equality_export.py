"""MJCF export of mimic joints: <equality> couplings + default slave damping.

A URDF ``<mimic>`` (``Joint.mimic``) must become an MJCF
``<equality><joint joint1=slave joint2=driver polycoef="offset multiplier 0 0 0"/>``
element, otherwise the slave joint is exported free and undamped — the Robotiq
2F-85 linkage joints then ring between their limits forever and shake the arm
(and its wrist camera) on the simulated plant.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET

import pytest

from cyberwave_robot_format.mjcf.exporter import MJCFExporter
from cyberwave_robot_format.schema import (
    Actuator,
    ActuatorType,
    CommonSchema,
    Inertia,
    Joint,
    JointDynamics,
    JointLimits,
    JointType,
    Link,
    Metadata,
    MimicJoint,
    Pose,
    Vector3,
)


def _link(name: str) -> Link:
    return Link(
        name=name,
        mass=0.05,
        inertia=Inertia(ixx=1e-4, iyy=1e-4, izz=1e-4),
    )


def _gripper_schema(*, slave_dynamics: JointDynamics | None = None) -> CommonSchema:
    """base → knuckle (driver) plus two mimic slaves (one counter-rotating)."""
    joints = [
        Joint(
            name="left_knuckle",
            type=JointType.REVOLUTE,
            parent_link="base",
            child_link="left_link",
            pose=Pose(position=Vector3(0.0, 0.0, 0.1)),
            axis=Vector3(0.0, -1.0, 0.0),
            limits=JointLimits(lower=0.0, upper=0.8, effort=5.0),
        ),
        Joint(
            name="right_knuckle",
            type=JointType.REVOLUTE,
            parent_link="base",
            child_link="right_link",
            pose=Pose(position=Vector3(0.0, 0.0, 0.1)),
            axis=Vector3(0.0, -1.0, 0.0),
            limits=JointLimits(lower=-0.8, upper=0.0, effort=5.0),
            mimic=MimicJoint(joint="left_knuckle", multiplier=-1.0, offset=0.0),
            dynamics=slave_dynamics,
        ),
        Joint(
            name="left_tip",
            type=JointType.REVOLUTE,
            parent_link="left_link",
            child_link="left_tip_link",
            pose=Pose(position=Vector3(0.0, 0.0, 0.05)),
            axis=Vector3(0.0, -1.0, 0.0),
            limits=JointLimits(lower=0.0, upper=0.8, effort=5.0),
            mimic=MimicJoint(joint="left_knuckle", multiplier=1.0, offset=0.0),
            dynamics=slave_dynamics,
        ),
    ]
    return CommonSchema(
        metadata=Metadata(name="mimic_gripper"),
        links=[
            _link("base"),
            _link("left_link"),
            _link("right_link"),
            _link("left_tip_link"),
        ],
        joints=joints,
    )


def _export(schema: CommonSchema, tmp_path) -> ET.Element:
    out = tmp_path / "robot.xml"
    MJCFExporter().export(schema, out)
    return ET.parse(out).getroot()


def test_mimic_joints_export_equality_couplings(tmp_path) -> None:
    root = _export(_gripper_schema(), tmp_path)

    couplings = {e.get("joint1"): e for e in root.findall("./equality/joint")}
    assert set(couplings) == {"right_knuckle", "left_tip"}

    right = couplings["right_knuckle"]
    assert right.get("joint2") == "left_knuckle"
    coeffs = [float(v) for v in right.get("polycoef").split()]
    assert coeffs == pytest.approx([0.0, -1.0, 0.0, 0.0, 0.0])

    tip = couplings["left_tip"]
    assert tip.get("joint2") == "left_knuckle"
    coeffs = [float(v) for v in tip.get("polycoef").split()]
    assert coeffs == pytest.approx([0.0, 1.0, 0.0, 0.0, 0.0])


def test_mimic_slaves_get_default_damping(tmp_path) -> None:
    root = _export(_gripper_schema(), tmp_path)

    by_name = {e.get("name"): e for e in root.iter("joint")}
    for slave in ("right_knuckle", "left_tip"):
        elem = by_name[slave]
        assert float(elem.get("damping")) > 0.0, slave
        assert float(elem.get("armature")) > 0.0, slave
    # The driver has no authored dynamics and no mimic — untouched.
    assert by_name["left_knuckle"].get("damping") is None


def test_mimic_slave_authored_dynamics_win_over_default(tmp_path) -> None:
    root = _export(
        _gripper_schema(
            slave_dynamics=JointDynamics(damping=0.5, friction=0.0, armature=0.002)
        ),
        tmp_path,
    )

    by_name = {e.get("name"): e for e in root.iter("joint")}
    assert float(by_name["right_knuckle"].get("damping")) == pytest.approx(0.5)
    assert float(by_name["right_knuckle"].get("armature")) == pytest.approx(0.002)


def test_no_equality_section_without_mimic_joints(tmp_path) -> None:
    schema = _gripper_schema()
    for joint in schema.joints:
        joint.mimic = None
    root = _export(schema, tmp_path)
    assert root.find("equality") is None


def test_mimic_to_missing_driver_is_skipped(tmp_path) -> None:
    schema = _gripper_schema()
    schema.joints[1].mimic = MimicJoint(joint="not_a_joint", multiplier=1.0)
    schema.joints[2].mimic = None
    root = _export(schema, tmp_path)
    assert root.find("equality") is None


def test_mimic_to_fixed_driver_is_skipped(tmp_path) -> None:
    # B1: a mimic whose driver is a FIXED joint has no scalar <joint> element in
    # MJCF. Emitting the <equality> would dangle-reference it and fail to
    # compile the whole model, so the coupling must be dropped.
    schema = _gripper_schema()
    schema.joints[0].type = JointType.FIXED  # left_knuckle is the driver
    root = _export(schema, tmp_path)
    assert root.find("equality") is None
    joint_names = {e.get("name") for e in root.iter("joint")}
    assert "left_knuckle" not in joint_names  # fixed → no scalar joint element


def test_self_referential_mimic_is_skipped(tmp_path) -> None:
    # B6 (exporter defence-in-depth): a joint that mimics itself would emit a
    # degenerate <equality joint1==joint2>; drop it.
    schema = _gripper_schema()
    schema.joints[2].mimic = None
    schema.joints[1].mimic = MimicJoint(joint="right_knuckle", multiplier=1.0)
    root = _export(schema, tmp_path)
    assert root.find("equality") is None


def test_fallback_actuator_skips_mimic_slaves(tmp_path) -> None:
    # B5: with no explicit actuators the exporter emits one fallback <motor> per
    # movable joint — but NOT for mimic slaves, which the <equality> drives.
    root = _export(_gripper_schema(), tmp_path)
    motor_joints = {m.get("joint") for m in root.findall("./actuator/motor")}
    assert "left_knuckle" in motor_joints
    assert "right_knuckle" not in motor_joints
    assert "left_tip" not in motor_joints


def test_continuous_driver_with_none_limits_and_mimic_slave_exports(tmp_path) -> None:
    # #4 regression: a CONTINUOUS driver carries no positional limits
    # (limits=None, e.g. Kinova joints 1/3/5/7 after the honest-unlimited patch)
    # and still drives a mimic slave. Export must not crash on the None limits:
    # the continuous joint stays limited="false", its position actuator gets a
    # finite home±span ctrlrange (never [0, 0]), and the <equality> is emitted.
    joints = [
        Joint(
            name="wrist_roll",
            type=JointType.CONTINUOUS,
            parent_link="base",
            child_link="tool",
            pose=Pose(position=Vector3(0.0, 0.0, 0.1)),
            axis=Vector3(0.0, 0.0, 1.0),
            home_position=0.25,
            limits=JointLimits(lower=None, upper=None, effort=5.0, velocity=1.0),
        ),
        Joint(
            name="passive_follower",
            type=JointType.REVOLUTE,
            parent_link="tool",
            child_link="follower_link",
            pose=Pose(position=Vector3(0.0, 0.0, 0.05)),
            axis=Vector3(0.0, 0.0, 1.0),
            limits=JointLimits(lower=-2.0, upper=2.0, effort=5.0),
            mimic=MimicJoint(joint="wrist_roll", multiplier=1.0, offset=0.0),
        ),
    ]
    schema = CommonSchema(
        metadata=Metadata(name="continuous_mimic"),
        links=[_link("base"), _link("tool"), _link("follower_link")],
        joints=joints,
        actuators=[
            Actuator(name="wrist_act", joint="wrist_roll", type=ActuatorType.POSITION)
        ],
    )

    root = _export(schema, tmp_path)

    driver = next(e for e in root.iter("joint") if e.get("name") == "wrist_roll")
    assert driver.get("limited") == "false"
    assert driver.get("range") is None

    act = next(a for a in root.iter("position") if a.get("joint") == "wrist_roll")
    lo, hi = (float(v) for v in act.get("ctrlrange").split())
    assert lo < hi  # finite, non-degenerate band centered on home

    coupling = root.find("./equality/joint")
    assert coupling is not None
    assert coupling.get("joint1") == "passive_follower"
    assert coupling.get("joint2") == "wrist_roll"


def test_explicit_actuator_on_mimic_slave_is_skipped(tmp_path, caplog) -> None:
    # B5: an explicit actuator targeting a mimic slave fights the <equality>
    # coupling — an authoring error. The exporter drops that ONE actuator and
    # logs an error, rather than raising: export runs per scene, so raising would
    # fail every scene containing the asset. The rest of the model still exports,
    # and the coupling (not an actuator) drives the slave.
    schema = _gripper_schema()
    schema.actuators = [
        Actuator(name="driver_act", joint="left_knuckle", type=ActuatorType.POSITION),
        Actuator(name="slave_act", joint="right_knuckle", type=ActuatorType.POSITION),
    ]
    with caplog.at_level(logging.ERROR):
        root = _export(schema, tmp_path)

    actuator_names = {elem.get("name") for elem in root.findall("./actuator/*")}
    assert "driver_act" in actuator_names
    assert "slave_act" not in actuator_names
    assert "mimic slave" in caplog.text

    # The coupling is still what drives the slave.
    coupling = root.find("./equality/joint")
    assert coupling is not None
    assert coupling.get("joint1") == "right_knuckle"
