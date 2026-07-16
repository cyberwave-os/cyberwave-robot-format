"""MJCF export of mimic joints: <equality> couplings + default slave damping.

A URDF ``<mimic>`` (``Joint.mimic``) must become an MJCF
``<equality><joint joint1=slave joint2=driver polycoef="offset multiplier 0 0 0"/>``
element, otherwise the slave joint is exported free and undamped — the Robotiq
2F-85 linkage joints then ring between their limits forever and shake the arm
(and its wrist camera) on the simulated plant.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from cyberwave_robot_format.mjcf.exporter import MJCFExporter
from cyberwave_robot_format.schema import (
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
