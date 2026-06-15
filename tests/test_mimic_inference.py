"""Tests for URDF mimic joint inference and patching."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from cyberwave_robot_format.urdf import (
    URDFParser,
    infer_and_patch_if_needed,
    infer_mimic_joints,
    write_mimic_patched_urdf,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_infer_piper_prismatic_gripper_pair():
    """PiPER joint7/joint8 should be inferred as mimic with high confidence."""
    urdf_path = FIXTURES / "piper_description.urdf"
    result = infer_mimic_joints(urdf_path)

    assert result.should_patch is True
    assert len(result.inferred_mimics) == 1

    mimic = result.inferred_mimics[0]
    assert mimic.driver_joint == "joint7"
    assert mimic.slave_joint == "joint8"
    assert mimic.multiplier == pytest.approx(-1.0)
    assert mimic.offset == pytest.approx(0.0)
    assert mimic.confidence >= 0.85
    assert "opposing_parallel_axes" in mimic.evidence
    assert "complementary_limits" in mimic.evidence
    assert "terminal_fork" in mimic.evidence


def test_infer_and_patch_piper_writes_new_file(tmp_path):
    """Pipeline must write -mimic-joint.urdf and leave the original untouched."""
    urdf_path = FIXTURES / "piper_description.urdf"
    original_bytes = urdf_path.read_bytes()

    work_dir = tmp_path / "piper"
    work_dir.mkdir()
    local_urdf = work_dir / "piper_description.urdf"
    local_urdf.write_bytes(original_bytes)

    result = infer_and_patch_if_needed(local_urdf)
    assert result.should_patch is True
    assert result.output_path is not None
    assert result.output_path.name == "piper_description-mimic-joint.urdf"
    assert result.output_path.exists()

    # Original unchanged
    assert local_urdf.read_bytes() == original_bytes

    tree = ET.parse(result.output_path)
    joint8 = next(j for j in tree.getroot().iter("joint") if j.get("name") == "joint8")
    mimic = joint8.find("mimic")
    assert mimic is not None
    assert mimic.get("joint") == "joint7"
    assert float(mimic.get("multiplier", "0")) == pytest.approx(-1.0)
    assert float(mimic.get("offset", "0")) == pytest.approx(0.0)

    schema = URDFParser().parse(result.output_path)
    slave = next(j for j in schema.joints if j.name == "joint8")
    assert slave.mimic is not None
    assert slave.mimic.joint == "joint7"


def test_explicit_mimic_urdf_is_not_patched(tmp_path):
    """URDF that already has <mimic> must not produce a patched copy."""
    urdf = """<?xml version="1.0"?>
<robot name="gripper_test">
  <link name="base"/><link name="finger_l"/><link name="finger_r"/>
  <joint name="finger_joint1" type="revolute">
    <parent link="base"/><child link="finger_l"/>
    <axis xyz="0 0 1"/>
    <limit lower="-0.8" upper="0.8" effort="10" velocity="1"/>
  </joint>
  <joint name="finger_joint2" type="revolute">
    <parent link="base"/><child link="finger_r"/>
    <mimic joint="finger_joint1" multiplier="-1" offset="0"/>
  </joint>
</robot>"""
    urdf_path = tmp_path / "explicit.urdf"
    urdf_path.write_text(urdf, encoding="utf-8")

    result = infer_and_patch_if_needed(urdf_path)
    assert result.should_patch is False
    assert len(result.existing_mimics) == 1
    assert not (tmp_path / "explicit-mimic-joint.urdf").exists()


def test_unrelated_prismatic_pair_not_inferred():
    """Orthogonal large-travel prismatic joints on the base should not mimic."""
    result = infer_mimic_joints(FIXTURES / "arm_two_prismatic.urdf")
    assert result.should_patch is False
    assert result.inferred_mimics == []


def test_minimal_gripper_fixture_infers_mimic():
    """Minimal PiPER-like pair should infer mimic."""
    result = infer_mimic_joints(FIXTURES / "gripper_prismatic_pair.urdf")
    assert result.should_patch is True
    assert result.inferred_mimics[0].driver_joint == "joint7"
    assert result.inferred_mimics[0].slave_joint == "joint8"


def test_write_mimic_patched_urdf_requires_candidates(tmp_path):
    urdf_path = tmp_path / "empty.urdf"
    urdf_path.write_text('<?xml version="1.0"?><robot name="r"/>', encoding="utf-8")
    with pytest.raises(ValueError, match="No inferred mimic"):
        write_mimic_patched_urdf(urdf_path, [])
