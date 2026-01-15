"""Tests for MJCF parser."""

from pathlib import Path

import pytest

from cyberwave_robot_format.mjcf import MJCFParser, MJCFExporter


@pytest.fixture
def simple_mjcf_path():
    """Path to the simple robot MJCF fixture."""
    return Path(__file__).parent / "fixtures" / "simple_robot.xml"


def test_mjcf_parser_basic(simple_mjcf_path):
    """Test that MJCF parser can parse a simple robot."""
    parser = MJCFParser()
    schema = parser.parse(simple_mjcf_path)

    # Check metadata
    assert schema.metadata.name == "simple_robot"
    assert schema.metadata.source_format == "mjcf"

    # Check links (MJCF bodies become links)
    assert len(schema.links) >= 2
    link_names = {link.name for link in schema.links}
    assert "base_link" in link_names
    assert "link1" in link_names

    # Check joints
    assert len(schema.joints) >= 1
    joint = next((j for j in schema.joints if j.name == "joint1"), None)
    assert joint is not None
    assert joint.child_link == "link1"

    # Check actuators
    assert len(schema.actuators) >= 1
    actuator = schema.actuators[0]
    assert actuator.name == "motor1"
    assert actuator.joint == "joint1"

    # Validate schema
    issues = schema.validate()
    assert len(issues) == 0, f"Schema validation failed: {issues}"


def test_mjcf_parser_can_parse(simple_mjcf_path):
    """Test that can_parse correctly identifies MJCF files."""
    parser = MJCFParser()
    assert parser.can_parse(simple_mjcf_path) is True


def test_mjcf_parser_link_properties(simple_mjcf_path):
    """Test that link properties are correctly parsed."""
    parser = MJCFParser()
    schema = parser.parse(simple_mjcf_path)

    base_link = next((l for l in schema.links if l.name == "base_link"), None)
    assert base_link is not None
    assert base_link.mass == pytest.approx(1.0)
    assert base_link.inertia.ixx == pytest.approx(0.1)

    # Check visuals (geoms become visuals)
    assert len(base_link.visuals) > 0


def test_mjcf_parser_joint_properties(simple_mjcf_path):
    """Test that joint properties are correctly parsed."""
    parser = MJCFParser()
    schema = parser.parse(simple_mjcf_path)

    joint = next((j for j in schema.joints if j.name == "joint1"), None)
    assert joint is not None
    assert joint.limits is not None
    assert joint.limits.lower == pytest.approx(-3.14)
    assert joint.limits.upper == pytest.approx(3.14)

    assert joint.dynamics is not None
    assert joint.dynamics.damping == pytest.approx(0.1)
    assert joint.dynamics.friction == pytest.approx(0.05)


def test_mjcf_roundtrip(simple_mjcf_path, tmp_path):
    """Test roundtrip: MJCF → CommonSchema → MJCF."""
    parser = MJCFParser()
    schema = parser.parse(simple_mjcf_path)

    # Export to MJCF
    output_path = tmp_path / "roundtrip.xml"
    exporter = MJCFExporter()
    exporter.export(schema, output_path)

    # Verify file was created
    assert output_path.exists()

    # Parse the exported file
    schema2 = parser.parse(output_path)

    # Basic checks
    assert schema2.metadata.name == schema.metadata.name
    assert len(schema2.links) == len(schema.links)
    assert len(schema2.joints) == len(schema.joints)


def test_mjcf_exporter_basic(simple_mjcf_path, tmp_path):
    """Test that MJCF exporter can export a schema."""
    parser = MJCFParser()
    schema = parser.parse(simple_mjcf_path)

    output_path = tmp_path / "exported.xml"
    exporter = MJCFExporter()
    exporter.export(schema, output_path)

    assert output_path.exists()
    
    # Verify it's valid XML by parsing it
    import xml.etree.ElementTree as ET
    tree = ET.parse(output_path)
    root = tree.getroot()
    assert root.tag == "mujoco"
    assert root.get("model") == "simple_robot"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
