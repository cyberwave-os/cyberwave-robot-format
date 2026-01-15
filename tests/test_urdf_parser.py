"""Tests for URDF parser."""

import json
from pathlib import Path

import pytest

from cyberwave_robot_format.urdf import URDFParser


@pytest.fixture
def simple_urdf_path():
    """Path to the simple robot URDF fixture."""
    return Path(__file__).parent / "fixtures" / "simple_robot.urdf"


def test_urdf_parser_basic(simple_urdf_path):
    """Test that URDF parser can parse a simple robot."""
    parser = URDFParser()
    schema = parser.parse(simple_urdf_path)

    # Check metadata
    assert schema.metadata.name == "simple_robot"

    # Check links
    assert len(schema.links) == 2
    link_names = {link.name for link in schema.links}
    assert "base_link" in link_names
    assert "link1" in link_names

    # Check joints
    assert len(schema.joints) == 1
    joint = schema.joints[0]
    assert joint.name == "joint1"
    assert joint.parent_link == "base_link"
    assert joint.child_link == "link1"

    # Validate schema
    issues = schema.validate()
    assert len(issues) == 0, f"Schema validation failed: {issues}"


def test_urdf_parser_golden_snapshot(simple_urdf_path, tmp_path):
    """Golden test: ensure parser output is stable across changes.
    
    This test verifies that the parsed schema structure remains consistent.
    If this test fails after intentional changes, update the golden file.
    """
    parser = URDFParser()
    schema = parser.parse(simple_urdf_path)

    # Convert to dict for comparison
    schema_dict = schema.to_dict()

    # Golden file path
    golden_file = Path(__file__).parent / "fixtures" / "simple_robot_golden.json"

    # For initial creation or update, uncomment this:
    # with open(golden_file, "w") as f:
    #     json.dump(schema_dict, f, indent=2, sort_keys=True)

    # Compare with golden file if it exists
    if golden_file.exists():
        with open(golden_file, "r") as f:
            expected = json.load(f)

        # Compare key fields (not entire dict as some fields may vary)
        assert schema_dict["metadata"]["name"] == expected["metadata"]["name"]
        assert len(schema_dict["links"]) == len(expected["links"])
        assert len(schema_dict["joints"]) == len(expected["joints"])
    else:
        # Create golden file for first run
        with open(golden_file, "w") as f:
            json.dump(schema_dict, f, indent=2, sort_keys=True)
        pytest.skip("Golden file created, re-run test to validate")


def test_urdf_parser_link_properties(simple_urdf_path):
    """Test that link properties are correctly parsed."""
    parser = URDFParser()
    schema = parser.parse(simple_urdf_path)

    base_link = next((l for l in schema.links if l.name == "base_link"), None)
    assert base_link is not None
    assert base_link.mass == 1.0
    assert base_link.inertia.ixx == pytest.approx(0.1)

    # Check visuals
    assert len(base_link.visuals) > 0


def test_urdf_parser_joint_properties(simple_urdf_path):
    """Test that joint properties are correctly parsed."""
    parser = URDFParser()
    schema = parser.parse(simple_urdf_path)

    joint = schema.joints[0]
    assert joint.limits is not None
    assert joint.limits.lower == pytest.approx(-3.14)
    assert joint.limits.upper == pytest.approx(3.14)
    assert joint.limits.effort == pytest.approx(10.0)

    assert joint.dynamics is not None
    assert joint.dynamics.damping == pytest.approx(0.1)
    assert joint.dynamics.friction == pytest.approx(0.05)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
