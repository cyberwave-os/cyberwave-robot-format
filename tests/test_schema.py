"""Tests for the CommonSchema validation and serialization."""

import pytest

from cyberwave_robot_format.schema import CommonSchema, Joint, JointType, Link, Metadata, Vector3


def test_schema_validation_valid():
    """Test that a valid schema passes validation."""
    metadata = Metadata(name="test_robot")
    link1 = Link(name="base_link", mass=1.0)
    link2 = Link(name="link1", mass=0.5)
    joint = Joint(
        name="joint1",
        type=JointType.REVOLUTE,
        parent_link="base_link",
        child_link="link1",
    )

    schema = CommonSchema(
        metadata=metadata,
        links=[link1, link2],
        joints=[joint],
    )

    issues = schema.validate()
    assert len(issues) == 0, f"Expected no validation issues, got: {issues}"


def test_schema_validation_missing_parent_link():
    """Test that validation catches missing parent link."""
    metadata = Metadata(name="test_robot")
    link2 = Link(name="link1", mass=0.5)
    joint = Joint(
        name="joint1",
        type=JointType.REVOLUTE,
        parent_link="nonexistent_link",
        child_link="link1",
    )

    schema = CommonSchema(
        metadata=metadata,
        links=[link2],
        joints=[joint],
    )

    issues = schema.validate()
    assert len(issues) > 0, "Expected validation issues for missing parent link"
    assert any("nonexistent_link" in issue for issue in issues)


def test_schema_validation_duplicate_links():
    """Test that validation catches duplicate link names."""
    metadata = Metadata(name="test_robot")
    link1 = Link(name="base_link", mass=1.0)
    link2 = Link(name="base_link", mass=0.5)  # Duplicate name

    schema = CommonSchema(
        metadata=metadata,
        links=[link1, link2],
    )

    issues = schema.validate()
    assert len(issues) > 0, "Expected validation issues for duplicate link names"
    assert any("Duplicate link names" in issue for issue in issues)


def test_schema_get_root_links():
    """Test that get_root_links correctly identifies root links."""
    metadata = Metadata(name="test_robot")
    link1 = Link(name="base_link", mass=1.0)
    link2 = Link(name="link1", mass=0.5)
    link3 = Link(name="link2", mass=0.3)
    joint1 = Joint(
        name="joint1",
        type=JointType.REVOLUTE,
        parent_link="base_link",
        child_link="link1",
    )
    joint2 = Joint(
        name="joint2",
        type=JointType.REVOLUTE,
        parent_link="link1",
        child_link="link2",
    )

    schema = CommonSchema(
        metadata=metadata,
        links=[link1, link2, link3],
        joints=[joint1, joint2],
    )

    root_links = schema.get_root_links()
    assert len(root_links) == 1
    assert root_links[0].name == "base_link"


def test_schema_get_single_root_link_success():
    """get_single_root_link returns the single canonical root."""
    metadata = Metadata(name="test_robot")
    link1 = Link(name="base_link", mass=1.0)
    link2 = Link(name="link1", mass=0.5)
    joint = Joint(
        name="joint1",
        type=JointType.REVOLUTE,
        parent_link="base_link",
        child_link="link1",
    )

    schema = CommonSchema(
        metadata=metadata,
        links=[link1, link2],
        joints=[joint],
    )

    root = schema.get_single_root_link()
    assert root.name == "base_link"


def test_schema_get_single_root_link_errors():
    """get_single_root_link fails for 0 or multiple roots."""
    # No roots (kinematic loop)
    metadata = Metadata(name="loop_robot")
    link1 = Link(name="l1")
    link2 = Link(name="l2")
    joint1 = Joint(
        name="j1",
        type=JointType.REVOLUTE,
        parent_link="l1",
        child_link="l2",
    )
    joint2 = Joint(
        name="j2",
        type=JointType.REVOLUTE,
        parent_link="l2",
        child_link="l1",
    )
    loop_schema = CommonSchema(
        metadata=metadata,
        links=[link1, link2],
        joints=[joint1, joint2],
    )

    with pytest.raises(ValueError, match="no root links"):
        loop_schema.get_single_root_link()

    # Multiple roots
    metadata_multi = Metadata(name="multi_root_robot")
    root1 = Link(name="root1")
    root2 = Link(name="root2")
    child = Link(name="child")
    joint = Joint(
        name="j1",
        type=JointType.REVOLUTE,
        parent_link="root1",
        child_link="child",
    )
    multi_schema = CommonSchema(
        metadata=metadata_multi,
        links=[root1, root2, child],
        joints=[joint],
    )

    with pytest.raises(ValueError, match="multiple root links"):
        multi_schema.get_single_root_link()


def test_schema_to_dict():
    """Test that to_dict produces a serializable dictionary."""
    metadata = Metadata(name="test_robot", version="1.0")
    link1 = Link(name="base_link", mass=1.0)
    joint = Joint(
        name="joint1",
        type=JointType.REVOLUTE,
        parent_link="world",
        child_link="base_link",
    )

    schema = CommonSchema(
        metadata=metadata,
        links=[link1],
        joints=[joint],
    )

    schema_dict = schema.to_dict()

    # Check structure
    assert "metadata" in schema_dict
    assert "links" in schema_dict
    assert "joints" in schema_dict
    assert schema_dict["metadata"]["name"] == "test_robot"
    assert len(schema_dict["links"]) == 1
    assert len(schema_dict["joints"]) == 1

    # Check that enums are converted to values
    assert schema_dict["joints"][0]["type"] == "revolute"


def test_vector3_operations():
    """Test Vector3 utility methods."""
    v = Vector3(x=1.0, y=2.0, z=3.0)

    # Test to_list
    assert v.to_list() == [1.0, 2.0, 3.0]

    # Test from_list
    v2 = Vector3.from_list([4.0, 5.0, 6.0])
    assert v2.x == 4.0
    assert v2.y == 5.0
    assert v2.z == 6.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
