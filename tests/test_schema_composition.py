"""Tests for CommonSchema composition/merge functionality."""

import pytest

from cyberwave_robot_format.schema import (
    Actuator,
    ActuatorType,
    CommonSchema,
    Contact,
    ContactSurface,
    Geometry,
    GeometryType,
    Joint,
    JointType,
    Link,
    Metadata,
    Pose,
    Quaternion,
    Sensor,
    Vector3,
    Visual,
)


def create_simple_robot(name: str, link_names: list[str]) -> CommonSchema:
    """Create a simple robot schema for testing."""
    links = [Link(name=link_name) for link_name in link_names]
    
    # Create a chain: link_names[0] is root, each subsequent link connects to previous
    joints = []
    for i in range(1, len(link_names)):
        joints.append(
            Joint(
                name=f"joint_{i}",
                type=JointType.REVOLUTE,
                parent_link=link_names[i - 1],
                child_link=link_names[i],
            )
        )
    
    # Add an actuator for the first joint if there are joints
    actuators = []
    if joints:
        actuators.append(
            Actuator(
                name="actuator_1",
                joint="joint_1",
                type=ActuatorType.POSITION,
            )
        )
    
    # Add a sensor on the first link
    sensors = []
    if links:
        sensors.append(
            Sensor(
                name="sensor_1",
                type="imu",
                parent_link=link_names[0],
            )
        )
    
    # Add a contact on the last link
    contacts = []
    if links:
        contacts.append(
            Contact(
                name="contact_1",
                link=link_names[-1],
            )
        )
    
    return CommonSchema(
        metadata=Metadata(name=name),
        links=links,
        joints=joints,
        actuators=actuators,
        sensors=sensors,
        contacts=contacts,
    )


def test_merge_basic():
    """Test basic merge without spawn."""
    base = create_simple_robot("base", ["base_link"])
    arm = create_simple_robot("arm", ["arm_base", "arm_link1", "arm_link2"])
    
    base.merge_in(arm, "arm1", spawn_pose=Pose())
    
    # Check that arm links are prefixed
    assert base.get_link("arm1__arm_base") is not None
    assert base.get_link("arm1__arm_link1") is not None
    assert base.get_link("arm1__arm_link2") is not None
    
    # Check that arm joints are prefixed and references updated
    assert base.get_joint("arm1__joint_1") is not None
    joint1 = base.get_joint("arm1__joint_1")
    assert joint1.parent_link == "arm1__arm_base"
    assert joint1.child_link == "arm1__arm_link1"
    
    assert base.get_joint("arm1__joint_2") is not None
    joint2 = base.get_joint("arm1__joint_2")
    assert joint2.parent_link == "arm1__arm_link1"
    assert joint2.child_link == "arm1__arm_link2"
    
    # Check that actuators are prefixed and references updated
    assert base.get_actuator("arm1__actuator_1") is not None
    actuator = base.get_actuator("arm1__actuator_1")
    assert actuator.joint == "arm1__joint_1"
    
    # Check that sensors are prefixed and references updated
    sensor = next((s for s in base.sensors if s.name == "arm1__sensor_1"), None)
    assert sensor is not None
    assert sensor.parent_link == "arm1__arm_base"
    
    # Check that contacts are prefixed and references updated
    contact = next((c for c in base.contacts if c.name == "arm1__contact_1"), None)
    assert contact is not None
    assert contact.link == "arm1__arm_link2"
    
    # Original base link should still exist
    assert base.get_link("base_link") is not None


def test_merge_with_spawn():
    """Test merge with spawn pose (creates world->root fixed joint)."""
    base = create_simple_robot("base", ["base_link"])
    arm = create_simple_robot("arm", ["arm_base", "arm_link1"])
    
    spawn_pose = Pose(
        position=Vector3(1.0, 2.0, 3.0),
        orientation=Quaternion(0.0, 0.0, 0.707, 0.707),
    )
    
    base.merge_in(arm, "arm1", spawn_pose=spawn_pose)
    
    # Check that spawn joint was created
    spawn_joint = base.get_joint("arm1__spawn")
    assert spawn_joint is not None
    assert spawn_joint.type == JointType.FIXED
    assert spawn_joint.parent_link == "world"
    assert spawn_joint.child_link == "arm1__arm_base"
    assert spawn_joint.pose.position.x == 1.0
    assert spawn_joint.pose.position.y == 2.0
    assert spawn_joint.pose.position.z == 3.0


def test_merge_collision_free_naming():
    """Test that merging schemas with overlapping names doesn't cause collisions."""
    robot1 = create_simple_robot("robot1", ["base", "link1", "link2"])
    robot2 = create_simple_robot("robot2", ["base", "link1", "link2"])
    
    robot1.merge_in(robot2, "r2", spawn_pose=Pose())
    
    # Both sets of links should exist with different names
    assert robot1.get_link("base") is not None
    assert robot1.get_link("link1") is not None
    assert robot1.get_link("link2") is not None
    
    assert robot1.get_link("r2__base") is not None
    assert robot1.get_link("r2__link1") is not None
    assert robot1.get_link("r2__link2") is not None
    
    # Check joints
    assert robot1.get_joint("joint_1") is not None
    assert robot1.get_joint("joint_2") is not None
    assert robot1.get_joint("r2__joint_1") is not None
    assert robot1.get_joint("r2__joint_2") is not None


def test_merge_multiple_times():
    """Test merging multiple schemas into one."""
    scene = CommonSchema(metadata=Metadata(name="scene"))
    
    arm1 = create_simple_robot("arm", ["base", "link1"])
    arm2 = create_simple_robot("arm", ["base", "link1"])
    arm3 = create_simple_robot("arm", ["base", "link1"])
    
    scene.merge_in(arm1, "arm1", spawn_pose=Pose(position=Vector3(0, 0, 0)))
    scene.merge_in(arm2, "arm2", spawn_pose=Pose(position=Vector3(1, 0, 0)))
    scene.merge_in(arm3, "arm3", spawn_pose=Pose(position=Vector3(2, 0, 0)))
    
    # All three arms should be present
    assert scene.get_link("arm1__base") is not None
    assert scene.get_link("arm2__base") is not None
    assert scene.get_link("arm3__base") is not None
    
    # All spawn joints should exist
    assert scene.get_joint("arm1__spawn") is not None
    assert scene.get_joint("arm2__spawn") is not None
    assert scene.get_joint("arm3__spawn") is not None


def test_merge_invalid_instance_name():
    """Test that invalid instance names are rejected."""
    base = create_simple_robot("base", ["base_link"])
    arm = create_simple_robot("arm", ["arm_base"])
    
    # Empty name
    with pytest.raises(ValueError, match="instance_name cannot be empty"):
        base.merge_in(arm, "", spawn_pose=Pose())
    
    # Invalid characters
    with pytest.raises(ValueError, match="instance_name must match"):
        base.merge_in(arm, "arm-1", spawn_pose=Pose())
    
    with pytest.raises(ValueError, match="instance_name must match"):
        base.merge_in(arm, "arm 1", spawn_pose=Pose())
    
    with pytest.raises(ValueError, match="instance_name must match"):
        base.merge_in(arm, "arm.1", spawn_pose=Pose())


def test_merge_single_root_required():
    """Test that schemas with 0 or multiple roots are rejected."""
    base = create_simple_robot("base", ["base_link"])
    
    # Schema with no root (kinematic loop)
    loop_schema = CommonSchema(
        metadata=Metadata(name="loop"),
        links=[Link(name="link1"), Link(name="link2")],
        joints=[
            Joint(name="j1", type=JointType.REVOLUTE, parent_link="link1", child_link="link2"),
            Joint(name="j2", type=JointType.REVOLUTE, parent_link="link2", child_link="link1"),
        ],
    )
    
    with pytest.raises(ValueError, match="has no root links"):
        base.merge_in(loop_schema, "loop", spawn_pose=Pose())
    
    # Schema with multiple roots
    multi_root = CommonSchema(
        metadata=Metadata(name="multi"),
        links=[Link(name="root1"), Link(name="root2"), Link(name="child")],
        joints=[
            Joint(name="j1", type=JointType.REVOLUTE, parent_link="root1", child_link="child"),
        ],
    )
    
    with pytest.raises(ValueError, match="has multiple root links"):
        base.merge_in(multi_root, "multi", spawn_pose=Pose())


def test_merge_preserves_original():
    """Test that merging doesn't mutate the original 'other' schema."""
    base = create_simple_robot("base", ["base_link"])
    arm = create_simple_robot("arm", ["arm_base", "arm_link1"])
    
    # Store original names
    original_link_names = [link.name for link in arm.links]
    original_joint_names = [joint.name for joint in arm.joints]
    
    base.merge_in(arm, "arm1", spawn_pose=Pose())
    
    # Original arm schema should be unchanged
    current_link_names = [link.name for link in arm.links]
    current_joint_names = [joint.name for joint in arm.joints]
    
    assert original_link_names == current_link_names
    assert original_joint_names == current_joint_names


def test_merge_with_world_parent_preserved():
    """Test that existing world parent links in joints are preserved."""
    base = CommonSchema(
        metadata=Metadata(name="base"),
        links=[Link(name="floating_base")],
        joints=[
            Joint(
                name="float_joint",
                type=JointType.FLOATING,
                parent_link="world",
                child_link="floating_base",
            )
        ],
    )
    
    arm = create_simple_robot("arm", ["arm_base"])
    
    base.merge_in(arm, "arm1", spawn_pose=Pose())
    
    # Original world joint should still have world as parent
    float_joint = base.get_joint("float_joint")
    assert float_joint is not None
    assert float_joint.parent_link == "world"
    
    # Merged arm's root should not have its joint's parent changed to world
    # (unless spawn was used)
    arm_joints = [j for j in base.joints if j.name.startswith("arm1__") and j.name != "arm1__spawn"]
    for joint in arm_joints:
        # These should all have prefixed parents (not world)
        if joint.parent_link != "world":
            assert joint.parent_link.startswith("arm1__")


def test_get_root_links_with_world_joints():
    """Test that get_root_links correctly handles world parent joints."""
    schema = CommonSchema(
        metadata=Metadata(name="test"),
        links=[
            Link(name="base"),
            Link(name="child1"),
            Link(name="child2"),
        ],
        joints=[
            Joint(name="j1", type=JointType.FIXED, parent_link="world", child_link="base"),
            Joint(name="j2", type=JointType.REVOLUTE, parent_link="base", child_link="child1"),
            Joint(name="j3", type=JointType.REVOLUTE, parent_link="base", child_link="child2"),
        ],
    )
    
    roots = schema.get_root_links()
    
    # base should be considered a root even though it has a parent joint to world
    assert len(roots) == 1
    assert roots[0].name == "base"
