#!/usr/bin/env python3
"""
Example demonstrating CommonSchema composition/merge functionality.

This shows how to compose multiple robot schemas into one scene by merging them
with deterministic namespacing and optional spawn poses.
"""

from cyberwave_robot_format.schema import (
    CommonSchema,
    Geometry,
    GeometryType,
    Joint,
    JointType,
    Link,
    Metadata,
    Pose,
    Quaternion,
    Vector3,
    Visual,
)


def create_simple_arm() -> CommonSchema:
    """Create a simple 2-link robot arm."""
    return CommonSchema(
        metadata=Metadata(name="simple_arm", description="A 2-link robot arm"),
        links=[
            Link(
                name="base_link",
                mass=1.0,
                visuals=[
                    Visual(
                        geometry=Geometry(
                            type=GeometryType.BOX,
                            size=Vector3(0.1, 0.1, 0.2),
                        )
                    )
                ],
            ),
            Link(
                name="link1",
                mass=0.5,
                visuals=[
                    Visual(
                        geometry=Geometry(
                            type=GeometryType.CYLINDER,
                            radius=0.05,
                            length=0.3,
                        )
                    )
                ],
            ),
            Link(
                name="link2",
                mass=0.3,
                visuals=[
                    Visual(
                        geometry=Geometry(
                            type=GeometryType.CYLINDER,
                            radius=0.04,
                            length=0.25,
                        )
                    )
                ],
            ),
        ],
        joints=[
            Joint(
                name="joint1",
                type=JointType.REVOLUTE,
                parent_link="base_link",
                child_link="link1",
                pose=Pose(position=Vector3(0, 0, 0.1)),
            ),
            Joint(
                name="joint2",
                type=JointType.REVOLUTE,
                parent_link="link1",
                child_link="link2",
                pose=Pose(position=Vector3(0, 0, 0.3)),
            ),
        ],
    )


def create_wheeled_platform() -> CommonSchema:
    """Create a simple wheeled platform."""
    return CommonSchema(
        metadata=Metadata(name="wheeled_platform", description="A mobile platform"),
        links=[
            Link(
                name="chassis",
                mass=5.0,
                visuals=[
                    Visual(
                        geometry=Geometry(
                            type=GeometryType.BOX,
                            size=Vector3(0.6, 0.4, 0.1),
                        )
                    )
                ],
            ),
            Link(
                name="wheel_left",
                mass=0.5,
                visuals=[
                    Visual(
                        geometry=Geometry(
                            type=GeometryType.CYLINDER,
                            radius=0.1,
                            length=0.05,
                        )
                    )
                ],
            ),
            Link(
                name="wheel_right",
                mass=0.5,
                visuals=[
                    Visual(
                        geometry=Geometry(
                            type=GeometryType.CYLINDER,
                            radius=0.1,
                            length=0.05,
                        )
                    )
                ],
            ),
        ],
        joints=[
            Joint(
                name="wheel_left_joint",
                type=JointType.CONTINUOUS,
                parent_link="chassis",
                child_link="wheel_left",
                pose=Pose(position=Vector3(-0.2, 0.25, 0)),
            ),
            Joint(
                name="wheel_right_joint",
                type=JointType.CONTINUOUS,
                parent_link="chassis",
                child_link="wheel_right",
                pose=Pose(position=Vector3(-0.2, -0.25, 0)),
            ),
        ],
    )


def example_basic_merge():
    """Example 1: Basic merge without spawn."""
    print("=" * 60)
    print("Example 1: Basic merge (no spawn)")
    print("=" * 60)
    
    scene = CommonSchema(metadata=Metadata(name="scene"))
    arm = create_simple_arm()
    
    # Merge arm into scene with prefix "arm1"
    scene.merge_in(arm, "arm1", spawn_pose=Pose())
    
    print(f"Scene now has {len(scene.links)} links:")
    for link in scene.links:
        print(f"  - {link.name}")
    
    print(f"\nScene now has {len(scene.joints)} joints:")
    for joint in scene.joints:
        print(f"  - {joint.name}: {joint.parent_link} -> {joint.child_link}")
    
    print()


def example_merge_with_spawn():
    """Example 2: Merge with spawn pose (creates world joint)."""
    print("=" * 60)
    print("Example 2: Merge with spawn pose")
    print("=" * 60)
    
    scene = CommonSchema(metadata=Metadata(name="scene"))
    arm = create_simple_arm()
    
    # Spawn arm at position (1, 0, 0.5)
    spawn_pose = Pose(
        position=Vector3(1.0, 0.0, 0.5),
        orientation=Quaternion(0, 0, 0, 1),
    )
    
    scene.merge_in(arm, "arm1", spawn_pose=spawn_pose)
    
    print(f"Scene now has {len(scene.links)} links")
    print(f"Scene now has {len(scene.joints)} joints:")
    for joint in scene.joints:
        print(f"  - {joint.name}: {joint.parent_link} -> {joint.child_link}")
        if joint.parent_link == "world":
            pos = joint.pose.position
            print(f"    (spawned at position: x={pos.x}, y={pos.y}, z={pos.z})")
    
    print()


def example_multiple_robots():
    """Example 3: Compose multiple robots into one scene."""
    print("=" * 60)
    print("Example 3: Multiple robots in one scene")
    print("=" * 60)
    
    scene = CommonSchema(metadata=Metadata(name="multi_robot_scene"))
    
    # Add a platform
    platform = create_wheeled_platform()
    scene.merge_in(platform, "platform1", spawn_pose=Pose(position=Vector3(0, 0, 0)))
    
    # Add two arms at different positions
    arm1 = create_simple_arm()
    arm2 = create_simple_arm()
    
    scene.merge_in(arm1, "arm_left", spawn=Pose(position=Vector3(0, 1, 0)))
    scene.merge_in(arm2, "arm_right", spawn=Pose(position=Vector3(0, -1, 0)))
    
    print(f"Scene composition:")
    print(f"  - Total links: {len(scene.links)}")
    print(f"  - Total joints: {len(scene.joints)}")
    
    print(f"\nInstances in scene:")
    instances = set()
    for link in scene.links:
        if "__" in link.name:
            instance = link.name.split("__")[0]
            instances.add(instance)
    
    for instance in sorted(instances):
        links = [l for l in scene.links if l.name.startswith(f"{instance}__")]
        joints = [j for j in scene.joints if j.name.startswith(f"{instance}__")]
        print(f"  - {instance}: {len(links)} links, {len(joints)} joints")
    
    print()


def example_collision_free_naming():
    """Example 4: Merging robots with overlapping names."""
    print("=" * 60)
    print("Example 4: Collision-free naming")
    print("=" * 60)
    
    scene = CommonSchema(metadata=Metadata(name="scene"))
    
    # Create two identical arms (same link/joint names)
    arm1 = create_simple_arm()
    arm2 = create_simple_arm()
    
    # Merge both - names will be automatically prefixed
    scene.merge_in(arm1, "left_arm", spawn_pose=Pose())
    scene.merge_in(arm2, "right_arm", spawn_pose=Pose())
    
    print("Both arms have identical internal names, but after merge:")
    print(f"  - Total links: {len(scene.links)} (no collisions)")
    print(f"  - Total joints: {len(scene.joints)} (no collisions)")
    
    print("\nLink names:")
    for link in scene.links:
        print(f"  - {link.name}")
    
    print()


if __name__ == "__main__":
    example_basic_merge()
    example_merge_with_spawn()
    example_multiple_robots()
    example_collision_free_naming()
    
    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
