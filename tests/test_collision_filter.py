"""Test collision group functionality."""
import pytest
from cyberwave_robot_format.schema import (
    CommonSchema,
    Metadata,
    Link,
    Collision,
    CollisionGroup,
    CollisionConfig,
    CollisionExclude,
    CollisionPair,
    Geometry,
    GeometryType,
    Vector3,
)


def test_collision_group_defaults():
    """Test that CollisionGroup has correct default values."""
    group = CollisionGroup(name="default")
    assert group.name == "default"
    assert group.contype == 1
    assert group.conaffinity == 1


def test_collision_group_disabled():
    """Test disabled collision group."""
    group = CollisionGroup(name="disabled", contype=0, conaffinity=0)
    assert group.contype == 0
    assert group.conaffinity == 0


def test_collision_with_default_group():
    """Test that Collision objects reference 'default' group by default."""
    collision = Collision(
        geometry=Geometry(type=GeometryType.BOX, size=Vector3(1, 1, 1))
    )
    assert hasattr(collision, 'group')
    assert collision.group == "default"


def test_collision_custom_group():
    """Test Collision with custom group reference."""
    collision = Collision(
        geometry=Geometry(type=GeometryType.SPHERE, radius=0.5),
        group="robots"
    )
    assert collision.group == "robots"


def test_collision_config_empty():
    """Test empty CollisionConfig."""
    config = CollisionConfig()
    assert config.groups == {}
    assert config.excludes == []
    assert config.pairs == []


def test_collision_config_with_groups():
    """Test CollisionConfig with collision groups."""
    config = CollisionConfig(
        groups={
            "default": CollisionGroup(name="default", contype=1, conaffinity=1),
            "robots": CollisionGroup(name="robots", contype=1, conaffinity=2),
            "objects": CollisionGroup(name="objects", contype=2, conaffinity=1),
        }
    )
    assert len(config.groups) == 3
    assert config.groups["default"].contype == 1
    assert config.groups["robots"].conaffinity == 2
    assert config.groups["objects"].contype == 2


def test_collision_exclude():
    """Test CollisionExclude creation."""
    exclude = CollisionExclude(body1="link_a", body2="link_b")
    assert exclude.body1 == "link_a"
    assert exclude.body2 == "link_b"


def test_collision_pair():
    """Test CollisionPair creation with contact overrides."""
    pair = CollisionPair(
        geom1="finger_pad",
        geom2="object",
        friction=[1.5, 0.005, 0.0001],
        condim=4,
        solref=[0.01, 1.0],
        solimp=[0.9, 0.95, 0.001, 0.5, 2]
    )
    assert pair.geom1 == "finger_pad"
    assert pair.geom2 == "object"
    assert pair.friction == [1.5, 0.005, 0.0001]
    assert pair.condim == 4
    assert pair.solref == [0.01, 1.0]
    assert pair.solimp == [0.9, 0.95, 0.001, 0.5, 2]


def test_collision_config_with_rules():
    """Test CollisionConfig with groups, excludes and pairs."""
    config = CollisionConfig(
        groups={
            "default": CollisionGroup(name="default", contype=1, conaffinity=1),
        },
        excludes=[
            CollisionExclude(body1="upper_arm", body2="forearm"),
            CollisionExclude(body1="forearm", body2="hand"),
        ],
        pairs=[
            CollisionPair(
                geom1="gripper_left",
                geom2="object",
                friction=[2.0, 0.01, 0.001]
            )
        ]
    )
    assert len(config.groups) == 1
    assert len(config.excludes) == 2
    assert len(config.pairs) == 1
    assert config.excludes[0].body1 == "upper_arm"
    assert config.pairs[0].geom1 == "gripper_left"


def test_schema_with_collision_config():
    """Test CommonSchema with CollisionConfig."""
    schema = CommonSchema(
        metadata=Metadata(name="test_robot"),
        collision_config=CollisionConfig(
            groups={"default": CollisionGroup(name="default")},
            excludes=[CollisionExclude(body1="link1", body2="link2")]
        )
    )
    assert schema.collision_config is not None
    assert len(schema.collision_config.groups) == 1
    assert len(schema.collision_config.excludes) == 1


def test_collision_patterns_robots_vs_objects():
    """Test robot vs object collision pattern using groups."""
    # Robot group: type=1, collides with type=2
    robot_group = CollisionGroup(name="robots", contype=1, conaffinity=2)
    
    # Object group: type=2, collides with type=1
    object_group = CollisionGroup(name="objects", contype=2, conaffinity=1)
    
    # Verify robot-object collision
    assert (robot_group.contype & object_group.conaffinity) != 0
    assert (object_group.contype & robot_group.conaffinity) != 0
    
    # Verify no robot-robot collision
    assert (robot_group.contype & robot_group.conaffinity) == 0
    
    # Verify no object-object collision
    assert (object_group.contype & object_group.conaffinity) == 0


def test_collision_pattern_all_collides_all():
    """Test default all-collides-all pattern."""
    default_group = CollisionGroup(name="default", contype=1, conaffinity=1)
    
    # Everything collides with everything
    assert (default_group.contype & default_group.conaffinity) != 0


def test_collision_serialization():
    """Test that Collision with group can be serialized/deserialized."""
    collision = Collision(
        name="test_collision",
        geometry=Geometry(type=GeometryType.BOX, size=Vector3(1, 1, 1)),
        group="robots"
    )
    
    # Create a simple schema to test serialization
    schema = CommonSchema(
        metadata=Metadata(name="test"),
        collision_config=CollisionConfig(
            groups={
                "default": CollisionGroup(name="default"),
                "robots": CollisionGroup(name="robots", contype=1, conaffinity=2),
            }
        ),
        links=[Link(name="test_link", collisions=[collision])]
    )
    
    # Serialize to dict
    schema_dict = schema.to_dict()
    
    # Verify group is in the dict
    assert 'links' in schema_dict
    assert len(schema_dict['links']) > 0
    assert 'collisions' in schema_dict['links'][0]
    assert len(schema_dict['links'][0]['collisions']) > 0
    assert 'group' in schema_dict['links'][0]['collisions'][0]
    assert schema_dict['links'][0]['collisions'][0]['group'] == "robots"
    
    # Verify collision_config.groups is in the dict
    assert 'collision_config' in schema_dict
    assert 'groups' in schema_dict['collision_config']
    assert 'robots' in schema_dict['collision_config']['groups']
    
    # Deserialize back
    schema_restored = CommonSchema.from_dict(schema_dict)
    assert schema_restored.links[0].collisions[0].group == "robots"
    assert "robots" in schema_restored.collision_config.groups
    assert schema_restored.collision_config.groups["robots"].contype == 1
    assert schema_restored.collision_config.groups["robots"].conaffinity == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
