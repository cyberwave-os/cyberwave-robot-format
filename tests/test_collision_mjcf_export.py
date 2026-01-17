"""Test MuJoCo export with collision groups."""
import pytest
import xml.etree.ElementTree as ET
from pathlib import Path
import tempfile

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
    Visual,
    Material,
    Inertia,
)
from cyberwave_robot_format.mjcf import MJCFExporter


def test_mjcf_export_collision_group():
    """Test that collision groups are exported to MuJoCo contype/conaffinity."""
    # Create a simple schema with collision group
    schema = CommonSchema(
        metadata=Metadata(name="test_collision"),
        collision_config=CollisionConfig(
            groups={
                "custom": CollisionGroup(name="custom", contype=2, conaffinity=1)
            }
        ),
        links=[
            Link(
                name="link1",
                mass=1.0,
                inertia=Inertia(ixx=0.1, iyy=0.1, izz=0.1),
                collisions=[
                    Collision(
                        name="col1",
                        geometry=Geometry(type=GeometryType.BOX, size=Vector3(1, 1, 1)),
                        group="custom"
                    )
                ]
            )
        ]
    )
    
    # Export to MJCF
    exporter = MJCFExporter()
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test.xml"
        exporter.export(schema, output_path)
        
        # Parse the exported XML
        tree = ET.parse(output_path)
        root = tree.getroot()
        
        # Find collision geoms (class="collision")
        geoms = root.findall(".//geom[@class='collision']")
        assert len(geoms) >= 1
        geom = geoms[0]
        
        # Verify contype and conaffinity are set from the group
        assert geom.get("contype") == "2"
        assert geom.get("conaffinity") == "1"


def test_mjcf_export_default_group():
    """Test that default group (all-collides-all) is exported correctly."""
    schema = CommonSchema(
        metadata=Metadata(name="test_default"),
        collision_config=CollisionConfig(
            groups={
                "default": CollisionGroup(name="default", contype=1, conaffinity=1)
            }
        ),
        links=[
            Link(
                name="link1",
                mass=1.0,
                inertia=Inertia(ixx=0.1, iyy=0.1, izz=0.1),
                collisions=[
                    Collision(
                        name="col1",
                        geometry=Geometry(type=GeometryType.BOX, size=Vector3(1, 1, 1)),
                        group="default"
                    )
                ]
            )
        ]
    )
    
    exporter = MJCFExporter()
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test.xml"
        exporter.export(schema, output_path)
        
        tree = ET.parse(output_path)
        root = tree.getroot()
        
        # Find collision geoms (class="collision")
        geoms = root.findall(".//geom[@class='collision']")
        assert len(geoms) >= 1
        geom = geoms[0]
        
        assert geom.get("contype") == "1"
        assert geom.get("conaffinity") == "1"


def test_mjcf_export_disabled_collision_group():
    """Test that disabled collision groups export with contype=0, conaffinity=0."""
    schema = CommonSchema(
        metadata=Metadata(name="test_disabled"),
        collision_config=CollisionConfig(
            groups={
                "disabled": CollisionGroup(name="disabled", contype=0, conaffinity=0)
            }
        ),
        links=[
            Link(
                name="link1",
                mass=1.0,
                inertia=Inertia(ixx=0.1, iyy=0.1, izz=0.1),
                collisions=[
                    Collision(
                        name="disabled_col",
                        geometry=Geometry(type=GeometryType.SPHERE, radius=0.5),
                        group="disabled"
                    )
                ]
            )
        ]
    )
    
    exporter = MJCFExporter()
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test.xml"
        exporter.export(schema, output_path)
        
        tree = ET.parse(output_path)
        root = tree.getroot()
        
        # Find collision geoms (class="collision")
        geoms = root.findall(".//geom[@class='collision']")
        assert len(geoms) >= 1
        geom = geoms[0]
        
        assert geom.get("contype") == "0"
        assert geom.get("conaffinity") == "0"


def test_mjcf_export_visual_no_collision():
    """Test that visual geoms have contype=0, conaffinity=0."""
    schema = CommonSchema(
        metadata=Metadata(name="test_visual"),
        collision_config=CollisionConfig(
            groups={"default": CollisionGroup(name="default")}
        ),
        links=[
            Link(
                name="link1",
                mass=1.0,
                inertia=Inertia(ixx=0.1, iyy=0.1, izz=0.1),
                visuals=[
                    Visual(
                        name="vis1",
                        geometry=Geometry(type=GeometryType.BOX, size=Vector3(1, 1, 1)),
                        material=Material(color=[1.0, 0.0, 0.0, 1.0])
                    )
                ]
            )
        ]
    )
    
    exporter = MJCFExporter()
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test.xml"
        exporter.export(schema, output_path)
        
        tree = ET.parse(output_path)
        root = tree.getroot()
        
        # Visual geoms should have class="visual" and no collision
        geoms = root.findall(".//geom[@class='visual']")
        assert len(geoms) > 0
        for geom in geoms:
            assert geom.get("contype") == "0"
            assert geom.get("conaffinity") == "0"


def test_mjcf_export_collision_excludes():
    """Test that collision excludes are exported to <contact><exclude>."""
    schema = CommonSchema(
        metadata=Metadata(name="test_excludes"),
        collision_config=CollisionConfig(
            groups={"default": CollisionGroup(name="default")},
            excludes=[
                CollisionExclude(body1="link1", body2="link2"),
                CollisionExclude(body1="link2", body2="link3"),
            ]
        ),
        links=[
            Link(name="link1", mass=1.0, inertia=Inertia(ixx=0.1, iyy=0.1, izz=0.1)),
            Link(name="link2", mass=1.0, inertia=Inertia(ixx=0.1, iyy=0.1, izz=0.1)),
            Link(name="link3", mass=1.0, inertia=Inertia(ixx=0.1, iyy=0.1, izz=0.1)),
        ]
    )
    
    exporter = MJCFExporter()
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test.xml"
        exporter.export(schema, output_path)
        
        tree = ET.parse(output_path)
        root = tree.getroot()
        
        # Find contact section
        contact = root.find("contact")
        assert contact is not None
        
        # Find exclude elements
        excludes = contact.findall("exclude")
        assert len(excludes) == 2
        
        # Verify exclude pairs
        exclude_pairs = [(e.get("body1"), e.get("body2")) for e in excludes]
        assert ("link1", "link2") in exclude_pairs
        assert ("link2", "link3") in exclude_pairs


def test_mjcf_export_collision_pairs():
    """Test that collision pairs are exported to <contact><pair>."""
    schema = CommonSchema(
        metadata=Metadata(name="test_pairs"),
        collision_config=CollisionConfig(
            groups={"default": CollisionGroup(name="default")},
            pairs=[
                CollisionPair(
                    geom1="finger_pad",
                    geom2="object",
                    friction=[1.5, 0.005, 0.0001],
                    condim=4,
                    solref=[0.01, 1.0],
                    margin=0.001
                )
            ]
        ),
        links=[
            Link(
                name="link1",
                mass=1.0,
                inertia=Inertia(ixx=0.1, iyy=0.1, izz=0.1),
                collisions=[
                    Collision(
                        name="finger_pad",
                        geometry=Geometry(type=GeometryType.BOX, size=Vector3(0.1, 0.1, 0.1)),
                        group="default"
                    )
                ]
            ),
            Link(
                name="link2",
                mass=0.5,
                inertia=Inertia(ixx=0.05, iyy=0.05, izz=0.05),
                collisions=[
                    Collision(
                        name="object",
                        geometry=Geometry(type=GeometryType.SPHERE, radius=0.2),
                        group="default"
                    )
                ]
            )
        ]
    )
    
    exporter = MJCFExporter()
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test.xml"
        exporter.export(schema, output_path)
        
        tree = ET.parse(output_path)
        root = tree.getroot()
        
        # Find contact section
        contact = root.find("contact")
        assert contact is not None
        
        # Find pair elements
        pairs = contact.findall("pair")
        assert len(pairs) == 1
        
        pair = pairs[0]
        assert pair.get("geom1") == "finger_pad"
        assert pair.get("geom2") == "object"
        assert pair.get("friction") == "1.5 0.005 0.0001"
        assert pair.get("condim") == "4"
        assert pair.get("solref") == "0.01 1.0"
        assert pair.get("margin") == "0.001"


def test_mjcf_export_no_contact_section_when_empty():
    """Test that no <contact> section is created when there are no rules."""
    schema = CommonSchema(
        metadata=Metadata(name="test_no_contact"),
        collision_config=CollisionConfig(
            groups={"default": CollisionGroup(name="default")}
        ),
        links=[
            Link(name="link1", mass=1.0, inertia=Inertia(ixx=0.1, iyy=0.1, izz=0.1))
        ]
    )
    
    exporter = MJCFExporter()
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test.xml"
        exporter.export(schema, output_path)
        
        tree = ET.parse(output_path)
        root = tree.getroot()
        
        # Should not have contact section
        contact = root.find("contact")
        assert contact is None


def test_mjcf_export_extension_overrides():
    """Test that extensions can override collision group values."""
    schema = CommonSchema(
        metadata=Metadata(name="test_extensions"),
        collision_config=CollisionConfig(
            groups={"default": CollisionGroup(name="default", contype=1, conaffinity=1)}
        ),
        links=[
            Link(
                name="link1",
                mass=1.0,
                inertia=Inertia(ixx=0.1, iyy=0.1, izz=0.1),
                collisions=[
                    Collision(
                        name="col_with_ext",
                        geometry=Geometry(type=GeometryType.BOX, size=Vector3(1, 1, 1)),
                        group="default",
                        extensions={
                            "contype": 8,  # Override via extension
                            "conaffinity": 15,
                            "margin": 0.002
                        }
                    )
                ]
            )
        ]
    )
    
    exporter = MJCFExporter()
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test.xml"
        exporter.export(schema, output_path)
        
        tree = ET.parse(output_path)
        root = tree.getroot()
        
        # Find collision geoms (class="collision")
        geoms = root.findall(".//geom[@class='collision']")
        assert len(geoms) >= 1
        geom = geoms[0]
        
        # Extensions should override group values
        assert geom.get("contype") == "8"
        assert geom.get("conaffinity") == "15"
        assert geom.get("margin") == "0.002"


def test_mjcf_export_missing_group_raises_error():
    """Test that missing collision group raises ValueError."""
    schema = CommonSchema(
        metadata=Metadata(name="test_missing_group"),
        collision_config=CollisionConfig(
            groups={"default": CollisionGroup(name="default")}
        ),
        links=[
            Link(
                name="link1",
                mass=1.0,
                inertia=Inertia(ixx=0.1, iyy=0.1, izz=0.1),
                collisions=[
                    Collision(
                        name="col1",
                        geometry=Geometry(type=GeometryType.BOX, size=Vector3(1, 1, 1)),
                        group="nonexistent"  # This group doesn't exist
                    )
                ]
            )
        ]
    )
    
    exporter = MJCFExporter()
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test.xml"
        with pytest.raises(ValueError, match="nonexistent"):
            exporter.export(schema, output_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
