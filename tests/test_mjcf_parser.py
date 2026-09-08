"""Tests for MJCF parser."""

from pathlib import Path

import pytest

from cyberwave_robot_format.mjcf import MJCFParser, MJCFExporter
from cyberwave_robot_format.schema import (
    ActuatorType,
    CommonSchema,
    Geometry,
    GeometryType,
    Link,
    Material,
    Metadata,
    Pose,
    Visual,
)


@pytest.fixture
def simple_mjcf_path():
    """Path to the simple robot MJCF fixture."""
    return Path(__file__).parent / "fixtures" / "simple_robot.xml"


@pytest.mark.parametrize("priority", [0, 5])
def test_contact_priority_roundtrips(priority, tmp_path):
    import xml.etree.ElementTree as ET

    source = tmp_path / "priority.xml"
    source.write_text(
        '<mujoco><worldbody><body name="object"><geom name="contact" '
        f'type="sphere" size="0.03" priority="{priority}"/></body></worldbody></mujoco>'
    )
    schema = MJCFParser().parse(source)
    collision = next(link for link in schema.links if link.name == "object").collisions[0]
    assert str(collision.extensions["priority"]) == str(priority)
    destination = tmp_path / "export.xml"
    MJCFExporter().export(schema, destination)
    geom = ET.parse(destination).find(f".//geom[@name='{collision.name}']")
    assert geom is not None
    assert geom.get("priority") == str(priority)


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


def test_mjcf_parser_position_actuator_defaults(tmp_path):
    """MuJoCo Menagerie policies commonly use defaulted position actuators."""
    mjcf_path = tmp_path / "position_actuator.xml"
    mjcf_path.write_text(
        """
<mujoco model="defaulted_position_actuator">
  <default>
    <default class="affine">
      <position kp="100" kv="2" ctrlrange="-1 1" forcerange="-80 80"/>
    </default>
  </default>
  <worldbody>
    <body name="base">
      <body name="leg">
        <joint name="hip" type="hinge" range="-0.5 0.5"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position class="affine" joint="hip" name="hip"/>
  </actuator>
</mujoco>
""",
        encoding="utf-8",
    )

    schema = MJCFParser().parse(mjcf_path)

    assert len(schema.actuators) == 1
    actuator = schema.actuators[0]
    assert actuator.name == "hip"
    assert actuator.joint == "hip"
    assert actuator.type == ActuatorType.POSITION
    assert actuator.kp == pytest.approx(100.0)
    assert actuator.kd == pytest.approx(2.0)
    assert actuator.control_range == pytest.approx((-1.0, 1.0))
    assert actuator.force_range == pytest.approx((-80.0, 80.0))
    assert actuator.max_torque == pytest.approx(80.0)


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


def test_mjcf_roundtrip_preserves_bare_geom_rgba(tmp_path):
    """A geom coloured with a bare `rgba` keeps its colour through the schema.

    The parser turns a bare `rgba` into a *nameless* Material. Only named
    materials reach the `<asset>` block, so the exporter has to put the colour
    back on the geom instead of dropping it.
    """
    source = tmp_path / "colours.xml"
    source.write_text(
        """<mujoco model="colours">
          <asset>
            <material name="named_red" rgba="1 0 0 1"/>
          </asset>
          <worldbody>
            <body name="base">
              <geom type="box" size="0.1 0.1 0.1" material="named_red"/>
              <geom type="sphere" size="0.05" rgba="0 0.4 0.9 1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )

    parser = MJCFParser()
    schema = parser.parse(source)

    # Survives the universal-schema JSON round trip.
    schema = CommonSchema.from_dict(schema.to_dict())

    output_path = tmp_path / "roundtrip.xml"
    MJCFExporter().export(schema, output_path)

    base = next(l for l in parser.parse(output_path).links if l.name == "base")
    colours = [v.material.color for v in base.visuals if v.material]
    assert [1.0, 0.0, 0.0, 1.0] in colours
    assert [0.0, 0.4, 0.9, 1.0] in colours


def test_mjcf_export_pads_a_colour_that_carries_no_alpha(tmp_path):
    """A three-component colour still exports as a compilable `rgba`.

    `USDParser` yields an RGB-only colour when a UsdPreviewSurface has no
    `opacity` input, and MuJoCo refuses to compile a model whose `rgba` is short
    of four reals. Alpha defaults to opaque rather than emitting a broken model.
    """
    schema = CommonSchema(metadata=Metadata(name="short_colour"))
    link = Link(name="base")
    link.visuals.append(
        Visual(
            name="v0",
            geometry=Geometry(type=GeometryType.SPHERE, radius=0.05),
            material=Material(color=[0.2, 0.4, 0.6]),
            pose=Pose(),
        )
    )
    schema.links.append(link)

    output_path = tmp_path / "short_colour.xml"
    MJCFExporter().export(schema, output_path)

    import xml.etree.ElementTree as ET

    geom = ET.parse(output_path).getroot().find(".//worldbody//geom[@rgba]")
    assert geom is not None
    assert len(geom.get("rgba").split()) == 4

    base = next(l for l in MJCFParser().parse(output_path).links if l.name == "base")
    assert base.visuals[0].material.color == [0.2, 0.4, 0.6, 1.0]


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
