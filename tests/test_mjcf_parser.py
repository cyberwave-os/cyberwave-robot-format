"""Tests for MJCF parser."""

import json
from pathlib import Path

import pytest

from cyberwave_robot_format.core import ParseError
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


def test_non_finite_attributes_never_enter_the_schema(tmp_path):
    """MJCF ingest is a user-upload path (``import/mujoco-scene``), and a bare
    ``float()`` accepts ``nan``/``inf``/``Infinity``. Each non-finite attribute is
    reported and dropped like non-numeric text, so the schema stays exportable."""
    mjcf = """<mujoco model="hostile">
      <worldbody>
        <body name="b1" pos="nan 0 0">
          <inertial pos="0 0 0" mass="Infinity" diaginertia="0.1 0.1 0.1"/>
          <joint name="j1" type="hinge" axis="0 0 1" range="-inf inf"/>
          <geom type="box" size="0.1 0.1 0.1" rgba="1 0 0 nan"/>
        </body>
      </worldbody>
    </mujoco>"""
    path = tmp_path / "hostile.xml"
    path.write_text(mjcf)

    schema = MJCFParser().parse(path)

    assert schema.validate() == []
    parse_context = schema.extensions["parse_context"]
    assert any("Invalid mass value" in error for error in parse_context["errors"])
    warnings = parse_context["warnings"]
    link = schema.get_link("b1")
    assert link.mass == 0.0  # the Infinity mass was refused, not stored
    assert any("Invalid position for body b1" in warning for warning in warnings)
    joint = schema.get_joint("j1")
    assert joint.limits is None or (
        joint.limits.lower is None and joint.limits.upper is None
    )
    # The whole schema now serializes as strict JSON.
    json.dumps(schema.to_dict(), allow_nan=False)


@pytest.mark.parametrize(
    "inertial",
    [
        'pos="0 0 0" mass="inf" diaginertia="0.1 0.1 0.1"',
        'pos="0 0 0" mass="1" diaginertia="inf inf inf"',
        'pos="inf 0 0" mass="1" diaginertia="0.1 0.1 0.1"',
        'pos="0 0 0" quat="nan 0 0 1" mass="1" diaginertia="0.1 0.1 0.1"',
        'pos="0 0 0" euler="nan 0 0" mass="1" diaginertia="0.1 0.1 0.1"',
    ],
    ids=["mass", "inertia", "center_of_mass", "quat", "euler"],
)
def test_resolve_inertia_rejects_a_non_finite_compiled_value(tmp_path, inertial):
    """``resolve_inertia=True`` re-reads mass, inertia and center of mass from the
    compiled MuJoCo model, which accepts ``inf`` for all three and a NaN inertial
    orientation, so the attribute-level drop is undone there unless the compiled
    values are checked too."""
    pytest.importorskip("mujoco")
    mjcf = f"""<mujoco model="hostile">
      <worldbody>
        <body name="b1">
          <inertial {inertial}/>
          <joint name="j1" type="hinge" axis="0 0 1"/>
          <geom type="box" size="0.1 0.1 0.1"/>
        </body>
      </worldbody>
    </mujoco>"""
    path = tmp_path / "hostile.xml"
    path.write_text(mjcf)

    with pytest.raises(ParseError, match="non-finite mass, inertia or center of mass"):
        MJCFParser().parse(path, resolve_inertia=True)


# Float max. A rotated diagonal tensor's entries are bounded by its largest
# eigenvalue, so it only overflows through rounding. ``20 0 0 99`` (wxyz) has an
# exact norm of 101 and trimesh builds its rotation elementwise, so the matrix is
# the same on every platform, and its row norm² sits ~4.6 ulps above 1. That puts
# the exact tensor entry ~17 half-ulps of the float maximum past the round-to-inf
# threshold, so every BLAS summation order or FMA choice in
# ``rotation @ D @ rotation.T`` still rounds to inf.
_FLOAT_MAX = "1.7976931348623157e308"


@pytest.mark.parametrize(
    ("body", "warning", "error"),
    [
        ('<geom type="box" size="1e308 0.1 0.1"/>', "Invalid size values for geometry in link b1", None),
        ('<geom type="cylinder" size="0.1 1e308"/>', "Invalid size values for geometry in link b1", None),
        ('<geom type="capsule" size="0.1 1e308"/>', "Invalid size values for geometry in link b1", None),
        (
            f'<inertial pos="0 0 0" mass="1" quat="20 0 0 99" diaginertia="{_FLOAT_MAX} {_FLOAT_MAX} 1"/>'
            '<geom type="sphere" size="0.1"/>',
            None,
            "Invalid inertia values for link b1",
        ),
    ],
    ids=["box", "cylinder", "capsule", "rotated_inertia"],
)
def test_a_finite_value_that_overflows_in_conversion_never_enters_the_schema(tmp_path, body, warning, error):
    """The parser doubles box half-sizes and cylinder/capsule half-lengths and
    rotates a diagonal inertia, so a finite attribute near the float maximum
    becomes ``inf`` without any ``nan``/``inf`` text in the file."""
    mjcf = f"""<mujoco model="hostile">
      <worldbody>
        <body name="b1">
          <joint name="j1" type="hinge" axis="0 0 1"/>
          {body}
        </body>
      </worldbody>
    </mujoco>"""
    path = tmp_path / "hostile.xml"
    path.write_text(mjcf)

    schema = MJCFParser().parse(path)

    json.dumps(schema.to_dict(), allow_nan=False)
    parse_context = schema.extensions["parse_context"]
    if warning:
        assert warning in parse_context["warnings"]
    if error:
        assert error in parse_context["errors"]
        # The overflowing rotation falls back to the unrotated diagonal, as an
        # invalid inertial quaternion already does.
        inertia = next(link for link in schema.links if link.name == "b1").inertia
        assert (inertia.ixx, inertia.iyy, inertia.izz) == (float(_FLOAT_MAX), float(_FLOAT_MAX), 1.0)
        assert (inertia.ixy, inertia.ixz, inertia.iyz) == (0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    ("geom", "check"),
    [
        ('<geom type="sphere" size="1e308"/>', lambda g: g.radius == 1e308),
        ('<geom type="ellipsoid" size="1e308 0.1 0.1"/>', lambda g: g.size.x == 1e308),
        ('<geom type="plane" size="1e308 1e308 0.1"/>', lambda g: g.size.x == 1e308),
        # Only the half-length is doubled; the radius is stored as is.
        ('<geom type="cylinder" size="1e308 0.1"/>', lambda g: g.radius == 1e308 and g.length == 0.2),
        ('<geom type="capsule" size="1e308 0.1"/>', lambda g: g.radius == 1e308 and g.length == 0.2),
    ],
    ids=["sphere", "ellipsoid", "plane", "cylinder_radius", "capsule_radius"],
)
def test_a_large_size_that_is_never_doubled_is_kept(tmp_path, geom, check):
    """The doubling overflow guard covers only the values the parser doubles, so a
    large but finite size that is stored unchanged still parses."""
    mjcf = f"""<mujoco model="large">
      <worldbody>
        <body name="b1">
          <joint name="j1" type="hinge" axis="0 0 1"/>
          {geom}
        </body>
      </worldbody>
    </mujoco>"""
    path = tmp_path / "large.xml"
    path.write_text(mjcf)

    schema = MJCFParser().parse(path)

    parse_context = schema.extensions["parse_context"]
    assert "Invalid size values for geometry in link b1" not in parse_context["warnings"]
    link = schema.get_link("b1")
    assert link.visuals and check(link.visuals[0].geometry)
    json.dumps(schema.to_dict(), allow_nan=False)


def test_a_joint_position_that_overflows_with_its_body_position_is_rejected(tmp_path):
    """A joint's schema position is its own plus its body's; two finite values near
    the float maximum sum to ``inf``."""
    mjcf = """<mujoco model="hostile">
      <worldbody>
        <body name="b1" pos="1e308 0 0">
          <joint name="j1" type="hinge" pos="1e308 0 0" axis="0 0 1"/>
          <geom type="sphere" size="0.1"/>
        </body>
      </worldbody>
    </mujoco>"""
    path = tmp_path / "hostile.xml"
    path.write_text(mjcf)

    with pytest.raises(ParseError, match="'j1' position overflows"):
        MJCFParser().parse(path)


def test_finite_attributes_still_parse(tmp_path):
    """The guard must not cost the ordinary case."""
    mjcf = """<mujoco model="fine">
      <worldbody>
        <body name="b1" pos="0.1 0.2 0.3">
          <inertial pos="0 0 0" mass="2.5" diaginertia="0.1 0.1 0.1"/>
          <joint name="j1" type="hinge" axis="0 0 1" range="-1.5 1.5"/>
          <geom type="box" size="0.1 0.1 0.1"/>
        </body>
      </worldbody>
    </mujoco>"""
    path = tmp_path / "fine.xml"
    path.write_text(mjcf)

    schema = MJCFParser().parse(path)

    assert schema.get_link("b1").mass == 2.5
    limits = schema.get_joint("j1").limits
    assert (limits.lower, limits.upper) == (-1.5, 1.5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
