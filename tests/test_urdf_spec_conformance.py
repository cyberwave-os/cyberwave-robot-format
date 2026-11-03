"""Regression tests for URDF spec conformance in the parse/export round trip.

Each test here covers a defect that was silent by construction: the parser and the
exporter agreed with each other while both disagreed with the URDF spec, so the
schema→URDF→schema fidelity check could never catch it.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from cyberwave_robot_format.schema import (
    CommonSchema,
    Geometry,
    GeometryType,
    Link,
    Metadata,
    Vector3,
    Visual,
)
from cyberwave_robot_format.urdf import URDF_DEFAULT_AXIS, URDFExporter, URDFParser


def _parse(tmp_path, urdf_text: str) -> CommonSchema:
    path = tmp_path / "robot.urdf"
    path.write_text(urdf_text)
    return URDFParser().parse(path)


def _joints(schema: CommonSchema) -> dict:
    return {joint.name: joint for joint in schema.joints}


# ------------------------------------------------------------------------ axis


_AXIS_URDF = """<robot name="t">
  <link name="base"/><link name="l1"/><link name="l2"/><link name="l3"/>
  <joint name="implicit" type="revolute">
    <parent link="base"/><child link="l1"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
  <joint name="explicit" type="revolute">
    <parent link="base"/><child link="l2"/><axis xyz="0 1 0"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
  <joint name="fixed_joint" type="fixed">
    <parent link="base"/><child link="l3"/>
  </joint>
</robot>"""


def test_missing_axis_defaults_to_the_urdf_spec_axis(tmp_path):
    # URDF: "axis (optional: defaults to (1,0,0))". The dataclass default is (0,0,1),
    # so a joint with no <axis> used to come back rotated 90 degrees off spec.
    joint = _joints(_parse(tmp_path, _AXIS_URDF))["implicit"]
    assert (joint.axis.x, joint.axis.y, joint.axis.z) == URDF_DEFAULT_AXIS
    assert URDF_DEFAULT_AXIS == (1.0, 0.0, 0.0)


def test_explicit_axis_is_untouched(tmp_path):
    joint = _joints(_parse(tmp_path, _AXIS_URDF))["explicit"]
    assert (joint.axis.x, joint.axis.y, joint.axis.z) == (0.0, 1.0, 0.0)


def test_defaulted_axis_survives_a_round_trip(tmp_path):
    # The exporter always writes an explicit <axis>, so a wrong default is not just
    # wrong in memory — it gets persisted into the served URDF on first re-render.
    once = _parse(tmp_path, _AXIS_URDF)
    rendered = URDFExporter().export_string(once)
    reparsed_path = tmp_path / "roundtrip.urdf"
    reparsed_path.write_text(rendered)
    joint = _joints(URDFParser().parse(reparsed_path))["implicit"]
    assert (joint.axis.x, joint.axis.y, joint.axis.z) == URDF_DEFAULT_AXIS


# ----------------------------------------------------------------------- mimic


def test_mimic_driver_reference_is_sanitized_like_joint_names(tmp_path):
    # Joint names are sanitized ("5" -> "_5") but the mimic reference was stored raw,
    # so the coupling pointed at a joint that does not exist in the schema.
    urdf = """<robot name="t">
      <link name="base"/><link name="l1"/><link name="l2"/>
      <joint name="5" type="revolute"><parent link="base"/><child link="l1"/>
        <limit lower="-1" upper="1" effort="1" velocity="1"/></joint>
      <joint name="6" type="revolute"><parent link="base"/><child link="l2"/>
        <limit lower="-1" upper="1" effort="1" velocity="1"/>
        <mimic joint="5" multiplier="-1" offset="0.02"/></joint>
    </robot>"""
    joints = _joints(_parse(tmp_path, urdf))
    slave = joints["_6"]
    assert slave.mimic is not None
    assert slave.mimic.joint == "_5"
    assert slave.mimic.joint in joints, "driver must resolve against the schema"
    assert slave.mimic.multiplier == -1.0
    assert slave.mimic.offset == 0.02


def test_self_referential_mimic_is_still_dropped(tmp_path):
    urdf = """<robot name="t">
      <link name="base"/><link name="l1"/>
      <joint name="5" type="revolute"><parent link="base"/><child link="l1"/>
        <limit lower="-1" upper="1" effort="1" velocity="1"/>
        <mimic joint="5" multiplier="1"/></joint>
    </robot>"""
    assert _joints(_parse(tmp_path, urdf))["_5"].mimic is None


# -------------------------------------------------------------------- material


_NAMELESS_MATERIAL_URDF = """<robot name="t">
  <link name="a"><visual><geometry><box size="1 1 1"/></geometry>
    <material name=""><color rgba="1 0 0 1"/></material></visual></link>
  <link name="b"><visual><geometry><box size="1 1 1"/></geometry>
    <material name=""><color rgba="1  0 0 1"/></material></visual></link>
  <link name="c"><visual><geometry><box size="1 1 1"/></geometry>
    <material name=""><color rgba="0 0 1 1"/></material></visual></link>
  <link name="d"><visual><geometry><box size="1 1 1"/></geometry>
    <material name=""/></visual></link>
  <joint name="ab" type="fixed"><parent link="a"/><child link="b"/></joint>
  <joint name="ac" type="fixed"><parent link="a"/><child link="c"/></joint>
  <joint name="ad" type="fixed"><parent link="a"/><child link="d"/></joint>
</robot>"""


def _visual_materials(schema: CommonSchema) -> dict:
    return {link.name: link.visuals[0].material for link in schema.links}


def test_nameless_visual_material_keeps_its_colour(tmp_path):
    # urdfdom reads a visual's <material name=""> and uses its colour, which is how
    # the SolidWorks exporter writes colours. It used to be dropped, so the visual
    # rendered in default grey everywhere except the browser's URDF viewer.
    materials = _visual_materials(_parse(tmp_path, _NAMELESS_MATERIAL_URDF))
    assert materials["a"].color == [1.0, 0.0, 0.0, 1.0]
    assert materials["c"].color == [0.0, 0.0, 1.0, 1.0]
    assert materials["d"] is None


def test_nameless_materials_are_named_by_content(tmp_path):
    # Exports key materials by name: equal colours may share one, different
    # colours must not.
    materials = _visual_materials(_parse(tmp_path, _NAMELESS_MATERIAL_URDF))
    assert materials["a"].name == materials["b"].name
    assert materials["a"].name != materials["c"].name


def test_nameless_material_colours_survive_a_round_trip(tmp_path):
    once = _parse(tmp_path, _NAMELESS_MATERIAL_URDF)
    reparsed_path = tmp_path / "roundtrip.urdf"
    reparsed_path.write_text(URDFExporter().export_string(once))
    materials = _visual_materials(URDFParser().parse(reparsed_path))
    assert materials["a"].color == [1.0, 0.0, 0.0, 1.0]
    assert materials["c"].color == [0.0, 0.0, 1.0, 1.0]


# -------------------------------------------------------------------- geometry


def _exported_geometry(geometry: Geometry) -> ET.Element:
    schema = CommonSchema(
        metadata=Metadata(name="t"),
        links=[Link(name="a", visuals=[Visual(geometry=geometry)])],
    )
    root = ET.fromstring(URDFExporter().export_string(schema))
    found = root.find(".//visual/geometry")
    assert found is not None
    return found


@pytest.mark.parametrize(
    ("geometry", "expected_child"),
    [
        (Geometry(type=GeometryType.CAPSULE, radius=0.1, length=0.5), "cylinder"),
        (Geometry(type=GeometryType.ELLIPSOID, size=Vector3(0.1, 0.2, 0.3)), "box"),
        (Geometry(type=GeometryType.UNKNOWN), "box"),
    ],
)
def test_inexpressible_geometry_still_emits_a_valid_shape(geometry, expected_child):
    # URDF requires exactly one shape child; an empty <geometry/> makes the whole
    # document unparseable for most consumers, so one unhandled type used to take
    # down the entire robot rather than one link.
    element = _exported_geometry(geometry)
    children = list(element)
    assert len(children) == 1, "URDF <geometry> needs exactly one shape child"
    assert children[0].tag == expected_child


def test_capsule_keeps_its_radius_and_length():
    cylinder = _exported_geometry(
        Geometry(type=GeometryType.CAPSULE, radius=0.1, length=0.5)
    )[0]
    assert float(cylinder.get("radius")) == pytest.approx(0.1)
    assert float(cylinder.get("length")) == pytest.approx(0.5)


def test_ellipsoid_exports_its_bounding_box():
    box = _exported_geometry(
        Geometry(type=GeometryType.ELLIPSOID, size=Vector3(0.1, 0.2, 0.3))
    )[0]
    # ``size`` carries semi-axes, so the box is twice each of them.
    assert [float(v) for v in box.get("size").split()] == pytest.approx([0.2, 0.4, 0.6])
