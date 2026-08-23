"""Tests for URDF parser."""

import json
import math
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

    base_link = next((lk for lk in schema.links if lk.name == "base_link"), None)
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


def _parse_urdf(tmp_path, urdf, filename):
    """Write ``urdf`` to ``filename`` and return (schema, parse_context)."""
    urdf_path = tmp_path / filename
    urdf_path.write_text(urdf, encoding="utf-8")
    schema = URDFParser().parse(urdf_path)
    return schema, schema.extensions["parse_context"]


@pytest.mark.parametrize("raw_type", [" revolute ", "Revolute", "REVOLUTE"])
def test_urdf_parser_normalizes_joint_type_case_and_padding(tmp_path, raw_type):
    """Vendor casing/padding must not cost the joint.

    An unnormalized value misses the type map, which used to drop the joint
    and detach ``arm`` from the tree.
    """
    urdf = f"""<?xml version="1.0"?>
<robot name="casing">
  <link name="base"/>
  <link name="arm"/>
  <joint name="shoulder" type="{raw_type}">
    <parent link="base"/>
    <child link="arm"/>
  </joint>
</robot>"""
    schema, parse_context = _parse_urdf(tmp_path, urdf, "casing.urdf")

    assert [j.name for j in schema.joints] == ["shoulder"]
    assert schema.joints[0].type.value == "revolute"
    assert parse_context["errors"] == []
    # The repair is silent to the caller unless it is recorded; the README
    # promises it in parse_context["warnings"].
    assert any("normalized to 'revolute'" in w for w in parse_context["warnings"])


def test_urdf_parser_keeps_a_typeless_joint_as_fixed(tmp_path):
    """A joint without ``type`` is kept rigid rather than dropped.

    Dropping it left the child link in ``links`` with nothing attaching it to
    the tree, so the robot imported with a detached body.
    """
    urdf = """<?xml version="1.0"?>
<robot name="missing_joint_type">
  <link name="base"/>
  <link name="camera_mount"/>
  <joint name="MHS_NAVCAM">
    <parent link="base"/>
    <child link="camera_mount"/>
  </joint>
</robot>"""
    schema, parse_context = _parse_urdf(tmp_path, urdf, "missing_joint_type.urdf")

    assert [j.name for j in schema.joints] == ["MHS_NAVCAM"]
    assert schema.joints[0].type.value == "fixed"
    assert schema.joints[0].child_link == "camera_mount"
    assert parse_context["errors"] == []
    assert any("assuming 'fixed'" in w for w in parse_context["warnings"])
    # The point of keeping it: no link is left dangling.
    assert not any("Orphaned links" in w for w in parse_context["warnings"])


def test_urdf_parser_reports_a_typeless_joint_that_declares_motion(tmp_path):
    """A typeless joint carrying <axis>/<limit> loses a DOF, so it must not be silent.

    "fixed" keeps the tree connected, which is the point of the recovery, but
    the file plainly described a moving joint. Only ``errors`` reaches Sentry,
    and nothing reads ``parse_context``, so a warning here goes unnoticed.
    """
    urdf = """<?xml version="1.0"?>
<robot name="armbot">
  <link name="base"/>
  <link name="arm"/>
  <joint name="shoulder">
    <parent link="base"/>
    <child link="arm"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="2.0"/>
  </joint>
</robot>"""
    schema, parse_context = _parse_urdf(tmp_path, urdf, "armbot.urdf")

    # Still kept, so the child link stays attached and composition succeeds.
    assert [j.name for j in schema.joints] == ["shoulder"]
    assert schema.joints[0].type.value == "fixed"
    assert not any("Orphaned links" in w for w in parse_context["warnings"])
    # ...but the discarded motion is reported, not merely warned about.
    assert any("only a moving joint can use" in e for e in parse_context["errors"]), (
        parse_context["errors"]
    )


def test_urdf_parser_reports_a_typeless_mimic_joint(tmp_path):
    """A typeless joint that mimics another cannot be rigid.

    ``<mimic>`` means "follow this other joint", so freezing the joint yields a
    weld carrying a coupling coefficient -- and the MJCF exporter drops the
    coupling silently, because it skips FIXED joints before it warns.
    """
    urdf = """<?xml version="1.0"?>
<robot name="gripper">
  <link name="base"/>
  <link name="left"/>
  <link name="right"/>
  <joint name="left_finger" type="prismatic">
    <parent link="base"/>
    <child link="left"/>
    <axis xyz="1 0 0"/>
    <limit lower="0" upper="0.04" effort="10" velocity="1"/>
  </joint>
  <joint name="right_finger">
    <parent link="base"/>
    <child link="right"/>
    <mimic joint="left_finger" multiplier="-1.0"/>
  </joint>
</robot>"""
    schema, parse_context = _parse_urdf(tmp_path, urdf, "gripper.urdf")

    assert [j.name for j in schema.joints] == ["left_finger", "right_finger"]
    assert any("<mimic>" in e for e in parse_context["errors"]), parse_context["errors"]


def test_urdf_parser_stays_quiet_for_dynamics_on_a_typeless_joint(tmp_path):
    """``<dynamics>`` must not trigger the report.

    Real vendor files carry damping/friction on genuinely ``type="fixed"``
    joints, so treating it as motion evidence would manufacture false reports.
    """
    urdf = """<?xml version="1.0"?>
<robot name="bracket">
  <link name="base"/>
  <link name="pad"/>
  <joint name="pad_joint">
    <parent link="base"/>
    <child link="pad"/>
    <dynamics damping="0.7" friction="0.05"/>
  </joint>
</robot>"""
    schema, parse_context = _parse_urdf(tmp_path, urdf, "bracket.urdf")

    assert schema.joints[0].type.value == "fixed"
    assert parse_context["errors"] == []
    assert any("assuming 'fixed'" in w for w in parse_context["warnings"])


def test_urdf_parser_error_quotes_the_unsupported_type_as_written(tmp_path):
    """Normalizing must not rewrite the spelling the file actually used."""
    urdf = """<?xml version="1.0"?>
<robot name="bogus_type">
  <link name="base"/>
  <link name="arm"/>
  <joint name="shoulder" type=" Screw ">
    <parent link="base"/>
    <child link="arm"/>
  </joint>
</robot>"""
    schema, parse_context = _parse_urdf(tmp_path, urdf, "bogus_type.urdf")

    assert schema.joints == []
    assert any("' Screw '" in e for e in parse_context["errors"]), parse_context[
        "errors"
    ]


def test_urdf_parser_still_rejects_an_unknown_joint_type(tmp_path):
    """Normalizing must not turn a genuinely unsupported type into a default."""
    urdf = """<?xml version="1.0"?>
<robot name="bogus_type">
  <link name="base"/>
  <link name="arm"/>
  <joint name="shoulder" type="screw">
    <parent link="base"/>
    <child link="arm"/>
  </joint>
</robot>"""
    schema, parse_context = _parse_urdf(tmp_path, urdf, "bogus_type.urdf")

    assert schema.joints == []
    assert any("Unsupported joint type" in e for e in parse_context["errors"])


@pytest.mark.parametrize("root_attrs", ['name=""', 'name="   "', ""])
def test_urdf_parser_falls_back_to_the_file_stem_for_a_nameless_robot(
    tmp_path, root_attrs
):
    """Blank, whitespace and absent root names all warn and use the file stem."""
    urdf = f"""<?xml version="1.0"?>
<robot {root_attrs}>
  <link name="base_link"/>
</robot>"""
    schema, parse_context = _parse_urdf(tmp_path, urdf, "nasa_jpl-ingenuity.urdf")

    assert schema.metadata.name == "nasa_jpl-ingenuity"
    assert parse_context["errors"] == []
    assert any("Robot name missing or empty" in w for w in parse_context["warnings"])


def test_urdf_parser_keeps_an_explicit_robot_name(tmp_path):
    """The fallback must not fire when the file names itself."""
    urdf = """<?xml version="1.0"?>
<robot name="explicitly_named">
  <link name="base_link"/>
</robot>"""
    schema, parse_context = _parse_urdf(tmp_path, urdf, "some_other_filename.urdf")

    assert schema.metadata.name == "explicitly_named"
    assert not any(
        "Robot name missing or empty" in w for w in parse_context["warnings"]
    )


def test_urdf_parser_preserves_mimic_joint(tmp_path):
    """Gripper-style URDF: slave joint mimic metadata must survive parsing."""
    urdf = """<?xml version="1.0"?>
<robot name="gripper_test">
  <link name="base"/><link name="finger_l"/><link name="finger_r"/>
  <joint name="finger_joint1" type="revolute">
    <parent link="base"/><child link="finger_l"/>
    <axis xyz="0 0 1"/>
    <limit lower="-0.8" upper="0.8" effort="10" velocity="1"/>
  </joint>
  <joint name="finger_joint2" type="revolute">
    <parent link="base"/><child link="finger_r"/>
    <mimic joint="finger_joint1" multiplier="-1" offset="0"/>
  </joint>
</robot>"""
    urdf_path = tmp_path / "gripper_test.urdf"
    urdf_path.write_text(urdf, encoding="utf-8")

    parser = URDFParser()
    schema = parser.parse(urdf_path)

    assert schema.metadata.name == "gripper_test"
    assert len(schema.joints) == 2

    driver = next(j for j in schema.joints if j.name == "finger_joint1")
    slave = next(j for j in schema.joints if j.name == "finger_joint2")

    assert driver.mimic is None
    assert slave.mimic is not None
    assert slave.mimic.joint == "finger_joint1"
    assert slave.mimic.multiplier == pytest.approx(-1.0)
    assert slave.mimic.offset == pytest.approx(0.0)


def test_urdf_parser_drops_self_referential_mimic(tmp_path):
    """B6: a <mimic> that points at its own joint is degenerate and dropped."""
    urdf = """<?xml version="1.0"?>
<robot name="selfref">
  <link name="base"/><link name="finger"/>
  <joint name="finger_joint" type="revolute">
    <parent link="base"/><child link="finger"/>
    <axis xyz="0 0 1"/>
    <limit lower="-0.8" upper="0.8" effort="10" velocity="1"/>
    <mimic joint="finger_joint" multiplier="-1" offset="0"/>
  </joint>
</robot>"""
    urdf_path = tmp_path / "selfref.urdf"
    urdf_path.write_text(urdf, encoding="utf-8")

    schema = URDFParser().parse(urdf_path)
    joint = next(j for j in schema.joints if j.name == "finger_joint")
    assert joint.mimic is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ---------------------------------------------------------------------------
# Gazebo sensor extensions
#
# Sensor specs live in <gazebo> blocks, which is the only place a vendor URDF
# states scan geometry, range, resolution and update rate. These were previously
# untested, which is how the <hfov> unit/axis defect below survived.
# ---------------------------------------------------------------------------

_SENSOR_URDF = """<?xml version="1.0"?>
<robot name="sensor_bot">
  <link name="base_link" />
  <link name="scan_link" />
  <joint name="base_to_scan" type="fixed">
    <parent link="base_link" /><child link="scan_link" />
  </joint>
  <link name="cam_link" />
  <joint name="base_to_cam" type="fixed">
    <parent link="base_link" /><child link="cam_link" />
  </joint>
  <link name="dome_link" />
  <joint name="base_to_dome" type="fixed">
    <parent link="base_link" /><child link="dome_link" />
  </joint>

  <gazebo reference="scan_link">
    <sensor name="planar_scanner" type="ray">
      <update_rate>12.5</update_rate>
      <ray>
        <scan><horizontal>
          <samples>541</samples>
          <min_angle>-2.35619449615</min_angle>
          <max_angle>2.35619449615</max_angle>
        </horizontal></scan>
        <range><min>0.05</min><max>29.0</max></range>
        <noise><type>gaussian</type><mean>0.0</mean><stddev>0.01</stddev></noise>
      </ray>
    </sensor>
  </gazebo>

  <gazebo reference="dome_link">
    <sensor name="dome_scanner" type="ray">
      <update_rate>10.0</update_rate>
      <ray>
        <scan>
          <horizontal><samples>360</samples><min_angle>-3.14159265</min_angle><max_angle>3.14159265</max_angle></horizontal>
          <vertical><samples>16</samples><min_angle>-0.261799</min_angle><max_angle>0.261799</max_angle></vertical>
        </scan>
        <range><min>0.1</min><max>100.0</max></range>
      </ray>
    </sensor>
  </gazebo>

  <gazebo reference="cam_link">
    <sensor name="front_cam" type="camera">
      <update_rate>30.0</update_rate>
      <camera>
        <hfov>1.5707963267948966</hfov>
        <image><width>1280</width><height>720</height></image>
      </camera>
    </sensor>
  </gazebo>
</robot>
"""


@pytest.fixture
def sensor_urdf_path(tmp_path):
    path = tmp_path / "sensor_bot.urdf"
    path.write_text(_SENSOR_URDF)
    return path


@pytest.fixture
def sensors_by_name(sensor_urdf_path):
    schema = URDFParser().parse(sensor_urdf_path)
    return {s.name: s for s in schema.sensors}


def test_gazebo_sensor_parent_link_and_rate(sensors_by_name):
    """Every sensor lands on the link it references, with its update rate."""
    assert set(sensors_by_name) == {"planar_scanner", "dome_scanner", "front_cam"}
    assert sensors_by_name["planar_scanner"].parent_link == "scan_link"
    assert sensors_by_name["planar_scanner"].update_rate == 12.5
    assert sensors_by_name["front_cam"].parent_link == "cam_link"
    assert sensors_by_name["front_cam"].update_rate == 30.0


def test_gazebo_ray_resolves_to_2d_when_scan_is_planar(sensors_by_name):
    """Gazebo calls every scanner "ray"; a horizontal-only scan is a 2D lidar.

    Left as "ray" the type reaches consumers unrecognized, invisible to range
    rendering and to the sensor-type union.
    """
    assert sensors_by_name["planar_scanner"].type == "lidar_2d"


def test_gazebo_ray_resolves_to_3d_when_scan_has_vertical_block(sensors_by_name):
    assert sensors_by_name["dome_scanner"].type == "lidar_3d"


def test_gazebo_ray_parameters_are_extracted(sensors_by_name):
    """<ray> scan/range convert to schema parameters; angles become degrees."""
    params = sensors_by_name["planar_scanner"].parameters
    assert params["fov_horizontal"] == pytest.approx(270.0, abs=1e-3)
    assert params["min_range"] == pytest.approx(0.05)
    assert params["max_range"] == pytest.approx(29.0)
    # samples are per-scan, so the rate has to be applied: 541 * 12.5
    assert params["points_per_second"] == 6762


def test_gazebo_ray_3d_multiplies_samples_across_both_axes(sensors_by_name):
    params = sensors_by_name["dome_scanner"].parameters
    assert params["fov_horizontal"] == pytest.approx(360.0, abs=1e-3)
    assert params["fov_vertical"] == pytest.approx(30.0, abs=1e-2)
    assert params["points_per_second"] == 360 * 16 * 10


def test_gazebo_ray_emits_hfov_rad_alongside_the_degree_field(sensors_by_name):
    params = sensors_by_name["planar_scanner"].parameters
    assert params["hfov_rad"] == pytest.approx(2.35619449615 * 2)
    assert params["hfov_rad"] == pytest.approx(math.radians(params["fov_horizontal"]))


def test_gazebo_ray_3d_emits_both_hfov_rad_and_vfov_rad(sensors_by_name):
    params = sensors_by_name["dome_scanner"].parameters
    assert params["hfov_rad"] == pytest.approx(3.14159265 * 2)
    assert params["vfov_rad"] == pytest.approx(0.261799 * 2)
    assert params["hfov_rad"] == pytest.approx(math.radians(params["fov_horizontal"]))
    assert params["vfov_rad"] == pytest.approx(
        math.radians(params["fov_vertical"]), abs=1e-6
    )


def test_gazebo_ray_noise_is_not_emitted(sensors_by_name):
    """``Sensor.noise`` has no cattrs structure hook, so populating it would make
    ``CommonSchema.from_dict`` raise and silently degrade the twin."""
    assert sensors_by_name["planar_scanner"].noise is None


_DEGENERATE_VERTICAL_URDF = """<?xml version="1.0"?>
<robot name="degenerate_bot">
  <link name="base_link" />
  <link name="scan_link" />
  <gazebo reference="scan_link">
    <sensor name="flat_dome_scanner" type="ray">
      <update_rate>10.0</update_rate>
      <ray>
        <scan>
          <horizontal><samples>360</samples><min_angle>-3.14159265</min_angle><max_angle>3.14159265</max_angle></horizontal>
          <vertical><samples>1</samples><min_angle>0</min_angle><max_angle>0</max_angle></vertical>
        </scan>
        <range><min>0.1</min><max>100.0</max></range>
      </ray>
    </sensor>
  </gazebo>
</robot>
"""


def test_a_degenerate_vertical_block_does_not_plant_a_zero_vfov_rad(tmp_path):
    # Regression: SDF's own default <vertical> (samples=1, min_angle==max_angle)
    # sweeps 0 rad. The old code wrote vfov_rad unconditionally, planting a
    # canonical key the write gate rejects as out of range and that
    # fov_canonicalize cannot purge -- canonical keys are excluded from
    # LEGACY_FOV_KEYS by construction, so the purge loop never reaches them.
    # The type resolver keys off presence of <vertical>, not extent, so this
    # sensor is still labeled lidar_3d and vertical is still a stored axis.
    path = tmp_path / "degenerate_bot.urdf"
    path.write_text(_DEGENERATE_VERTICAL_URDF)
    schema = URDFParser().parse(path)
    sensor = next(s for s in schema.sensors if s.name == "flat_dome_scanner")

    assert sensor.type == "lidar_3d"
    assert sensor.parameters["fov_vertical"] == pytest.approx(0.0)
    assert "vfov_rad" not in sensor.parameters
    # The horizontal sweep is real and must still get its canonical key.
    assert sensor.parameters["hfov_rad"] == pytest.approx(3.14159265 * 2)


def test_gazebo_camera_hfov_is_converted_to_degrees_not_copied_to_fovy(
    sensors_by_name,
):
    """Regression: <hfov> is horizontal and in radians.

    It used to be copied verbatim into ``fovy``, which consumers read as a
    vertical FOV in degrees — importing a 90-degree camera as a 1.57-degree
    pinhole.
    """
    params = sensors_by_name["front_cam"].parameters
    # native Gazebo name keeps radians
    assert params["hfov"] == pytest.approx(1.5707963267948966)
    # horizontal, in degrees
    assert params["fov_horizontal"] == pytest.approx(90.0)
    # vertical derived from the 1280x720 aspect: 2*atan(tan(45deg) * 720/1280)
    assert params["fov_vertical"] == pytest.approx(58.715, abs=1e-2)
    assert params["fovy"] == pytest.approx(58.715, abs=1e-2)
    # the old bug: fovy must never be the raw radian horizontal value
    assert params["fovy"] != pytest.approx(1.5707963267948966)


def test_gazebo_camera_emits_vfov_rad_but_never_hfov_rad(sensors_by_name):
    """A camera's FOV policy stores vertical only, so hfov_rad on a camera
    fails schema validation even though the Gazebo-native ``hfov`` value is
    legitimately radians."""
    params = sensors_by_name["front_cam"].parameters
    assert "hfov_rad" not in params
    assert params["hfov"] == pytest.approx(1.5707963267948966)
    assert params["vfov_rad"] == pytest.approx(math.radians(58.715), abs=1e-4)
    assert params["vfov_rad"] == pytest.approx(math.radians(params["fovy"]), abs=1e-9)
    assert params["vfov_rad"] != pytest.approx(1.5707963267948966)


def test_gazebo_camera_resolution_is_extracted(sensors_by_name):
    params = sensors_by_name["front_cam"].parameters
    assert params["width"] == 1280
    assert params["height"] == 720


def test_joint_origin_rpy_round_trips_through_parse_and_export(tmp_path):
    """``parse -> export -> parse`` must preserve a joint's zero exactly.

    The backend's stored-URDF sync writes ``<origin xyz rpy>`` back from
    ``joints[i].pose`` using ``Quaternion.to_rpy``, and a wholesale schema
    rebuild then re-parses that file. If this fixed point ever breaks, a joint's
    authored zero silently drifts (or reverts) on every rebuild — so pin it here,
    where a refactor of the conversion breaks first.
    """
    from cyberwave_robot_format.urdf import URDFExporter

    source = tmp_path / "rotated.urdf"
    source.write_text(
        """<robot name="rotated">
  <link name="base"/>
  <link name="l1"/>
  <joint name="j1" type="revolute">
    <parent link="base"/>
    <child link="l1"/>
    <origin xyz="0.1 -0.2 0.3" rpy="0.4 -0.5 0.6"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
</robot>"""
    )

    first = URDFParser().parse(source)
    exported = tmp_path / "roundtrip.urdf"
    URDFExporter().export(first, exported)
    second = URDFParser().parse(exported)

    before = first.joints[0].pose
    after = second.joints[0].pose

    assert after.position.x == pytest.approx(before.position.x, abs=1e-12)
    assert after.position.y == pytest.approx(before.position.y, abs=1e-12)
    assert after.position.z == pytest.approx(before.position.z, abs=1e-12)

    # q and -q are the same rotation, so compare via |dot| rather than per-component.
    dot = (
        before.orientation.x * after.orientation.x
        + before.orientation.y * after.orientation.y
        + before.orientation.z * after.orientation.z
        + before.orientation.w * after.orientation.w
    )
    assert abs(dot) == pytest.approx(1.0, abs=1e-12)
