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


def test_gazebo_ray_noise_is_not_emitted(sensors_by_name):
    """``Sensor.noise`` has no cattrs structure hook, so populating it would make
    ``CommonSchema.from_dict`` raise and silently degrade the twin."""
    assert sensors_by_name["planar_scanner"].noise is None


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


def test_gazebo_camera_resolution_is_extracted(sensors_by_name):
    params = sensors_by_name["front_cam"].parameters
    assert params["width"] == 1280
    assert params["height"] == 720
