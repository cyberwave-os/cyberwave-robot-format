"""MJCF export of non-camera sensors, mount sites, and passive joints.

Before this, only ``camera``/``depth_camera`` survived export: everything else
logged "is not supported" and produced nothing, so the ``<sensor>`` block came
out empty. An ``imu`` in the schema could never reach MuJoCo, and any sensor that
*did* emit wrote ``site="<parent_link>"`` while no ``<site>`` was ever created —
which makes MuJoCo fail to compile the model rather than merely ignore it.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import pytest

from cyberwave_robot_format import CommonSchema, export_mujoco_scene_xml


def _schema(sensors, *, joints_passive=False, actuators=None):
    joints = [
        {
            "name": "left_wheel_joint",
            "type": "continuous",
            "parent_link": "base_link",
            "child_link": "left_wheel_link",
            "axis": {"x": 0, "y": 1, "z": 0},
            "pose": {"position": {"x": 0, "y": 0.2015, "z": 0.10}},
            "limits": {"effort": 100.0, "velocity": 20.0},
        },
        {
            "name": "caster_fl_wheel_joint",
            "type": "continuous",
            "parent_link": "base_link",
            "child_link": "caster_fl_wheel_link",
            "axis": {"x": 0, "y": 1, "z": 0},
            "pose": {"position": {"x": 0.30, "y": 0.188, "z": 0.0625}},
            **({"extensions": {"passive": True}} if joints_passive else {}),
        },
        {
            "name": "base_to_mount",
            "type": "fixed",
            "parent_link": "base_link",
            "child_link": "mount_link",
            "pose": {"position": {"x": 0.3164, "y": 0.2064, "z": 0.1914}},
        },
    ]
    return CommonSchema.from_dict(
        {
            "metadata": {"name": "t", "units": "SI"},
            "links": [
                {
                    "name": "base_link",
                    "collisions": [
                        {
                            "geometry": {
                                "type": "box",
                                "size": {"x": 0.8, "y": 0.6, "z": 0.27},
                            }
                        }
                    ],
                    "inertial": {
                        "mass": 97.0,
                        "inertia": {"ixx": 1.05, "iyy": 2.34, "izz": 2.75},
                    },
                },
                {
                    "name": "left_wheel_link",
                    "collisions": [
                        {"geometry": {"type": "cylinder", "radius": 0.1, "length": 0.038}}
                    ],
                    "inertial": {
                        "mass": 1.0,
                        "inertia": {"ixx": 0.003, "iyy": 0.005, "izz": 0.003},
                    },
                },
                {
                    "name": "caster_fl_wheel_link",
                    "collisions": [
                        {
                            "geometry": {
                                "type": "cylinder",
                                "radius": 0.0625,
                                "length": 0.032,
                            }
                        }
                    ],
                    "inertial": {
                        "mass": 1.0,
                        "inertia": {"ixx": 0.001, "iyy": 0.002, "izz": 0.001},
                    },
                },
                {"name": "mount_link"},
            ],
            "joints": joints,
            "sensors": sensors,
            "actuators": actuators or [],
        }
    )


def _export(sensors, **kwargs):
    return ET.fromstring(export_mujoco_scene_xml(_schema(sensors, **kwargs)))


def _sensor_block(root):
    return root.find("sensor")


def _sites(root):
    return {s.get("name"): s for s in root.iter("site")}


class TestSensorVocabulary:
    def test_imu_expands_to_gyro_and_accelerometer(self):
        """MuJoCo has no IMU primitive; it has a gyro and an accelerometer."""
        root = _export(
            [{"name": "imu", "type": "imu", "parent_link": "mount_link", "update_rate": 50.0}]
        )
        block = _sensor_block(root)
        assert [child.tag for child in block] == ["gyro", "accelerometer"]
        assert block[0].get("name") == "imu_gyro"
        assert block[1].get("name") == "imu_accelerometer"
        # Both must share one site, or they describe different frames.
        assert block[0].get("site") == block[1].get("site") == "imu_site"

    @pytest.mark.parametrize(
        "schema_type,expected_tag",
        [("ultrasound", "rangefinder"), ("sonar", "rangefinder"), ("contact", "touch")],
    )
    def test_device_types_map_to_mujoco_tags(self, schema_type, expected_tag):
        root = _export([{"name": "s", "type": schema_type, "parent_link": "mount_link"}])
        assert [child.tag for child in _sensor_block(root)] == [expected_tag]

    def test_raw_mujoco_tag_names_pass_through(self):
        root = _export([{"name": "vel", "type": "velocimeter", "parent_link": "mount_link"}])
        assert [child.tag for child in _sensor_block(root)] == ["velocimeter"]

    def test_single_primitive_sensor_keeps_its_own_name(self):
        """Downstream lookups key on the sensor id, so don't rename 1:1 sensors."""
        root = _export([{"name": "front_us", "type": "ultrasound", "parent_link": "mount_link"}])
        assert _sensor_block(root)[0].get("name") == "front_us"

    def test_lidar_is_skipped_without_being_treated_as_an_error(self, caplog):
        """Lidar is raycast at runtime; absence from MJCF is correct, not a gap."""
        with caplog.at_level("WARNING"):
            root = _export(
                [{"name": "front_laser", "type": "lidar_2d", "parent_link": "mount_link"}]
            )
        assert _sensor_block(root) is None or len(_sensor_block(root)) == 0
        assert "not supported" not in caplog.text

    def test_genuinely_unknown_type_still_warns(self, caplog):
        with caplog.at_level("WARNING"):
            _export([{"name": "x", "type": "tachyon_detector", "parent_link": "mount_link"}])
        assert "not supported" in caplog.text


class TestMountSites:
    def test_site_is_created_for_site_attached_sensors(self):
        root = _export([{"name": "imu", "type": "imu", "parent_link": "mount_link"}])
        assert "imu_site" in _sites(root)

    def test_site_is_created_on_the_parent_link_body(self):
        root = _export([{"name": "imu", "type": "imu", "parent_link": "mount_link"}])
        mount_body = next(b for b in root.iter("body") if b.get("name") == "mount_link")
        assert [s.get("name") for s in mount_body.findall("site")] == ["imu_site"]

    def test_sensor_references_a_site_that_actually_exists(self):
        """A sensor pointing at a missing site is an MJCF *compile* failure."""
        root = _export(
            [
                {"name": "imu", "type": "imu", "parent_link": "mount_link"},
                {"name": "us", "type": "ultrasound", "parent_link": "mount_link"},
            ]
        )
        declared = set(_sites(root))
        referenced = {
            child.get("site") for child in _sensor_block(root) if child.get("site")
        }
        assert referenced, "expected site-attached sensors"
        assert referenced <= declared

    def test_sensor_pose_lands_on_the_site(self):
        root = _export(
            [
                {
                    "name": "imu",
                    "type": "imu",
                    "parent_link": "mount_link",
                    "pose": {
                        "position": {"x": 0.1, "y": 0.2, "z": 0.3},
                        "orientation": {"w": 0.9238795, "x": 0, "y": 0, "z": 0.3826834},
                    },
                }
            ]
        )
        site = _sites(root)["imu_site"]
        assert site.get("pos") == "0.1 0.2 0.3"
        assert site.get("quat") is not None

    def test_site_is_created_for_a_sensor_on_the_ROOT_link(self):
        """Root links take a different export path than jointed children.

        ``base_link`` here is the root, and mounting the IMU on the chassis root is
        the common case for a mobile robot. That path emitted cameras only, so the
        ``<gyro site="imu_site">`` reference came out dangling and MuJoCo rejected
        the *entire* model — one root-mounted IMU took the whole scene down, which
        is strictly worse than the warn-and-drop it replaced.
        """
        root = _export([{"name": "imu", "type": "imu", "parent_link": "base_link"}])
        assert "imu_site" in _sites(root)
        base_body = next(b for b in root.iter("body") if b.get("name") == "base_link")
        assert [s.get("name") for s in base_body.findall("site")] == ["imu_site"]
        referenced = {child.get("site") for child in _sensor_block(root) if child.get("site")}
        assert referenced == {"imu_site"}
        assert referenced <= set(_sites(root))

    @pytest.mark.parametrize("sensor_type", ["rgb", "rgbd", "depth", "depth_camera"])
    def test_every_camera_spelling_becomes_a_model_camera(self, sensor_type):
        """The simulator renders by camera name: a camera stored as ``depth``
        or ``rgb`` (spellings the platform writes) must exist in the model."""
        root = _export(
            [{"name": "cam", "type": sensor_type, "parent_link": "base_link"}]
        )
        base_body = root.find(".//body[@name='base_link']")
        assert [c.get("name") for c in base_body.findall("camera")] == ["cam"]

    def test_root_link_camera_still_exports(self):
        """Guard the rewritten loop: cameras on the root must not regress."""
        root = _export([{"name": "cam", "type": "camera", "parent_link": "base_link"}])
        base_body = next(b for b in root.iter("body") if b.get("name") == "base_link")
        assert [c.get("name") for c in base_body.findall("camera")] == ["cam"]

    def test_vfov_rad_wins_over_a_disagreeing_legacy_fovy(self):
        vfov_rad = math.radians(45.0)
        root = _export(
            [
                {
                    "name": "cam",
                    "type": "camera",
                    "parent_link": "base_link",
                    "parameters": {"vfov_rad": vfov_rad, "fovy": 90.0},
                }
            ]
        )
        base_body = next(b for b in root.iter("body") if b.get("name") == "base_link")
        cam = base_body.find("camera")
        assert cam is not None
        assert float(cam.get("fovy")) == pytest.approx(45.0, abs=1e-6)

    def test_legacy_fovy_used_when_vfov_rad_is_absent(self):
        root = _export(
            [
                {
                    "name": "cam",
                    "type": "camera",
                    "parent_link": "base_link",
                    "parameters": {"fovy": 60.0},
                }
            ]
        )
        base_body = next(b for b in root.iter("body") if b.get("name") == "base_link")
        cam = base_body.find("camera")
        assert cam is not None
        assert float(cam.get("fovy")) == pytest.approx(60.0, abs=1e-6)

    def test_legacy_fovy_used_when_vfov_rad_is_invalid(self):
        root = _export(
            [
                {
                    "name": "cam",
                    "type": "camera",
                    "parent_link": "base_link",
                    "parameters": {"vfov_rad": 0, "fovy": 70.0},
                }
            ]
        )
        base_body = next(b for b in root.iter("body") if b.get("name") == "base_link")
        cam = base_body.find("camera")
        assert cam is not None
        assert float(cam.get("fovy")) == pytest.approx(70.0, abs=1e-6)

    def test_sensor_on_an_unknown_link_is_dropped_not_dangling(self):
        """An unmatched ``parent_link`` must cost that sensor, not the whole model.

        ``"world"`` is the realistic spelling: it matches no exported body, so no
        site is ever emitted for it.
        """
        root = _export(
            [
                {"name": "imu", "type": "imu", "parent_link": "world"},
                {"name": "good", "type": "imu", "parent_link": "mount_link"},
            ]
        )
        declared = set(_sites(root))
        referenced = {child.get("site") for child in _sensor_block(root) if child.get("site")}
        assert referenced <= declared, "a dangling site reference would not compile"
        # The well-formed sensor still exports.
        assert "good_site" in declared
        assert "imu_site" not in declared

    def test_no_site_for_joint_attached_sensors(self):
        root = _export(
            [
                {
                    "name": "enc",
                    "type": "jointpos",
                    "parent_link": "base_link",
                    "parameters": {"joint": "left_wheel_joint"},
                }
            ]
        )
        assert _sites(root) == {}
        assert _sensor_block(root)[0].get("joint") == "left_wheel_joint"


class TestCutoffAndNoise:
    def test_update_rate_is_not_written_as_cutoff(self):
        """``cutoff`` clamps the sensor *value*; it is not a sample rate."""
        root = _export(
            [
                {
                    "name": "imu",
                    "type": "imu",
                    "parent_link": "mount_link",
                    "update_rate": 12.5,
                }
            ]
        )
        for child in _sensor_block(root):
            assert child.get("cutoff") is None

    def test_rangefinder_cutoff_comes_from_max_range(self):
        root = _export(
            [
                {
                    "name": "us",
                    "type": "ultrasound",
                    "parent_link": "mount_link",
                    "update_rate": 20.0,
                    "parameters": {"max_range": 4.0},
                }
            ]
        )
        assert _sensor_block(root)[0].get("cutoff") == "4.0"

    def test_explicit_cutoff_wins(self):
        root = _export(
            [
                {
                    "name": "us",
                    "type": "ultrasound",
                    "parent_link": "mount_link",
                    "parameters": {"max_range": 4.0, "cutoff": 2.5},
                }
            ]
        )
        assert _sensor_block(root)[0].get("cutoff") == "2.5"

    def test_declared_noise_is_carried_onto_the_sensor(self):
        root = _export(
            [
                {
                    "name": "imu",
                    "type": "imu",
                    "parent_link": "mount_link",
                    "parameters": {"noise_stddev": 0.00017},
                }
            ]
        )
        assert _sensor_block(root)[0].get("noise") == "0.00017"


class TestPassiveJoints:
    def test_passive_joint_gets_no_auto_motor(self):
        """A caster must roll freely; motorizing it fights the physics."""
        root = _export([], joints_passive=True)
        names = [a.get("joint") for a in root.find("actuator")]
        assert "left_wheel_joint" in names
        assert "caster_fl_wheel_joint" not in names

    def test_unmarked_joints_keep_the_legacy_fallback(self):
        """Existing assets rely on auto-motors; don't silently disarm them."""
        root = _export([], joints_passive=False)
        names = [a.get("joint") for a in root.find("actuator")]
        assert {"left_wheel_joint", "caster_fl_wheel_joint"} <= set(names)

    def test_declared_actuators_bypass_the_fallback_entirely(self):
        root = _export(
            [],
            actuators=[
                {
                    "name": "left_wheel_velocity",
                    "joint": "left_wheel_joint",
                    "type": "velocity",
                    "max_torque": 250.0,
                    "kd": 35.0,
                    "control_range": [-15.0, 15.0],
                }
            ],
        )
        actuators = list(root.find("actuator"))
        assert [a.tag for a in actuators] == ["velocity"]
        assert actuators[0].get("joint") == "left_wheel_joint"
        assert actuators[0].get("kv") == "35.0"
        assert actuators[0].get("ctrlrange") == "-15.0 15.0"


@pytest.mark.parametrize("parent_link", ["base_link", "left_wheel_link"])
def test_camera_parent_remains_visible_to_other_cameras(parent_link):
    from cyberwave_robot_format.schema import Geometry, GeometryType, Visual

    schema = _schema([{"name": "cam", "type": "camera", "parent_link": parent_link}])
    link = next(link for link in schema.links if link.name == parent_link)
    link.visuals.append(Visual(geometry=Geometry(type=GeometryType.SPHERE, radius=0.1)))
    root = ET.fromstring(export_mujoco_scene_xml(schema))
    body = root.find(f".//body[@name='{parent_link}']")
    assert body is not None
    visual = body.find("geom[@class='visual']")
    assert visual is not None and visual.get("group") == "2"
    assert all(geom.get("group") != "5" for geom in body.findall("geom"))
