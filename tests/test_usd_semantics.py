# Copyright [2025] Cyberwave
# Licensed under the Apache License, Version 2.0

"""Unit tests for ``usd.semantics`` — the pxr-free rules both USD writers share.

No ``importorskip``: this module must work without ``usd-core``.
"""

from __future__ import annotations

import math

import pytest

from cyberwave_robot_format.fov_convention import KNOWN_FOV_KEYS
from cyberwave_robot_format.schema import (
    Actuator,
    ActuatorType,
    CommonSchema,
    Joint,
    JointType,
    Link,
    Metadata,
    Sensor,
)
from cyberwave_robot_format.usd import semantics as sem


class TestSensorKind:
    @pytest.mark.parametrize(
        ("sensor_type", "kind"),
        [
            ("camera", "rgb"),
            ("RGBD", "rgb"),
            ("depth_camera", "depth"),
            ("imu", "imu"),
            ("lidar_4d", "lidar"),
            ("pointcloud", "point_cloud"),
            ("ultrasonic", "range"),
            ("first_person_camera", "first_person_camera"),
            ("gps", None),
            ("force_torque", None),
            ("", None),
            (None, None),
        ],
    )
    def test_mapping(self, sensor_type, kind):
        assert sem.sensor_kind(sensor_type) == kind

    def test_every_mapped_kind_is_a_v1_kind(self):
        assert set(sem._SENSOR_KIND_BY_TYPE.values()) <= set(sem.SENSOR_KINDS)
        assert sem.CAMERA_KINDS <= set(sem.SENSOR_KINDS)


class TestCameraIntrinsics:
    def test_defaults_match_mujoco(self):
        intrinsics = sem.camera_intrinsics({})
        assert (intrinsics.width, intrinsics.height) == (640, 480)
        assert intrinsics.vfov_rad == pytest.approx(math.radians(45.0))
        assert intrinsics.fy == pytest.approx(240.0 / math.tan(math.radians(22.5)))
        assert intrinsics.fx == intrinsics.fy
        assert (intrinsics.cx, intrinsics.cy) == (320.0, 240.0)
        assert intrinsics.notes == ("camera_resolution_defaulted", "camera_fov_defaulted")

    def test_vertical_fov_in_degrees(self):
        intrinsics = sem.camera_intrinsics({"width": 100, "height": 100, "fovy": 90.0})
        assert intrinsics.fy == pytest.approx(50.0)
        assert intrinsics.notes == ()

    def test_horizontal_fov_goes_through_the_aspect_ratio(self):
        intrinsics = sem.camera_intrinsics({"width": 200, "height": 100, "hfov_deg": 90.0})
        assert intrinsics.fx == pytest.approx(100.0)
        assert intrinsics.fy == pytest.approx(100.0)
        assert intrinsics.vfov_rad == pytest.approx(2 * math.atan(0.5))

    def test_fx_only_means_square_pixels(self):
        intrinsics = sem.camera_intrinsics({"width": 480, "height": 270, "fx": 336.0})
        assert intrinsics.fx == intrinsics.fy == 336.0
        assert intrinsics.notes == ()

    def test_usd_aperture_reproduces_the_fov(self):
        intrinsics = sem.camera_intrinsics({"width": 1920, "height": 1080, "fovy": 50.0})
        vfov = 2 * math.atan(intrinsics.usd_vertical_aperture / (2 * intrinsics.usd_focal_length))
        hfov = 2 * math.atan(intrinsics.usd_horizontal_aperture / (2 * intrinsics.usd_focal_length))
        assert vfov == pytest.approx(math.radians(50.0))
        assert hfov == pytest.approx(intrinsics.hfov_rad)

    def test_off_centre_principal_point_becomes_aperture_offset(self):
        intrinsics = sem.camera_intrinsics({"width": 100, "height": 100, "fovy": 90.0, "cx": 40.0, "cy": 60.0})
        assert intrinsics.usd_horizontal_aperture_offset == pytest.approx(0.1 * intrinsics.usd_horizontal_aperture)
        assert intrinsics.usd_vertical_aperture_offset == pytest.approx(0.1 * intrinsics.usd_vertical_aperture)

    def test_clipping_from_depth_range(self):
        intrinsics = sem.camera_intrinsics({"min_depth": 0.001, "max_depth": 5000.0})
        assert (intrinsics.near_clip, intrinsics.far_clip) == (0.001, 5000.0)

    def test_fov_keys_cover_every_known_spelling_per_axis(self):
        # The precedence tuples are hand-ordered; the table decides axis and unit.
        by_axis = {
            axis: {k for k, v in KNOWN_FOV_KEYS.items() if v.axis == axis} for axis in ("vertical", "horizontal")
        }
        assert set(sem._VFOV_KEYS) == by_axis["vertical"]
        assert set(sem._HFOV_KEYS) == by_axis["horizontal"]
        assert sem._VFOV_KEYS[0] == "vfov_rad"
        assert sem._HFOV_KEYS[0] == "hfov_rad"


class TestNormalizedParams:
    def test_lidar_params_are_canonical_radians(self):
        sensor = Sensor(
            name="l",
            type="lidar",
            parent_link="a",
            update_rate=10.0,
            parameters={"hfov_deg": 360.0, "range_max": 30.0, "samples": 1800, "mode": "x"},
        )
        params = sem.normalized_sensor_params(sensor, "lidar")
        assert params == {
            "hfov_rad": pytest.approx(2 * math.pi),
            "max_range": 30.0,
            "mode": "x",
            "samples": 1800,
            "update_rate_hz": 10.0,
        }
        assert list(params) == sorted(params)

    def test_non_scalar_params_are_dropped(self):
        sensor = Sensor(name="s", type="imu", parent_link="a", parameters={"axes": [1, 2]})
        assert sem.normalized_sensor_params(sensor, "imu") == {}


def _site_schema(gear: str, **site) -> tuple[Actuator, dict]:
    link = Link(name="body", extensions={"mjcf_sites": [{"name": "r", **site}]})
    actuator = Actuator(
        name="m",
        joint="",
        type=ActuatorType.DC_MOTOR,
        control_range=(0.0, 2.0),
        extensions={"mjcf_site_transmission": {"site": "r", "gear": gear}},
    )
    schema = CommonSchema(metadata=Metadata(name="x"), links=[link], actuators=[actuator])
    return actuator, sem.sites_by_name(schema)


class TestRotors:
    def test_thrust_rotor(self):
        actuator, sites = _site_schema("0 0 5 0 0 -0.1", pos="1 2 3")
        rotor = sem.rotor_spec(actuator, sites)
        assert rotor is not None
        assert rotor.thrust_axis == (0.0, 0.0, 1.0)
        assert rotor.thrust_gain == 5.0
        assert rotor.torque_to_thrust == pytest.approx(-0.02)
        assert rotor.max_thrust == 10.0
        assert (rotor.position.x, rotor.position.y, rotor.position.z) == (1.0, 2.0, 3.0)
        assert rotor.to_custom_data()["thrustAxis"] == [0.0, 0.0, 1.0]

    def test_reaction_wheel_is_not_a_rotor(self):
        actuator, sites = _site_schema("0 0 0 0 0 1")
        assert sem.rotor_spec(actuator, sites) is None

    def test_zaxis_site_orientation(self):
        actuator, sites = _site_schema("0 0 1", zaxis="1 0 0")
        rotor = sem.rotor_spec(actuator, sites)
        assert rotor is not None and rotor.notes == ()
        # The site's +Z points along world +X.
        q = rotor.orientation
        assert abs(q.w) == pytest.approx(math.cos(math.pi / 4))

    def test_unsupported_site_orientation_is_noted(self):
        actuator, sites = _site_schema("0 0 1", fromto="0 0 0 0 0 1")
        rotor = sem.rotor_spec(actuator, sites)
        assert rotor is not None and "site_orientation_unsupported" in rotor.notes

    def test_joint_actuator_is_not_a_rotor(self):
        actuator = Actuator(name="a", joint="j", type=ActuatorType.POSITION)
        assert sem.rotor_spec(actuator, {}) is None


class TestJointsAndGains:
    def test_dof_indices_skip_fixed_and_floating(self):
        joints = [
            Joint(name="w", type=JointType.FLOATING, parent_link="world", child_link="a"),
            Joint(name="r", type=JointType.REVOLUTE, parent_link="a", child_link="b"),
            Joint(name="f", type=JointType.FIXED, parent_link="b", child_link="c"),
            Joint(name="p", type=JointType.PRISMATIC, parent_link="c", child_link="d"),
            Joint(name="c", type=JointType.CONTINUOUS, parent_link="d", child_link="e"),
        ]
        schema = CommonSchema(metadata=Metadata(name="x"), joints=joints)
        assert sem.dof_indices(schema) == {"r": 0, "p": 1, "c": 2}

    def test_primary_actuator_is_the_first(self):
        first = Actuator(name="pos", joint="j", type=ActuatorType.POSITION)
        second = Actuator(name="tau", joint="j", type=ActuatorType.TORQUE)
        schema = CommonSchema(metadata=Metadata(name="x"), actuators=[first, second])
        assert sem.primary_actuators(schema) == {"j": first}

    def test_angular_gain_round_trip(self):
        usd = sem.angular_gain_to_usd(100.0, True)
        assert usd == pytest.approx(100.0 * math.pi / 180.0)
        assert sem.angular_gain_from_usd(usd, True) == pytest.approx(100.0)
        assert sem.angular_gain_to_usd(100.0, False) == 100.0

    def test_rounded_normalizes_negative_zero(self):
        assert math.copysign(1.0, sem.rounded(-0.0)) == 1.0
        assert sem.rounded(0.1 + 0.2) == 0.3
