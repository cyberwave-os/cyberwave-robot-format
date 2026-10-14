# Copyright [2025] Cyberwave
# Licensed under the Apache License, Version 2.0

"""Pure-Python semantics shared by the USD exporters: sensor kinds, camera
intrinsics, rotors, control modes and the canonical DOF order.

Nothing here imports ``pxr``. It decides *what* a CommonSchema object means for a
simulator reading the portable USD schema; :mod:`usd.exporter` and
:mod:`usd.scene` decide how to author it. Keeping the decisions here makes
them testable without ``usd-core`` and keeps the two exporters from disagreeing.

Every derived float is rounded to :data:`FLOAT_DIGITS` decimal places. The inputs
are already deterministic, but ``tan``/``atan`` may differ in the last ulp across
libm builds, and a bundle digest must not depend on the host that built it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cyberwave_robot_format.fov_convention import find_fov_value
from cyberwave_robot_format.schema import (
    Actuator,
    ActuatorType,
    CommonSchema,
    JointType,
    Link,
    Quaternion,
    Sensor,
    Vector3,
)
from cyberwave_robot_format.usd import frames as f

FLOAT_DIGITS = 12

# --------------------------------------------------------------------------
# Sensor kinds
# --------------------------------------------------------------------------

SENSOR_KINDS = (
    "rgb",
    "depth",
    "imu",
    "lidar",
    "range",
    "point_cloud",
    "first_person_camera",
    "sim_viewport",
)
"""The frozen v1 sensor modalities (``simulation.feature_profile.v1``)."""

CAMERA_KINDS = frozenset({"rgb", "depth", "first_person_camera", "sim_viewport"})
"""Kinds authored as ``UsdGeom.Camera`` rather than a parametric ``Xform``."""

_SENSOR_KIND_BY_TYPE: dict[str, str] = {
    # RGB. ``rgbd`` is streamed as RGB only by cyberwave-sim today (only
    # depth/depth_camera count as depth), so it maps the same way here.
    "camera": "rgb",
    "rgb": "rgb",
    "rgb_camera": "rgb",
    "color_camera": "rgb",
    "rgbd": "rgb",
    # Depth.
    "depth": "depth",
    "depth_camera": "depth",
    # IMU.
    "imu": "imu",
    # Scanning range sensors, sampled at runtime (mj_ray in cyberwave-sim).
    "lidar": "lidar",
    "lidar_2d": "lidar",
    "lidar_3d": "lidar",
    "lidar_4d": "lidar",
    "laser": "lidar",
    "laser_scan": "lidar",
    "pointcloud": "point_cloud",
    "point_cloud": "point_cloud",
    # Single-beam range.
    "range": "range",
    "rangefinder": "range",
    "sonar": "range",
    "ultrasound": "range",
    "ultrasonic": "range",
    "tof": "range",
    # Explicit first-person camera.
    "first_person_camera": "first_person_camera",
}


def sensor_kind(sensor_type: str | None) -> str | None:
    """Return the v1 modality for a free-form schema sensor ``type``, or ``None``.

    ``None`` means the sensor has no v1 modality (``gps``, ``radar``,
    ``force_torque``, ``thermal`` ...). It is still exported with its raw
    ``cyberwave:sensor:*`` attributes; the scene exporter reports it as a
    rejected feature rather than silently dropping it.
    """
    if not sensor_type:
        return None
    return _SENSOR_KIND_BY_TYPE.get(sensor_type.strip().lower())


THERMAL_DISPLAY_PALETTE = "thermal"
"""``params.display_palette`` value that marks an RGB sensor as a thermal camera."""

THERMAL_FIDELITY = "heuristic"
"""Declared fidelity of a thermal camera: a link heat model, not radiometry."""

_THERMAL_IMAGE_SENSOR_TYPES = frozenset({"rgb", "camera"})


def is_thermal_sensor(sensor: Sensor) -> bool:
    """Whether ``sensor`` is a thermal camera, by cyberwave-sim's rule.

    Same predicate as cyberwave-sim's ``camera/thermal.py``, the frontend's
    ``isThermalCamera`` and the AX8 driver: an RGB sensor (``type`` exactly
    ``rgb`` or ``camera``) whose ``extensions.cyberwave.display_palette`` is
    ``"thermal"``, or, without that key, whose ``parameters.color_mode`` is
    ``"thermal"``. A thermal camera is a display type on an RGB stream, so its
    v1 kind stays ``rgb``; the exporter marks it with
    ``params.display_palette = "thermal"`` for the runtime.
    """
    if sensor.type not in _THERMAL_IMAGE_SENSOR_TYPES:
        return False
    extensions = sensor.extensions or {}
    cyberwave = extensions.get("cyberwave") if isinstance(extensions, dict) else None
    if isinstance(cyberwave, dict) and "display_palette" in cyberwave:
        return bool(cyberwave.get("display_palette") == THERMAL_DISPLAY_PALETTE)
    parameters = sensor.parameters or {}
    return isinstance(parameters, dict) and parameters.get("color_mode") == THERMAL_DISPLAY_PALETTE


# --------------------------------------------------------------------------
# Camera intrinsics
# --------------------------------------------------------------------------

USD_DEFAULT_HORIZONTAL_APERTURE = 20.955
"""``UsdGeomCamera``'s schema default (tenths of a scene unit, i.e. "mm").

Only the focal-length/aperture *ratio* is physically meaningful; anchoring the
aperture to the schema default keeps the authored numbers familiar.
"""

DEFAULT_CAMERA_WIDTH = 640
DEFAULT_CAMERA_HEIGHT = 480
DEFAULT_VFOV_RAD = math.radians(45.0)
"""MuJoCo's default ``fovy`` — what cyberwave-sim renders when none is given."""
DEFAULT_NEAR_CLIP = 0.01
DEFAULT_FAR_CLIP = 1000.0

# Lookup precedence, canonical radians first; the axis and unit of each name
# come from fov_convention.KNOWN_FOV_KEYS, whose order is not a precedence.
# tests/test_usd_semantics.py keeps these in step with that table.
_VFOV_KEYS = (
    "vfov_rad",
    "vfov_deg",
    "fovy",
    "fov_degrees",
    "fov_vertical",
    "vertical_fov",
    "vertical_fov_degrees",
    "vfov",
)
_HFOV_KEYS = (
    "hfov_rad",
    "hfov_deg",
    "horizontal_fov",
    "hfov",
    "fov_horizontal",
    "horizontal_fov_degrees",
)
_FOV_KEYS = frozenset(_VFOV_KEYS + _HFOV_KEYS + ("dfov_rad", "dfov_deg"))


@dataclass(frozen=True)
class CameraIntrinsics:
    """A pinhole camera, in pixels and radians, plus its USD encoding.

    ``fx``/``fy``/``cx``/``cy`` follow the OpenCV convention (``v`` grows down).
    The ``usd_*`` fields are the ``UsdGeomCamera`` attributes that reproduce the
    same projection: ``fx = focal * width / hAperture``,
    ``fy = focal * height / vAperture``, and the aperture offsets move the
    principal point (``cx = width * (0.5 - hOffset / hAperture)``,
    ``cy = height * (0.5 + vOffset / vAperture)``).
    """

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    hfov_rad: float
    vfov_rad: float
    near_clip: float
    far_clip: float
    usd_focal_length: float
    usd_horizontal_aperture: float
    usd_vertical_aperture: float
    usd_horizontal_aperture_offset: float
    usd_vertical_aperture_offset: float
    notes: tuple[str, ...] = ()
    """Codes for values that were defaulted rather than read from the sensor."""


def _number(params: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = params.get(key)
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        if math.isfinite(float(value)):
            return float(value)
    return None


def rounded(value: float) -> float:
    """Round to :data:`FLOAT_DIGITS` places, normalizing ``-0.0`` to ``0.0``."""
    result = round(float(value), FLOAT_DIGITS)
    return 0.0 if result == 0.0 else result


_rounded = rounded


def camera_intrinsics(
    parameters: dict[str, Any] | None,
    *,
    default_width: int = DEFAULT_CAMERA_WIDTH,
    default_height: int = DEFAULT_CAMERA_HEIGHT,
) -> CameraIntrinsics:
    """Resolve a camera sensor's parameters into a full pinhole model.

    Precedence: the vertical FOV (``_VFOV_KEYS``: canonical ``vfov_rad``, then
    the legacy spellings), else the horizontal FOV (``_HFOV_KEYS``) converted
    through the aspect ratio, else ``fy``, else ``fx`` (square pixels), else
    MuJoCo's 45 degrees. The MJCF exporter reads a narrower set (``vfov_rad``,
    ``fovy``, ``fov_degrees``, ``hfov``), so a camera described only by another
    spelling gets its FOV here and MuJoCo's default there.
    ``fx``/``fy`` default to square pixels and ``cx``/``cy`` to the image centre,
    exactly what cyberwave-sim publishes for a MuJoCo camera.
    """
    params = parameters or {}
    notes: list[str] = []

    width_value = _number(params, "width", "image_width")
    height_value = _number(params, "height", "image_height")
    if width_value is None or height_value is None or width_value <= 0 or height_value <= 0:
        notes.append("camera_resolution_defaulted")
        width_value = width_value if width_value and width_value > 0 else default_width
        height_value = height_value if height_value and height_value > 0 else default_height
    width = int(round(width_value))
    height = int(round(height_value))

    fx = _number(params, "fx")
    fy = _number(params, "fy")
    vfov: float | None = None
    vertical = find_fov_value(params, _VFOV_KEYS)
    horizontal = find_fov_value(params, _HFOV_KEYS)
    if vertical is not None and vertical[1] > 0:
        vfov = vertical[1]
    elif horizontal is not None and horizontal[1] > 0:
        vfov = 2.0 * math.atan(math.tan(horizontal[1] / 2.0) * height / width)
    elif fy is not None and fy > 0:
        vfov = 2.0 * math.atan(height / (2.0 * fy))
    elif fx is not None and fx > 0:
        # Only fx given: square pixels, so it fixes the vertical FOV too.
        fy = fx
        vfov = 2.0 * math.atan(height / (2.0 * fy))
    else:
        notes.append("camera_fov_defaulted")
        vfov = DEFAULT_VFOV_RAD

    if fy is None or fy <= 0:
        fy = height / (2.0 * math.tan(vfov / 2.0))
    if fx is None or fx <= 0:
        if horizontal is not None and horizontal[1] > 0 and vertical is not None:
            # Both axes given explicitly: honour both (non-square pixels).
            fx = width / (2.0 * math.tan(horizontal[1] / 2.0))
        else:
            fx = fy
    hfov = 2.0 * math.atan(width / (2.0 * fx))

    cx = _number(params, "cx")
    cy = _number(params, "cy")
    cx = width / 2.0 if cx is None else cx
    cy = height / 2.0 if cy is None else cy

    near = _number(params, "near_clip", "clip_near", "near", "znear")
    far = _number(params, "far_clip", "clip_far", "far", "zfar")
    min_depth = _number(params, "min_depth")
    max_depth = _number(params, "max_depth")
    if near is None or near <= 0:
        near = DEFAULT_NEAR_CLIP
        if min_depth is not None and 0 < min_depth < near:
            near = min_depth
    if far is None or far <= near:
        far = DEFAULT_FAR_CLIP
        if max_depth is not None and max_depth > far:
            far = max_depth

    h_aperture = USD_DEFAULT_HORIZONTAL_APERTURE
    focal = fx * h_aperture / width
    v_aperture = focal * height / fy
    h_offset = (0.5 - cx / width) * h_aperture
    v_offset = (cy / height - 0.5) * v_aperture

    return CameraIntrinsics(
        width=width,
        height=height,
        fx=_rounded(fx),
        fy=_rounded(fy),
        cx=_rounded(cx),
        cy=_rounded(cy),
        hfov_rad=_rounded(hfov),
        vfov_rad=_rounded(vfov),
        near_clip=_rounded(near),
        far_clip=_rounded(far),
        usd_focal_length=_rounded(focal),
        usd_horizontal_aperture=_rounded(h_aperture),
        usd_vertical_aperture=_rounded(v_aperture),
        usd_horizontal_aperture_offset=_rounded(h_offset),
        usd_vertical_aperture_offset=_rounded(v_offset),
        notes=tuple(notes),
    )


# --------------------------------------------------------------------------
# Normalized sensor parameters
# --------------------------------------------------------------------------


def _scalar(value: Any) -> Any | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return _rounded(value) if math.isfinite(value) else None
    if isinstance(value, str):
        return value
    return None


def sensor_update_rate(sensor: Sensor) -> float | None:
    """``Sensor.update_rate``, else the ``update_rate``/``fps`` parameter (Hz)."""
    if sensor.update_rate is not None and sensor.update_rate > 0:
        return float(sensor.update_rate)
    value = _number(sensor.parameters or {}, "update_rate", "fps", "rate_hz", "rate")
    return value if value is not None and value > 0 else None


def normalized_sensor_params(sensor: Sensor, kind: str | None) -> dict[str, Any]:
    """Return the sensor's parameters as canonical scalars (SI, radians), sorted.

    Canonical keys derived per kind are authoritative; any other scalar raw
    parameter is carried through unchanged so a runtime can still read
    vendor-specific knobs (``velocity_sensing``, ``color_mode`` ...). Legacy FOV
    spellings are dropped in favour of their canonical ``*_rad`` values, and
    non-scalar values (lists, dicts) stay only in the raw
    ``cyberwave:sensor:parameters`` attribute.
    """
    raw = sensor.parameters or {}
    derived: dict[str, Any] = {}

    rate = sensor_update_rate(sensor)
    if rate is not None:
        derived["update_rate_hz"] = _rounded(rate)

    if kind in CAMERA_KINDS:
        intrinsics = camera_intrinsics(raw)
        derived.update(
            width=intrinsics.width,
            height=intrinsics.height,
            fx=intrinsics.fx,
            fy=intrinsics.fy,
            cx=intrinsics.cx,
            cy=intrinsics.cy,
            hfov_rad=intrinsics.hfov_rad,
            vfov_rad=intrinsics.vfov_rad,
            near_clip=intrinsics.near_clip,
            far_clip=intrinsics.far_clip,
        )
    elif kind in ("lidar", "point_cloud", "range"):
        horizontal = find_fov_value(raw, _HFOV_KEYS)
        vertical = find_fov_value(raw, _VFOV_KEYS)
        if horizontal is not None:
            derived["hfov_rad"] = _rounded(horizontal[1])
        if vertical is not None:
            derived["vfov_rad"] = _rounded(vertical[1])
        min_range = _number(raw, "min_range", "range_min")
        max_range = _number(raw, "max_range", "range_max")
        if min_range is not None:
            derived["min_range"] = _rounded(min_range)
        if max_range is not None:
            derived["max_range"] = _rounded(max_range)

    consumed = {
        "update_rate",
        "fps",
        "rate_hz",
        "rate",
        "range_min",
        "range_max",
        "image_width",
        "image_height",
        "near",
        "far",
        "znear",
        "zfar",
        "clip_near",
        "clip_far",
    }
    params: dict[str, Any] = {}
    for key, value in raw.items():
        if key in derived or key in consumed or key in _FOV_KEYS:
            continue
        scalar = _scalar(value)
        if scalar is not None:
            params[str(key)] = scalar
    params.update(derived)
    return dict(sorted(params.items()))


# --------------------------------------------------------------------------
# Rotors
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RotorSpec:
    """A thrust-producing site motor (``<motor site=... gear="0 0 1 0 0 k">``).

    ``thrust_axis`` is the unit force direction in the site (= rotor prim) frame;
    ``thrust_gain`` is ``|gear[:3]|`` (newtons per unit ``ctrl``);
    ``torque_to_thrust`` is the signed reaction torque about the thrust axis per
    newton of thrust (``gear[3:] . axis / |gear[:3]|``), which is what
    cyberwave-sim's aerial allocator reads from ``gear[5]``.
    """

    actuator: str
    site: str
    link: str
    position: Vector3
    orientation: Quaternion
    thrust_axis: tuple[float, float, float]
    thrust_gain: float
    torque_to_thrust: float
    control_range: tuple[float, float] | None
    max_thrust: float | None
    notes: tuple[str, ...] = field(default=())

    def to_custom_data(self) -> dict[str, Any]:
        """Scalars and float lists only, so it authors as a plain USD dictionary."""
        data: dict[str, Any] = {
            "actuator": self.actuator,
            "site": self.site,
            "link": self.link,
            "thrustAxis": [float(v) for v in self.thrust_axis],
            "thrustGain": self.thrust_gain,
            "torqueToThrust": self.torque_to_thrust,
        }
        if self.control_range is not None:
            data["controlRange"] = [float(v) for v in self.control_range]
        if self.max_thrust is not None:
            data["maxThrust"] = self.max_thrust
        return data


def _floats(text: str | None) -> list[float] | None:
    if not text:
        return None
    try:
        values = [float(v) for v in text.split()]
    except ValueError:
        return None
    return values if all(math.isfinite(v) for v in values) else None


def site_local_pose(site: dict[str, str]) -> tuple[Vector3, Quaternion, list[str]]:
    """Return a MJCF site's ``(position, orientation, notes)`` in its link frame.

    ``quat`` (``w x y z``), ``zaxis`` and ``xyaxes`` are honoured; ``fromto`` and
    malformed values fall back to identity orientation with a note.
    """
    notes: list[str] = []
    pos = _floats(site.get("pos")) or [0.0, 0.0, 0.0]
    position = Vector3(*pos[:3]) if len(pos) >= 3 else Vector3()
    orientation = Quaternion()
    quat = _floats(site.get("quat"))
    zaxis = _floats(site.get("zaxis"))
    xyaxes = _floats(site.get("xyaxes"))
    if quat is not None and len(quat) == 4 and any(quat):
        norm = math.sqrt(sum(v * v for v in quat))
        orientation = Quaternion(x=quat[1] / norm, y=quat[2] / norm, z=quat[3] / norm, w=quat[0] / norm)
    elif zaxis is not None and len(zaxis) == 3 and any(zaxis):
        orientation = f.axis_to_ref_quat(Vector3(*zaxis), "Z")
    elif xyaxes is not None and len(xyaxes) == 6:
        x_axis = np.asarray(xyaxes[:3], dtype=float)
        y_axis = np.asarray(xyaxes[3:], dtype=float)
        x_axis /= np.linalg.norm(x_axis) or 1.0
        y_axis = y_axis - x_axis * float(np.dot(x_axis, y_axis))
        y_axis /= np.linalg.norm(y_axis) or 1.0
        z_axis = np.cross(x_axis, y_axis)
        orientation = f.matrix_to_quat(np.column_stack([x_axis, y_axis, z_axis]))
    elif site.get("fromto") or site.get("euler") or site.get("axisangle"):
        notes.append("site_orientation_unsupported")
    return position, orientation, notes


def sites_by_name(schema: CommonSchema) -> dict[str, tuple[Link, dict[str, str]]]:
    """Map every MJCF site name to ``(link, attributes)``; invalid sites skipped."""
    found: dict[str, tuple[Link, dict[str, str]]] = {}
    for link in schema.links:
        try:
            sites = link.get_mjcf_sites()
        except ValueError:
            continue
        for site in sites:
            found.setdefault(site["name"], (link, site))
    return found


def rotor_spec(actuator: Actuator, sites: dict[str, tuple[Link, dict[str, str]]]) -> RotorSpec | None:
    """Return the rotor a site-transmission actuator describes, or ``None``.

    Mirrors cyberwave-sim's structural rotor detection: a site motor whose gear
    vector has a non-zero force component is a thruster. Pure-torque site
    motors (reaction wheels) are not rotors.
    """
    try:
        transmission = actuator.get_mjcf_site_transmission()
    except ValueError:
        return None
    if transmission is None:
        return None
    gear = _floats(transmission.get("gear", "1 0 0 0 0 0")) or [1.0]
    gear = (gear + [0.0] * 6)[:6]
    force = np.asarray(gear[:3], dtype=float)
    gain = float(np.linalg.norm(force))
    if gain == 0.0:
        return None
    site_name = transmission["site"]
    located = sites.get(site_name)
    if located is None:
        return None
    link, site = located
    position, orientation, notes = site_local_pose(site)
    axis = force / gain
    torque_to_thrust = float(np.dot(np.asarray(gear[3:], dtype=float), axis)) / gain
    control_range = (
        (float(actuator.control_range[0]), float(actuator.control_range[1]))
        if actuator.control_range is not None and len(actuator.control_range) == 2
        else None
    )
    max_thrust = _rounded(control_range[1] * gain) if control_range is not None and control_range[1] > 0 else None
    if transmission.get("refsite"):
        notes.append("rotor_refsite_ignored")
    return RotorSpec(
        actuator=actuator.name,
        site=site_name,
        link=link.name,
        position=position,
        orientation=orientation,
        thrust_axis=(_rounded(axis[0]), _rounded(axis[1]), _rounded(axis[2])),
        thrust_gain=_rounded(gain),
        torque_to_thrust=_rounded(torque_to_thrust),
        control_range=control_range,
        max_thrust=max_thrust,
        notes=tuple(notes),
    )


# --------------------------------------------------------------------------
# Joints and control
# --------------------------------------------------------------------------

DOF_JOINT_TYPES = (JointType.REVOLUTE, JointType.CONTINUOUS, JointType.PRISMATIC)
"""Single-DOF joints that get a canonical index in the robot's joint vector."""

CONTROL_MODE_FOR_ACTUATOR: dict[ActuatorType, str] = {
    ActuatorType.POSITION: "position",
    ActuatorType.SERVO: "position",
    ActuatorType.VELOCITY: "velocity",
    ActuatorType.TORQUE: "effort",
    ActuatorType.DC_MOTOR: "effort",
    ActuatorType.MUSCLE: "effort",
}
"""How the runtime commands a joint driven by each actuator type.

Mirrors cyberwave-sim's actuator kinds (position servo, velocity servo, motor);
``ADHESION`` has no joint and so no mode.
"""


def dof_indices(schema: CommonSchema) -> dict[str, int]:
    """Canonical DOF index per 1-DOF joint, in schema order.

    This is the Cyberwave joint-vector order. It is *not* a promise about a
    physics engine's internal DOF order (PhysX orders by articulation tree);
    runtimes must bind by USD path or joint name and permute.
    """
    indices: dict[str, int] = {}
    for joint in schema.joints:
        if joint.type in DOF_JOINT_TYPES:
            indices[joint.name] = len(indices)
    return indices


def primary_actuators(schema: CommonSchema) -> dict[str, Actuator]:
    """The first actuator per joint — the one that owns the USD drive."""
    primary: dict[str, Actuator] = {}
    for actuator in schema.actuators:
        if actuator.joint and actuator.joint not in primary:
            primary[actuator.joint] = actuator
    return primary


def angular_gain_to_usd(value: float, is_angular: bool) -> float:
    """Convert a per-radian drive gain to UsdPhysics' per-degree units.

    ``UsdPhysicsDriveAPI`` declares angular stiffness in torque/degree and angular
    damping in torque/(degree/s), so ``kp = 100 N m/rad`` authors as
    ``100 * pi / 180``. Linear gains are already SI.
    """
    return float(value) * math.pi / 180.0 if is_angular else float(value)


def angular_gain_from_usd(value: float, is_angular: bool) -> float:
    """Inverse of :func:`angular_gain_to_usd`."""
    return float(value) * 180.0 / math.pi if is_angular else float(value)
