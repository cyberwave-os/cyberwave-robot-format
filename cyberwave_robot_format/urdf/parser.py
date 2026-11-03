# Copyright [2021-2025] Thanh Nguyen
# Copyright [2022-2023] [CNRS, Toward SAS]

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""URDF parser for converting URDF files to common schema."""

from __future__ import annotations

import hashlib
import logging
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from cyberwave_robot_format.core import BaseParser, ParseContext, ParseError
from cyberwave_robot_format.fov_convention import find_fov_value, normalize_fov_key
from cyberwave_robot_format.mesh import resolve_mesh_uri
from cyberwave_robot_format.schema import (
    Collision,
    CollisionConfig,
    CollisionGroup,
    CommonSchema,
    Geometry,
    GeometryType,
    Inertia,
    Joint,
    JointDynamics,
    JointLimits,
    JointType,
    Link,
    Material,
    Metadata,
    MimicJoint,
    Pose,
    Quaternion,
    Sensor,
    Vector3,
    Visual,
)
from cyberwave_robot_format.urdf.conventions import (
    AXIS_JOINT_TYPES,
    URDF_DEFAULT_AXIS,
)
from cyberwave_robot_format.urdf.passthrough import (
    EXTENSION_KEY as PASSTHROUGH_EXTENSION_KEY,
)
from cyberwave_robot_format.urdf.passthrough import capture_urdf_passthrough
from cyberwave_robot_format.usd.semantics import CAMERA_KINDS, sensor_kind
from cyberwave_robot_format.utils import sanitize_name

logger = logging.getLogger(__name__)

# ``<axis>`` present but carrying no ``xyz``: URDF defaults it exactly as it defaults a
# missing ``<axis>`` element.
_URDF_DEFAULT_AXIS_ATTR = " ".join(str(component) for component in URDF_DEFAULT_AXIS)

# Sensor types whose parameters live in a Gazebo ``<ray>`` block. "ray" is
# Gazebo's own spelling and is resolved to lidar_2d/lidar_3d from the scan block;
# the concrete names are accepted too so a URDF may state the type outright.
_RAY_SENSOR_TYPES = frozenset(
    {"ray", "gpu_ray", "lidar", "lidar_2d", "lidar_3d", "rangefinder"}
)

# Sensor types whose frame looks along +X, as a Gazebo camera's does: every Gazebo/SDF
# camera type, whether or not Cyberwave renders it as a camera today. Their ``<pose>``
# rotation is in the Gazebo camera convention, not the schema's (-Z forward), so only
# the mount position is kept. Keyed off Gazebo's types rather than ``CAMERA_KINDS``
# on purpose: a type with no camera kind yet (``thermal_camera``, ``multicamera``,
# ...) would otherwise store a wrong-convention rotation that a future renderer reads.
# Types that map to a camera kind are covered too, see ``_is_gazebo_camera_type``.
_GAZEBO_CAMERA_SENSOR_TYPES = frozenset(
    {
        "camera",
        "depth",
        "depth_camera",
        "rgbd",
        "rgbd_camera",
        "multicamera",
        "wideanglecamera",
        "wide_angle_camera",
        "thermal_camera",
        "thermal",
        # gz-sim's segmentation, bounding-box and logical cameras. None has a camera
        # kind, but each frame looks along +X like any Gazebo camera. ``gpu_lidar``
        # is not listed: cyberwave-sim casts its beams along +X too, so it keeps
        # its rotation.
        "segmentation_camera",
        "boundingbox_camera",
        "logical_camera",
    }
)


def _is_gazebo_camera_type(sensor_type: str) -> bool:
    """Whether a Gazebo ``<sensor type>`` names a frame that looks along +X."""
    return (
        sensor_type.strip().lower() in _GAZEBO_CAMERA_SENSOR_TYPES
        or sensor_kind(sensor_type) in CAMERA_KINDS
    )


# ``<safety_controller>`` and ``<calibration>`` attributes, per the URDF spec. Both
# are stored as plain ``dict[str, float]`` on ``Joint``, so only the spec'd keys are
# read — an unknown attribute would travel into a typed field with no consumer.
_SAFETY_CONTROLLER_ATTRIBUTES = (
    "soft_lower_limit",
    "soft_upper_limit",
    "k_position",
    "k_velocity",
)
_CALIBRATION_ATTRIBUTES = ("rising", "falling")


def _finite_float(raw: str) -> float:
    """``float(raw)``, raising ``ValueError`` for NaN/Infinity as for non-numeric text."""
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"expected a finite number, got {raw!r}")
    return value


def _content_name(elem: ET.Element) -> str | None:
    """A name for a nameless ``<material>``, derived from its colour and texture."""
    color = elem.find("color")
    texture = elem.find("texture")
    content = (
        " ".join((color.get("rgba") or "").split()) if color is not None else "",
        (texture.get("filename") or "") if texture is not None else "",
    )
    if not any(content):
        return None
    return "material_" + hashlib.sha1("|".join(content).encode()).hexdigest()[:8]


class URDFParser(BaseParser):
    """URDF parser with enhanced validation and error handling."""

    def __init__(self) -> None:
        super().__init__()
        self.supported_versions = ["1.0"]

    def can_parse(self, file_path: str | Path) -> bool:
        """Check if file is a valid URDF file with enhanced validation."""
        try:
            parser = ET.XMLParser(encoding="utf-8")
            namespaces = {
                "sensor": "http://playerstage.sourceforge.net/gazebo/xmlschema/#sensor",
                "controller": "http://playerstage.sourceforge.net/gazebo/xmlschema/#controller",
                "interface": "http://playerstage.sourceforge.net/gazebo/xmlschema/#interface",
                "xacro": "http://www.ros.org/wiki/xacro",
            }
            for prefix, uri in namespaces.items():
                try:
                    ET.register_namespace(prefix, uri)
                except AttributeError:
                    pass

            try:
                tree = ET.parse(file_path, parser=parser)
            except ET.ParseError:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                content = re.sub(r"sensor:camera", "sensor_camera", content)
                content = re.sub(r"sensor:ray", "sensor_ray", content)
                root = ET.fromstring(content, parser=parser)
            else:
                root = tree.getroot()

            if root.tag != "robot":
                return False

            if not root.get("name"):
                logger.warning("URDF file %s missing robot name", file_path)

            links = root.findall("link")
            if not links:
                logger.warning("URDF file %s has no links", file_path)
                return False

            return True

        except ET.ParseError as e:
            logger.error("XML parsing error in %s: %s", file_path, e)
            return False
        except Exception as e:
            logger.error("Error checking URDF file %s: %s", file_path, e)
            return False

    def parse(self, input_path: str | Path) -> CommonSchema:
        """Parse URDF file with comprehensive validation and error handling."""
        file_path = Path(input_path)

        try:
            parser = ET.XMLParser(encoding="utf-8")

            try:
                tree = ET.parse(file_path, parser=parser)
                root = tree.getroot()
            except ET.ParseError as e:
                logger.warning(
                    "Standard parse failed (%s), attempting to clean XML namespaces...",
                    e,
                )
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                content = re.sub(r"<(\/?)(\w+):(\w+)", r"<\1\2_\3", content)

                try:
                    retry_parser = ET.XMLParser(encoding="utf-8")
                    root = ET.fromstring(content, parser=retry_parser)
                except ET.ParseError as e2:
                    prefixes = set(re.findall(r"<(\w+):", content))
                    if prefixes:
                        logger.info("Found potential prefixes: %s", prefixes)
                        for p in prefixes:
                            try:
                                ET.register_namespace(p, "http://dummy.url")
                            except Exception:
                                pass
                        raise ParseError(
                            f"XML parsing error after cleanup: {e2}"
                        ) from e2
                    raise ParseError(f"XML parsing error: {e2}") from e2

        except ET.ParseError as e:
            raise ParseError(f"XML parsing error: {e}") from e

        context = ParseContext(
            file_path=file_path,
            base_dir=file_path.parent,
        )

        robot_name = (root.get("name") or "").strip()
        if not robot_name:
            # Vendor URDFs omit or blank the root name. Fall back to the file
            # stem: stable, and distinctive where the old "robot" default
            # collided across every unnamed model.
            robot_name = sanitize_name(file_path.stem)
            context.add_warning(f"Robot name missing or empty; using {robot_name!r}")

        metadata = Metadata(
            name=sanitize_name(robot_name),
            source_format="urdf",
            description=f"Converted from URDF: {file_path.name}",
            version=root.get("version", "1.0"),
        )

        document_materials: list[Material] = []
        for material_elem in root.findall("material"):
            material = self._parse_material(material_elem, context)
            if material and material.name:
                context.materials[material.name] = material
                document_materials.append(material)

        links = []
        for link_elem in root.findall("link"):
            try:
                link = self._parse_link(link_elem, context)
                if link:
                    if link.name == "world":
                        logger.debug(
                            "Skipping 'world' link from URDF '%s' — asset schemas must not "
                            "contain a world link; the environment schema provides it.",
                            file_path.name,
                        )
                        continue
                    links.append(link)
            except Exception as e:
                context.add_error(f"Failed to parse link: {e}", link_elem.get("name"))

        joints = []
        link_names = {link.name for link in links}

        for joint_elem in root.findall("joint"):
            try:
                joint = self._parse_joint(joint_elem, context, link_names)
                if joint:
                    joints.append(joint)
            except Exception as e:
                context.add_error(f"Failed to parse joint: {e}", joint_elem.get("name"))

        self._validate_kinematic_tree(links, joints, context)

        if context.warnings:
            logger.info("Parsing completed with %d warnings", len(context.warnings))
        if context.errors:
            logger.warning("Parsing completed with %d errors", len(context.errors))

        sensors = self._parse_urdf_sensors(root, context)

        # Initialize collision_config with default group
        collision_config = CollisionConfig(
            groups={"default": CollisionGroup(name="default", contype=1, conaffinity=1)}
        )

        schema = CommonSchema(
            metadata=metadata,
            collision_config=collision_config,
            materials=document_materials,
            links=links,
            joints=joints,
            sensors=sensors,
        )

        schema.extensions["parse_context"] = {
            "warnings": context.warnings,
            "errors": context.errors,
            "materials": context.materials,
        }

        # Everything the schema cannot type — <ros2_control>, <transmission>,
        # <gazebo> plugins, unknown/namespaced elements — kept verbatim so the
        # exporter can re-emit it. Without this a schema -> URDF export deletes a
        # robot's whole control configuration. See :mod:`.passthrough`.
        passthrough = capture_urdf_passthrough(root)
        if passthrough:
            schema.extensions[PASSTHROUGH_EXTENSION_KEY] = passthrough

        return schema

    def _parse_urdf_sensors(
        self, root: ET.Element, context: ParseContext
    ) -> list[Sensor]:
        """Parse sensors from URDF (Gazebo extensions)."""
        sensors = []

        for gazebo_elem in root.findall("gazebo"):
            reference = gazebo_elem.get("reference")
            ref_str = reference or ""

            for sensor_elem in gazebo_elem.findall("sensor"):
                self._parse_gazebo_sensor(sensor_elem, ref_str, sensors, context)

            for cam_elem in gazebo_elem.findall("sensor_camera"):
                self._parse_gazebo_camera_direct(cam_elem, ref_str, sensors, context)

        return sensors

    def _parse_link(self, elem: ET.Element, context: ParseContext) -> Link | None:
        """Parse URDF link element with enhanced validation."""
        name = elem.get("name")
        if not name:
            context.add_error("Link name is required")
            return None

        name = sanitize_name(name)
        link = Link(name=name)

        inertial = elem.find("inertial")
        if inertial is not None:
            try:
                self._parse_inertial(inertial, link, context)
            except Exception as e:
                context.add_warning(f"Failed to parse inertial properties: {e}", name)

        for visual_elem in elem.findall("visual"):
            try:
                visual = self._parse_visual(visual_elem, context)
                if visual:
                    link.visuals.append(visual)
            except Exception as e:
                context.add_warning(f"Failed to parse visual element: {e}", name)

        for collision_elem in elem.findall("collision"):
            try:
                collision = self._parse_collision(collision_elem, context)
                if collision:
                    link.collisions.append(collision)
            except Exception as e:
                context.add_warning(f"Failed to parse collision element: {e}", name)

        return link

    def _parse_sensor_pose(
        self, sensor_elem: ET.Element, sensor_name: str, context: ParseContext
    ) -> Pose:
        """Mount pose of a Gazebo sensor, relative to its reference link.

        This was never read, so ``Sensor.pose`` stayed at the identity default for every
        URDF-sourced sensor — while the MJCF exporter *writes* that field into the
        ``<site>`` a sensor is sampled at, and ``cyberwave-sim/sensors/range.py`` depends
        on it ("a MiR250 carries two scanners on opposite corners, each yawed 45°"). Both
        scanners therefore ended up at the chassis centre pointing the same way, while
        Gazebo still saw the right pose via the ``<gazebo>`` passthrough.

        SDF spells it ``<pose>x y z roll pitch yaw</pose>``, with ``degrees="true"``
        switching the angles to degrees and ``rotation_format="quat_xyzw"`` taking
        ``x y z qx qy qz qw`` instead; both are honoured. A URDF-style
        ``<origin xyz= rpy=>`` is accepted too, because hand-written descriptions mix
        the two. A pose that cannot be read falls back to the identity with a warning
        rather than silently.
        """
        pose_elem = sensor_elem.find("pose")
        if pose_elem is not None and (pose_elem.text or "").strip():
            pose = self._parse_sdf_pose(pose_elem)
            if pose is not None:
                return pose
            context.add_warning(
                f"Unreadable sensor <pose>{(pose_elem.text or '').strip()}</pose> "
                f"(attributes {dict(pose_elem.attrib)}); using the identity",
                sensor_name,
            )
            return Pose()

        origin_elem = sensor_elem.find("origin")
        if origin_elem is not None:
            xyz = self._parse_xyz(origin_elem.get("xyz", "0 0 0"))
            rpy = self._parse_rpy(origin_elem.get("rpy", "0 0 0"))
            if xyz and rpy:
                return Pose.from_xyzrpy(xyz, rpy)
            context.add_warning(
                f"Unreadable sensor <origin> {dict(origin_elem.attrib)}; using the identity",
                sensor_name,
            )

        return Pose()

    @staticmethod
    def _parse_sdf_pose(pose_elem: ET.Element) -> Pose | None:
        """An SDF ``<pose>`` as a :class:`Pose`, or ``None`` if it cannot be read.

        Honours ``rotation_format`` (``euler_rpy``, the default, or ``quat_xyzw``) and
        ``degrees`` (Euler angles only, as in SDF). Wrong value counts, non-numeric or
        non-finite values and a zero quaternion are all unreadable.
        """
        rotation_format = pose_elem.get("rotation_format", "euler_rpy").strip().lower()
        try:
            values = [_finite_float(part) for part in (pose_elem.text or "").split()]
        except ValueError:
            return None

        if rotation_format == "euler_rpy" and len(values) == 6:
            rpy = values[3:6]
            if pose_elem.get("degrees", "false").strip().lower() in ("true", "1"):
                rpy = [math.radians(angle) for angle in rpy]
            return Pose.from_xyzrpy(values[0:3], rpy)

        if rotation_format == "quat_xyzw" and len(values) == 7:
            qx, qy, qz, qw = values[3:7]
            norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
            if norm < 1e-12:
                return None
            return Pose(
                position=Vector3.from_list(values[0:3]),
                orientation=Quaternion(
                    x=qx / norm, y=qy / norm, z=qz / norm, w=qw / norm
                ),
            )

        return None

    def _camera_mount_pose(
        self, pose: Pose, sensor_name: str, context: ParseContext
    ) -> Pose:
        """Keep a Gazebo camera's mount position but not its rotation.

        A Gazebo camera looks along +X of its pose frame, while ``CommonSchema``
        camera poses use the MuJoCo/USD convention (look along -Z, +Y up). Copying
        the Gazebo rotation verbatim aims the exported camera somewhere unrelated: one
        pitched straight down in Gazebo came out looking backwards. The rotation is
        dropped, as it was before sensor poses were read, and reported.
        """
        q = pose.orientation
        if not (
            abs(abs(q.w) - 1.0) < 1e-9
            and abs(q.x) < 1e-9
            and abs(q.y) < 1e-9
            and abs(q.z) < 1e-9
        ):
            context.add_warning(
                "Gazebo camera <pose> rotation not applied: Gazebo cameras look "
                "along +X, schema cameras along -Z",
                sensor_name,
            )
        return Pose(position=pose.position)

    def _parse_gazebo_sensor(
        self,
        sensor_elem: ET.Element,
        reference: str,
        sensors: list[Sensor],
        context: ParseContext,
    ) -> None:
        """Parse standard Gazebo sensor element."""
        name = sensor_elem.get("name")
        sensor_type = sensor_elem.get("type")

        if not name or not sensor_type:
            return

        parent_link = sanitize_name(reference) if reference else "world"

        pose = self._parse_sensor_pose(sensor_elem, name, context)
        if _is_gazebo_camera_type(sensor_type):
            pose = self._camera_mount_pose(pose, name, context)

        sensor = Sensor(
            name=sanitize_name(name),
            type=sensor_type,
            parent_link=parent_link,
            pose=pose,
        )

        update_rate = sensor_elem.find("update_rate")
        if update_rate is not None and update_rate.text:
            try:
                sensor.update_rate = float(update_rate.text)
            except ValueError:
                pass

        if sensor_type == "camera":
            camera_elem = sensor_elem.find("camera")
            if camera_elem is not None:
                self._extract_camera_properties(camera_elem, sensor)
        elif sensor_type in _RAY_SENSOR_TYPES:
            ray_elem = sensor_elem.find("ray")
            if ray_elem is not None:
                self._extract_ray_properties(ray_elem, sensor)
            # Gazebo spells every scanning range sensor "ray" regardless of
            # whether it sweeps in one plane or two. Resolve it to the concrete
            # type using the scan block, per the Gazebo/SDF spec: a <vertical>
            # scan means a 3D sweep, horizontal-only means a planar 2D scanner.
            # Left verbatim, "ray" reaches consumers as an unrecognized type that
            # range rendering and the sensor-type union both ignore.
            if sensor_type == "ray":
                scan = ray_elem.find("scan") if ray_elem is not None else None
                vertical = scan.find("vertical") if scan is not None else None
                sensor.type = "lidar_2d" if vertical is None else "lidar_3d"

        sensors.append(sensor)

    def _extract_ray_properties(self, elem: ET.Element, sensor: Sensor) -> None:
        """Extract lidar/rangefinder properties from a Gazebo ``<ray>`` block.

        Angles are radians in the URDF and degrees in the schema. ``samples`` is
        per-scan, so the platform's ``points_per_second`` needs the sensor's
        update rate applied.

        Only fields the schema actually models are emitted. ``<noise>`` is
        skipped: ``Sensor.noise`` is typed ``dict[str, float | list[float]]``,
        for which no structure hook is registered, so populating it makes
        ``CommonSchema.from_dict`` raise.
        """

        def _f(path: str) -> float | None:
            node = elem.find(path)
            if node is None or not node.text:
                return None
            try:
                return float(node.text)
            except ValueError:
                return None

        samples_total = 1.0
        for axis, fov_key, fov_rad_key in (
            ("horizontal", "fov_horizontal", "hfov_rad"),
            ("vertical", "fov_vertical", "vfov_rad"),
        ):
            min_angle = _f(f"scan/{axis}/min_angle")
            max_angle = _f(f"scan/{axis}/max_angle")
            if min_angle is not None and max_angle is not None:
                sweep_rad = abs(max_angle - min_angle)
                sensor.parameters[fov_key] = math.degrees(sweep_rad)
                # Only a real sweep gets a canonical key. A degenerate axis
                # (SDF's default <vertical> is samples=1, min_angle==max_angle)
                # would otherwise plant a 0.0 that the write gate rejects as
                # out of range and that fov_canonicalize cannot purge --
                # canonical keys are excluded from LEGACY_FOV_KEYS, so the
                # purge loop never reaches them.
                if sweep_rad > 0:
                    sensor.parameters[fov_rad_key] = sweep_rad
            samples = _f(f"scan/{axis}/samples")
            if samples is not None and samples > 0:
                samples_total *= samples
                sensor.parameters[f"{axis}_samples"] = int(samples)
            if min_angle is not None and max_angle is not None:
                # Preserve asymmetric sweeps; a FOV alone loses their centre.
                sensor.parameters[
                    "yaw_offset" if axis == "horizontal" else "pitch_offset"
                ] = (min_angle + max_angle) / 2.0

        min_range = _f("range/min")
        if min_range is not None:
            sensor.parameters["min_range"] = min_range
        max_range = _f("range/max")
        if max_range is not None:
            sensor.parameters["max_range"] = max_range

        # samples are per scan; the schema wants a rate.
        if samples_total > 1.0 and sensor.update_rate:
            sensor.parameters["points_per_second"] = int(
                samples_total * sensor.update_rate
            )

    def _parse_gazebo_camera_direct(
        self,
        cam_elem: ET.Element,
        reference: str,
        sensors: list[Sensor],
        context: ParseContext,
    ) -> None:
        """Parse direct sensor_camera element (cleaned from sensor:camera)."""
        name = cam_elem.get("name")
        if not name:
            name = f"camera_{len(sensors)}"

        parent_link = sanitize_name(reference) if reference else "world"

        sensor = Sensor(
            name=sanitize_name(name),
            type="camera",
            parent_link=parent_link,
            pose=self._camera_mount_pose(
                self._parse_sensor_pose(cam_elem, name, context), name, context
            ),
        )

        self._extract_camera_properties(cam_elem, sensor)

        sensors.append(sensor)

    # Vertical spellings first: a direct declaration outranks one that has
    # to be derived via an aspect-ratio assumption.
    _CAMERA_FOV_ELEMENT_NAMES: tuple[str, ...] = (
        "vfov",
        "fovy",
        "fov_vertical",
        "vertical_fov",
        "vertical_fov_degrees",
        "fov_degrees",
        "hfov",
        "horizontal_fov",
        "fov_horizontal",
        "horizontal_fov_degrees",
    )

    def _find_declared_camera_fov(self, elem: ET.Element) -> tuple[str, float] | None:
        """Resolve the highest-priority recognized FOV element to a
        canonical ``(key, radians)`` pair, or ``None`` if none is present.
        Logs a warning instead of silently picking one whenever more than
        one recognized element disagrees or spans both axes."""
        extracted: dict[str, float] = {}
        for tag in self._CAMERA_FOV_ELEMENT_NAMES:
            node = elem.find(tag)
            if node is None or not node.text:
                continue
            try:
                raw_value = float(node.text)
            except ValueError:
                continue
            if not math.isfinite(raw_value):
                continue
            extracted[tag] = raw_value

        if not extracted:
            return None

        resolved_by_canonical: dict[str, list[tuple[str, float]]] = {}
        for tag, raw_value in extracted.items():
            resolved = normalize_fov_key(tag, raw_value)
            if resolved is None:
                continue
            canonical_key, radians = resolved
            resolved_by_canonical.setdefault(canonical_key, []).append((tag, radians))

        if len(resolved_by_canonical) > 1:
            logger.warning(
                "Camera %r declares FOV on more than one axis (%s); the "
                "highest-priority value is used, the rest are discarded.",
                elem,
                resolved_by_canonical,
            )
        for canonical_key, entries in resolved_by_canonical.items():
            distinct_values = {round(radians, 9) for _, radians in entries}
            if len(distinct_values) > 1:
                logger.warning(
                    "Camera %r declares conflicting values for %s: %s",
                    elem,
                    canonical_key,
                    entries,
                )

        return find_fov_value(extracted, self._CAMERA_FOV_ELEMENT_NAMES)

    def _extract_camera_properties(self, elem: ET.Element, sensor: Sensor) -> None:
        """Extract camera properties like fov, image size from element.

        FOV is resolved through ``fov_convention`` regardless of spelling or
        axis. A declared horizontal value derives vertical via the image
        aspect ratio; a declared vertical value is stored directly. Only
        ``vfov_rad`` is ever written — a camera never stores horizontal.
        """
        declared = self._find_declared_camera_fov(elem)

        image = elem.find("image")
        if image is not None:
            width = image.find("width")
            height = image.find("height")

            if width is not None and width.text:
                sensor.parameters["width"] = int(width.text)
            if height is not None and height.text:
                sensor.parameters["height"] = int(height.text)

        if declared is not None:
            canonical_key, value_rad = declared
            if canonical_key == "vfov_rad":
                sensor.parameters["vfov_rad"] = value_rad
            elif canonical_key == "hfov_rad":
                width_px = sensor.parameters.get("width")
                height_px = sensor.parameters.get("height")
                if (
                    isinstance(width_px, int)
                    and isinstance(height_px, int)
                    and width_px > 0
                    and height_px > 0
                ):
                    fov_v_rad = 2.0 * math.atan(
                        math.tan(value_rad / 2.0) * (height_px / width_px)
                    )
                    sensor.parameters["vfov_rad"] = fov_v_rad

        img_size = elem.find("imageSize")
        if img_size is not None and img_size.text:
            try:
                parts = img_size.text.split()
                if len(parts) >= 2:
                    sensor.parameters["width"] = int(parts[0])
                    sensor.parameters["height"] = int(parts[1])
            except ValueError:
                pass

    def _parse_inertial(
        self, elem: ET.Element, link: Link, context: ParseContext
    ) -> None:
        """Parse inertial properties with enhanced validation."""
        mass_elem = elem.find("mass")
        if mass_elem is not None:
            try:
                mass = float(mass_elem.get("value", 0.0))
                if mass < 0:
                    context.add_warning("Negative mass detected", link.name)
                link.mass = mass
            except ValueError:
                context.add_error("Invalid mass value", link.name)

        origin = elem.find("origin")
        if origin is not None:
            xyz = self._parse_xyz(origin.get("xyz", "0 0 0"))
            if xyz:
                link.center_of_mass = Vector3(xyz[0], xyz[1], xyz[2])
            rpy = self._parse_rpy(origin.get("rpy", "0 0 0"))
            if rpy:
                link.inertial_orientation = Quaternion.from_rpy(rpy[0], rpy[1], rpy[2])

        inertia_elem = elem.find("inertia")
        if inertia_elem is not None:
            try:
                inertia = Inertia(
                    ixx=float(inertia_elem.get("ixx", 0.0)),
                    iyy=float(inertia_elem.get("iyy", 0.0)),
                    izz=float(inertia_elem.get("izz", 0.0)),
                    ixy=float(inertia_elem.get("ixy", 0.0)),
                    ixz=float(inertia_elem.get("ixz", 0.0)),
                    iyz=float(inertia_elem.get("iyz", 0.0)),
                )

                if not self._validate_inertia(inertia):
                    context.add_warning("Inertia tensor may be invalid", link.name)

                link.inertia = inertia

            except ValueError as e:
                context.add_error(f"Invalid inertia values: {e}", link.name)

    def _parse_float_attributes(
        self,
        elem: ET.Element | None,
        attributes: tuple[str, ...],
        context: ParseContext,
        joint_name: str,
    ) -> dict[str, float] | None:
        """Read ``attributes`` off ``elem`` as finite floats, or ``None`` when absent.

        Returns ``None`` rather than ``{}`` for a missing element so the schema
        distinguishes "not authored" from "authored empty" — the backend's
        rebuild-merge treats a falsy section as a gap to fill.

        Non-finite values are dropped with the same warning as non-numeric ones: a
        NaN soft limit disables the safety controller it was meant to configure.
        """
        if elem is None:
            return None
        parsed: dict[str, float] = {}
        for attribute in attributes:
            raw = elem.get(attribute)
            if raw is None:
                continue
            try:
                parsed[attribute] = _finite_float(raw)
            except ValueError:
                context.add_warning(
                    f"Ignoring non-numeric or non-finite <{elem.tag} {attribute}={raw!r}>",
                    joint_name,
                )
        return parsed or None

    def _validate_inertia(self, inertia: Inertia) -> bool:
        """Enhanced validation of inertia tensor for physical plausibility."""
        if inertia.ixx <= 0 or inertia.iyy <= 0 or inertia.izz <= 0:
            return False

        if (
            inertia.ixx + inertia.iyy <= inertia.izz
            or inertia.iyy + inertia.izz <= inertia.ixx
            or inertia.ixx + inertia.izz <= inertia.iyy
        ):
            return False

        return True

    def _parse_joint(
        self,
        elem: ET.Element,
        context: ParseContext,
        link_names: set,
    ) -> Joint | None:
        """Parse URDF joint element with enhanced validation."""
        name = elem.get("name")
        raw_joint_type = elem.get("type") or ""
        joint_type = raw_joint_type.strip().lower()

        if not name:
            context.add_error("Joint name is required")
            return None

        if not joint_type:
            # Dropping the joint detaches its child link from the tree, so keep
            # it and assume the rigid reading. Where the file declares a child
            # only a moving joint can use, report the discarded DOF rather than
            # just warn — the spec cannot say which moving type was meant.
            # <dynamics> is excluded: real files carry it on fixed joints.
            joint_type = "fixed"
            declared = [
                child
                for child in ("axis", "limit", "mimic", "safety_controller")
                if elem.find(child) is not None
            ]
            if declared:
                spelled = "/".join(f"<{child}>" for child in declared)
                context.add_error(
                    f"Joint type is missing but the file declares {spelled}, "
                    "which only a moving joint can use; keeping it 'fixed' and "
                    "discarding that motion",
                    name,
                )
            else:
                context.add_warning("Joint type is missing; assuming 'fixed'", name)

        type_mapping = {
            "revolute": JointType.REVOLUTE,
            "continuous": JointType.CONTINUOUS,
            "prismatic": JointType.PRISMATIC,
            "fixed": JointType.FIXED,
            "floating": JointType.FLOATING,
            "planar": JointType.PLANAR,
        }

        joint_type_enum = type_mapping.get(joint_type)
        if not joint_type_enum:
            context.add_error(f"Unsupported joint type: {raw_joint_type!r}.", name)
            return None

        if raw_joint_type.strip() and raw_joint_type != joint_type:
            # The README promises every tolerated quirk lands in parse_context;
            # a silent repair is one the caller cannot audit.
            context.add_warning(
                f"Joint type {raw_joint_type!r} normalized to {joint_type!r}", name
            )

        parent_elem = elem.find("parent")
        child_elem = elem.find("child")

        if parent_elem is None or child_elem is None:
            context.add_error("Joint must have parent and child links.", name)
            return None

        parent_link = sanitize_name(parent_elem.get("link", ""))
        child_link = sanitize_name(child_elem.get("link", ""))

        if parent_link not in link_names and parent_link != "world":
            context.add_error(f"Parent link {parent_link!r} not found.", name)
        if child_link not in link_names:
            context.add_error(f"Child link {child_link!r} not found.", name)

        joint = Joint(
            name=sanitize_name(name),
            type=joint_type_enum,
            parent_link=parent_link,
            child_link=child_link,
        )

        origin = elem.find("origin")
        if origin is not None:
            xyz = self._parse_xyz(origin.get("xyz", "0 0 0"))
            rpy = self._parse_rpy(origin.get("rpy", "0 0 0"))
            if xyz and rpy:
                joint.pose = Pose.from_xyzrpy(xyz, rpy)

        # A joint that moves always gets an explicit axis, because the exporter always
        # writes one (see ``AXIS_JOINT_TYPES``). Leaving the ``Joint`` dataclass default
        # of ``(0, 0, 1)`` in place for a document that omitted ``<axis>`` silently
        # rotated the joint off spec, and the next schema->URDF render then wrote that
        # wrong axis out explicitly, making it permanent.
        axis = elem.find("axis")
        if axis is None:
            if joint_type_enum.value in AXIS_JOINT_TYPES:
                joint.axis = Vector3(*URDF_DEFAULT_AXIS)
        else:
            axis_xyz = self._parse_xyz(axis.get("xyz", _URDF_DEFAULT_AXIS_ATTR))
            if axis_xyz:
                axis_vec = Vector3(axis_xyz[0], axis_xyz[1], axis_xyz[2])
                magnitude = math.sqrt(axis_vec.x**2 + axis_vec.y**2 + axis_vec.z**2)
                if magnitude > 1e-6:
                    joint.axis = Vector3(
                        axis_vec.x / magnitude,
                        axis_vec.y / magnitude,
                        axis_vec.z / magnitude,
                    )
                else:
                    context.add_warning("Zero-magnitude joint axis", name)
                    joint.axis = Vector3(*URDF_DEFAULT_AXIS)
            elif joint_type_enum.value in AXIS_JOINT_TYPES:
                # ``<axis>`` present but unparseable (non-numeric, wrong arity): same
                # spec default as a missing ``<axis>``, not the dataclass ``(0, 0, 1)``.
                context.add_warning("Malformed joint axis", name)
                joint.axis = Vector3(*URDF_DEFAULT_AXIS)

        limit = elem.find("limit")
        if limit is not None:
            try:
                lower = self._parse_float(limit.get("lower"))
                upper = self._parse_float(limit.get("upper"))
                effort = self._parse_float(limit.get("effort"))
                velocity = self._parse_float(limit.get("velocity"))

                if lower is not None and upper is not None and lower > upper:
                    context.add_warning("Lower limit greater than upper limit", name)

                joint.limits = JointLimits(
                    lower=lower,
                    upper=upper,
                    effort=effort,
                    velocity=velocity,
                )
            except ValueError as e:
                context.add_error(f"Invalid limit values: {e}", name)

        dynamics = elem.find("dynamics")
        if dynamics is not None:
            try:
                joint.dynamics = JointDynamics(
                    damping=float(dynamics.get("damping", 0.0)),
                    friction=float(dynamics.get("friction", 0.0)),
                )
            except ValueError as e:
                context.add_error(f"Invalid dynamics values: {e}", name)

        # ``safety_controller`` and ``calibration`` have had fields on ``Joint``
        # all along but were never read here, so the only thing keeping a
        # hand-authored value alive was the backend's rebuild-merge carry list.
        joint.safety_controller = self._parse_float_attributes(
            elem.find("safety_controller"), _SAFETY_CONTROLLER_ATTRIBUTES, context, name
        )
        joint.calibration = self._parse_float_attributes(
            elem.find("calibration"), _CALIBRATION_ATTRIBUTES, context, name
        )

        mimic = elem.find("mimic")
        if mimic is not None:
            raw_driver = mimic.get("joint")
            # Sanitized exactly as joint *names* are above. Without this, a URDF whose
            # joints are named "1".."6" parses to joints "_1".."_6" while the mimic
            # still points at "5" — a reference that matches nothing, so every consumer
            # drops the coupling.
            driver = sanitize_name(raw_driver) if raw_driver else None
            if driver and driver == joint.name:
                # A joint that mimics itself yields a degenerate coupling
                # (slave == its own driver). Drop it rather than propagate a
                # self-referential MimicJoint into the schema/MJCF equality.
                context.add_warning(
                    "Ignoring self-referential <mimic> (joint mimics itself)",
                    name,
                )
            elif driver:
                try:
                    # ``float`` accepts "nan"/"inf", and a non-finite coefficient makes
                    # every derived position NaN; drop the coupling instead.
                    multiplier = _finite_float(mimic.get("multiplier", "1"))
                    offset = _finite_float(mimic.get("offset", "0"))
                except ValueError as e:
                    context.add_error(f"Invalid mimic values: {e}", name)
                else:
                    joint.mimic = MimicJoint(
                        joint=driver, multiplier=multiplier, offset=offset
                    )

        return joint

    def _validate_kinematic_tree(
        self, links: list[Link], joints: list[Joint], context: ParseContext
    ) -> None:
        """Enhanced kinematic tree validation."""
        link_names = {link.name for link in links}

        parent_child_map = {joint.child_link: joint.parent_link for joint in joints}

        def has_cycle(node: str, visited: set, rec_stack: set) -> bool:
            visited.add(node)
            rec_stack.add(node)

            if node in parent_child_map:
                neighbor = parent_child_map[node]
                if neighbor != "world":
                    if neighbor not in visited:
                        if has_cycle(neighbor, visited, rec_stack):
                            return True
                    elif neighbor in rec_stack:
                        return True

            rec_stack.remove(node)
            return False

        visited: set = set()
        for link_name in link_names:
            if link_name not in visited:
                if has_cycle(link_name, visited, set()):
                    context.add_error("Circular dependency detected in kinematic tree")
                    break

        connected_links = {"world"}
        for joint in joints:
            connected_links.add(joint.parent_link)
            connected_links.add(joint.child_link)

        orphaned = link_names - connected_links
        if orphaned:
            context.add_warning(f"Orphaned links detected: {orphaned}")

    def _parse_material(
        self, elem: ET.Element, context: ParseContext, *, inline: bool = False
    ) -> Material | None:
        """Parse URDF material element with validation.

        ``inline`` marks a visual's own ``<material>``. urdfdom accepts one with an
        empty name and uses its colour, which is how the SolidWorks exporter writes
        colours, so it is named after its content: equal ones share a name and
        different ones keep theirs apart in exports that key materials by name.
        """
        name = elem.get("name") or (_content_name(elem) if inline else None)
        if not name:
            context.add_warning("Material missing name attribute")
            return None

        material = Material(name=sanitize_name(name))

        color_elem = elem.find("color")
        if color_elem is not None:
            rgba_str = color_elem.get("rgba")
            if rgba_str:
                try:
                    rgba_values = [float(x) for x in rgba_str.split()]
                    if len(rgba_values) == 4:
                        material.color = rgba_values
                    else:
                        context.add_warning(f"Invalid RGBA format for material {name}")
                except ValueError:
                    context.add_warning(f"Invalid RGBA values for material {name}")

        texture_elem = elem.find("texture")
        if texture_elem is not None:
            filename = texture_elem.get("filename")
            if filename:
                material.texture = filename

        return material

    def _parse_visual(self, elem: ET.Element, context: ParseContext) -> Visual | None:
        """Parse URDF visual element with validation."""
        geom_elem = elem.find("geometry")
        if geom_elem is None:
            context.add_warning("Visual element missing geometry")
            return None

        geometry = self._parse_geometry(geom_elem, context)
        if not geometry:
            return None

        material = None
        material_elem = elem.find("material")
        if material_elem is not None:
            material_name = material_elem.get("name")
            if material_name and material_name in context.materials:
                material = context.materials[material_name]
            else:
                material = self._parse_material(material_elem, context, inline=True)

        pose = Pose()
        origin_elem = elem.find("origin")
        if origin_elem is not None:
            xyz = self._parse_xyz(origin_elem.get("xyz", "0 0 0"))
            rpy = self._parse_rpy(origin_elem.get("rpy", "0 0 0"))
            if xyz and rpy:
                pose = Pose.from_xyzrpy(xyz, rpy)

        return Visual(
            name=elem.get("name"),
            geometry=geometry,
            material=material,
            pose=pose,
        )

    def _parse_collision(
        self, elem: ET.Element, context: ParseContext
    ) -> Collision | None:
        """Parse URDF collision element with validation."""
        geom_elem = elem.find("geometry")
        if geom_elem is None:
            context.add_warning("Collision element missing geometry")
            return None

        geometry = self._parse_geometry(geom_elem, context)
        if not geometry:
            return None

        pose = Pose()
        origin_elem = elem.find("origin")
        if origin_elem is not None:
            xyz = self._parse_xyz(origin_elem.get("xyz", "0 0 0"))
            rpy = self._parse_rpy(origin_elem.get("rpy", "0 0 0"))
            if xyz and rpy:
                pose = Pose.from_xyzrpy(xyz, rpy)

        return Collision(
            name=elem.get("name"),
            geometry=geometry,
            pose=pose,
        )

    def _parse_geometry(
        self, elem: ET.Element, context: ParseContext
    ) -> Geometry | None:
        """Parse URDF geometry element with comprehensive type support."""
        box_elem = elem.find("box")
        if box_elem is not None:
            size_str = box_elem.get("size")
            if size_str:
                try:
                    size_values = [float(x) for x in size_str.split()]
                    if len(size_values) == 3:
                        return Geometry(
                            type=GeometryType.BOX,
                            size=Vector3(
                                size_values[0], size_values[1], size_values[2]
                            ),
                        )
                except ValueError:
                    context.add_warning("Invalid box size values")

        cylinder_elem = elem.find("cylinder")
        if cylinder_elem is not None:
            try:
                radius = float(cylinder_elem.get("radius", 0))
                length = float(cylinder_elem.get("length", 0))
                return Geometry(
                    type=GeometryType.CYLINDER,
                    radius=radius,
                    length=length,
                )
            except ValueError:
                context.add_warning("Invalid cylinder parameters")

        sphere_elem = elem.find("sphere")
        if sphere_elem is not None:
            try:
                radius = float(sphere_elem.get("radius", 0))
                return Geometry(
                    type=GeometryType.SPHERE,
                    radius=radius,
                )
            except ValueError:
                context.add_warning("Invalid sphere radius")

        mesh_elem = elem.find("mesh")
        if mesh_elem is not None:
            filename = mesh_elem.get("filename")
            if filename:
                original_filename = filename
                filename = resolve_mesh_uri(filename, context.base_dir)
                if filename != original_filename:
                    logger.debug(
                        "URDFParser: updated mesh filename from %s to %s",
                        original_filename,
                        filename,
                    )

                scale = None
                scale_str = mesh_elem.get("scale")
                if scale_str:
                    try:
                        scale_vals = [float(x) for x in scale_str.split()]
                        scale = Vector3(scale_vals[0], scale_vals[1], scale_vals[2])
                    except (ValueError, IndexError):
                        context.add_warning("Invalid mesh scale values")

                return Geometry(
                    type=GeometryType.MESH,
                    filename=filename,
                    scale=scale,
                )

        context.add_warning("No valid geometry found")
        return None

    def _parse_xyz(self, xyz_str: str) -> list[float] | None:
        """Parse XYZ coordinate string with validation."""
        if not xyz_str:
            return None

        try:
            values = [float(x.strip()) for x in xyz_str.split()]
            if len(values) != 3:
                return None
            return values
        except ValueError:
            return None

    def _parse_rpy(self, rpy_str: str) -> list[float] | None:
        """Parse RPY angle string with validation."""
        if not rpy_str:
            return None

        try:
            values = [float(x.strip()) for x in rpy_str.split()]
            if len(values) != 3:
                return None
            return values
        except ValueError:
            return None

    def _parse_float(self, value: str | None) -> float | None:
        """Parse float value with error handling."""
        if value is None:
            return None
        try:
            return float(value)
        except ValueError:
            return None


__all__ = ["URDFParser"]
