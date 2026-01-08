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

import logging
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from cyberwave_robot_format.core import BaseParser, ParseContext, ParseError
from cyberwave_robot_format.mesh import resolve_mesh_uri
from cyberwave_robot_format.schema import (
    Collision,
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
    Pose,
    Sensor,
    Vector3,
    Visual,
)
from cyberwave_robot_format.utils import sanitize_name

logger = logging.getLogger(__name__)


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
                        raise ParseError(f"XML parsing error after cleanup: {e2}") from e2
                    raise ParseError(f"XML parsing error: {e2}") from e2

        except ET.ParseError as e:
            raise ParseError(f"XML parsing error: {e}") from e

        context = ParseContext(
            file_path=file_path,
            base_dir=file_path.parent,
        )

        robot_name = root.get("name", "robot")
        if not robot_name:
            context.add_error("Robot name is required")
            robot_name = "unnamed_robot"

        metadata = Metadata(
            name=sanitize_name(robot_name),
            source_format="urdf",
            description=f"Converted from URDF: {file_path.name}",
            version=root.get("version", "1.0"),
        )

        for material_elem in root.findall("material"):
            material = self._parse_material(material_elem, context)
            if material and material.name:
                context.materials[material.name] = material

        links = []
        for link_elem in root.findall("link"):
            try:
                link = self._parse_link(link_elem, context)
                if link:
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

        schema = CommonSchema(
            metadata=metadata,
            links=links,
            joints=joints,
            sensors=sensors,
        )

        schema.extensions["parse_context"] = {
            "warnings": context.warnings,
            "errors": context.errors,
            "materials": context.materials,
        }

        return schema

    def _parse_urdf_sensors(self, root: ET.Element, context: ParseContext) -> list[Sensor]:
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

        sensor = Sensor(
            name=sanitize_name(name),
            type=sensor_type,
            parent_link=parent_link,
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

        sensors.append(sensor)

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
        )

        self._extract_camera_properties(cam_elem, sensor)

        sensors.append(sensor)

    def _extract_camera_properties(self, elem: ET.Element, sensor: Sensor) -> None:
        """Extract camera properties like fov, image size from element."""
        hfov = elem.find("hfov")
        if hfov is not None and hfov.text:
            try:
                sensor.parameters["fovy"] = float(hfov.text)
                sensor.parameters["hfov"] = float(hfov.text)
            except ValueError:
                pass

        image = elem.find("image")
        if image is not None:
            width = image.find("width")
            height = image.find("height")

            if width is not None and width.text:
                sensor.parameters["width"] = int(width.text)
            if height is not None and height.text:
                sensor.parameters["height"] = int(height.text)

        if "hfov" not in sensor.parameters:
            hfov = elem.find("hfov")
            if hfov is not None and hfov.text:
                sensor.parameters["hfov"] = float(hfov.text)

        img_size = elem.find("imageSize")
        if img_size is not None and img_size.text:
            try:
                parts = img_size.text.split()
                if len(parts) >= 2:
                    sensor.parameters["width"] = int(parts[0])
                    sensor.parameters["height"] = int(parts[1])
            except ValueError:
                pass

    def _parse_inertial(self, elem: ET.Element, link: Link, context: ParseContext) -> None:
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
        joint_type = elem.get("type")

        if not name:
            context.add_error("Joint name is required")
            return None

        if not joint_type:
            context.add_error("Joint type is required", name)
            return None

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
            context.add_error(f"Unsupported joint type: {joint_type}.", name)
            return None

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

        axis = elem.find("axis")
        if axis is not None:
            axis_xyz = self._parse_xyz(axis.get("xyz", "0 0 1"))
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
                    joint.axis = Vector3(0, 0, 1)

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

        return joint

    def _validate_kinematic_tree(self, links: list[Link], joints: list[Joint], context: ParseContext) -> None:
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

    def _parse_material(self, elem: ET.Element, context: ParseContext) -> Material | None:
        """Parse URDF material element with validation."""
        name = elem.get("name")
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
                material = self._parse_material(material_elem, context)

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

    def _parse_collision(self, elem: ET.Element, context: ParseContext) -> Collision | None:
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

    def _parse_geometry(self, elem: ET.Element, context: ParseContext) -> Geometry | None:
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
                            size=Vector3(size_values[0], size_values[1], size_values[2]),
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
