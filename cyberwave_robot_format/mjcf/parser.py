# Copyright [2025] Tomáš Macháček <tomasmachacekw@gmail.com>

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MJCF parser for converting MuJoCo XML files to common schema."""

from __future__ import annotations

import logging
import math
import xml.etree.ElementTree as ET
from pathlib import Path

from cyberwave_robot_format.core import BaseParser, ParseContext, ParseError
from cyberwave_robot_format.schema import (
    Actuator,
    ActuatorType,
    Collision,
    CommonSchema,
    Contact,
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
    Physics,
    PhysicsSolver,
    Pose,
    Quaternion,
    Sensor,
    Vector3,
    Visual,
)

from cyberwave_robot_format.utils import sanitize_name

logger = logging.getLogger(__name__)


class MJCFParser(BaseParser):
    """MJCF parser with comprehensive MuJoCo support and enhanced validation."""

    def __init__(self) -> None:
        super().__init__()
        self.supported_versions = ["2.3", "2.4", "3.0"]
        self._default_geom_friction: list[float] | None = None
        self._default_geom_material: str | None = None
        self._default_size: dict[str, str] = {}

    def can_parse(self, file_path: str | Path) -> bool:
        """Check if file is a valid MJCF file."""
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            return root.tag == "mujoco"
        except Exception:
            return False

    def parse(self, input_path: str | Path) -> CommonSchema:
        """Parse MJCF file with enhanced validation and comprehensive support."""
        file_path = Path(input_path)

        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
        except ET.ParseError as e:
            raise ParseError(f"XML parsing error: {e}") from e

        context = ParseContext(
            file_path=file_path,
            base_dir=file_path.parent,
        )

        model_name = root.get("model", "mujoco_model")
        metadata = Metadata(
            name=sanitize_name(model_name),
            source_format="mjcf",
            description=f"Converted from MJCF: {file_path.name}",
            version=root.get("version", "2.3"),
        )

        asset_elem = root.find("asset")
        if asset_elem is not None:
            context.materials = self._parse_materials(asset_elem, context)
            context.meshes = self._parse_meshes(asset_elem, context)

        self._default_geom_friction = None
        self._default_geom_material = None
        self._default_size = {}

        default_elem = root.find("default")
        if default_elem is not None:
            geom_default = default_elem.find("geom")
            if geom_default is not None:
                friction_attr = geom_default.get("friction")
                if friction_attr:
                    try:
                        self._default_geom_friction = [float(x) for x in friction_attr.split()]
                    except ValueError:
                        context.add_warning("Invalid default geom friction")
                material_attr = geom_default.get("material")
                if material_attr:
                    self._default_geom_material = sanitize_name(material_attr)

        size_elem = root.find("size")
        if size_elem is not None:
            size_attrs = {}
            for key in ["njmax", "nconmax", "nstack", "memory", "nuserdata"]:
                if size_elem.get(key) is not None:
                    size_attrs[key] = size_elem.get(key)
            self._default_size = size_attrs

        worldbody = root.find("worldbody")
        if worldbody is None:
            context.add_error("MJCF file missing worldbody element")
            return CommonSchema(metadata=metadata)

        links = []
        joints = []

        world_link = Link(name="world")
        links.append(world_link)

        for body_elem in worldbody.findall("body"):
            try:
                body_links, body_joints = self._parse_body_hierarchy(body_elem, "world", context)
                links.extend(body_links)
                joints.extend(body_joints)
            except Exception as e:
                context.add_error(f"Failed to parse body hierarchy: {e}")

        actuators = []
        actuator_elem = root.find("actuator")
        if actuator_elem is not None:
            actuators = self._parse_actuators(actuator_elem, context)

        sensors = []
        sensor_elem = root.find("sensor")
        if sensor_elem is not None:
            sensors = self._parse_sensors(sensor_elem, context)

        contacts = []
        contact_elem = root.find("contact")
        if contact_elem is not None:
            contacts = self._parse_contact_pairs(contact_elem, context)

        physics = None
        option_elem = root.find("option")
        if option_elem is not None:
            physics = self._parse_mjcf_physics(option_elem, context)

        schema = CommonSchema(
            metadata=metadata,
            links=links,
            joints=joints,
            actuators=actuators,
            sensors=sensors,
            physics=physics,
            contacts=contacts,
        )

        schema.extensions["parse_context"] = {
            "warnings": context.warnings,
            "errors": context.errors,
            "materials": context.materials,
            "meshes": context.meshes,
        }
        if self._default_size:
            schema.extensions["mjcf_size"] = self._default_size
        if self._default_geom_friction or self._default_geom_material:
            schema.extensions["mjcf_defaults"] = {
                "geom": {
                    "friction": self._default_geom_friction,
                    "material": self._default_geom_material,
                }
            }

        return schema

    def _parse_body_hierarchy(
        self,
        body_elem: ET.Element,
        parent_link: str,
        context: ParseContext,
    ) -> tuple[list[Link], list[Joint]]:
        """Parse MJCF body hierarchy recursively with enhanced validation."""
        links = []
        joints = []

        body_name = body_elem.get("name")
        if not body_name:
            context.add_error("Body missing name attribute")
            return links, joints

        body_name = sanitize_name(body_name)

        link = Link(name=body_name)
        body_pose: Pose | None = None

        pos_str = body_elem.get("pos")
        if pos_str:
            try:
                pos_values = [float(x) for x in pos_str.split()]
                if len(pos_values) == 3:
                    body_pose = Pose(position=Vector3(pos_values[0], pos_values[1], pos_values[2]))
            except ValueError:
                context.add_warning(f"Invalid position for body {body_name}")

        quat_str = body_elem.get("quat")
        if quat_str:
            try:
                quat_values = [float(x) for x in quat_str.split()]
                if len(quat_values) == 4:
                    orientation = Quaternion(quat_values[1], quat_values[2], quat_values[3], quat_values[0])
                    if body_pose is None:
                        body_pose = Pose(orientation=orientation)
                    else:
                        body_pose.orientation = orientation
            except ValueError:
                context.add_warning(f"Invalid quaternion for body {body_name}")

        link.pose = body_pose

        inertial_elem = body_elem.find("inertial")
        if inertial_elem is not None:
            self._parse_mjcf_inertial(inertial_elem, link, context)

        for geom_elem in body_elem.findall("geom"):
            self._parse_mjcf_geometry(geom_elem, link, context)

        links.append(link)

        joint_elems = body_elem.findall("joint")
        if joint_elems:
            for joint_elem in joint_elems:
                joint = self._parse_mjcf_joint(joint_elem, parent_link, body_name, context, body_pose)
                if joint:
                    joints.append(joint)
        elif parent_link != "world":
            joint = Joint(
                name=f"{body_name}_fixed",
                type=JointType.FIXED,
                parent_link=parent_link,
                child_link=body_name,
                pose=body_pose,
            )
            joints.append(joint)

        for child_body in body_elem.findall("body"):
            child_links, child_joints = self._parse_body_hierarchy(child_body, body_name, context)
            links.extend(child_links)
            joints.extend(child_joints)

        return links, joints

    def _parse_mjcf_inertial(
        self,
        inertial_elem: ET.Element,
        link: Link,
        context: ParseContext,
    ) -> None:
        """Parse MJCF inertial properties with validation."""
        mass_str = inertial_elem.get("mass")
        if mass_str:
            try:
                link.mass = float(mass_str)
                if link.mass < 0:
                    context.add_warning(f"Negative mass for link {link.name}")
            except ValueError:
                context.add_error(f"Invalid mass value for link {link.name}")

        pos_str = inertial_elem.get("pos")
        if pos_str:
            try:
                pos_values = [float(x) for x in pos_str.split()]
                if len(pos_values) == 3:
                    link.center_of_mass = Vector3(pos_values[0], pos_values[1], pos_values[2])
            except ValueError:
                context.add_warning(f"Invalid center of mass for link {link.name}")

        diaginertia_str = inertial_elem.get("diaginertia")
        if diaginertia_str:
            try:
                diag_values = [float(x) for x in diaginertia_str.split()]
                if len(diag_values) == 3:
                    link.inertia = Inertia(
                        ixx=diag_values[0],
                        iyy=diag_values[1],
                        izz=diag_values[2],
                        ixy=0.0,
                        ixz=0.0,
                        iyz=0.0,
                    )
            except ValueError:
                context.add_error(f"Invalid inertia values for link {link.name}")

        fullinertia_str = inertial_elem.get("fullinertia")
        if fullinertia_str:
            try:
                inertia_values = [float(x) for x in fullinertia_str.split()]
                if len(inertia_values) == 6:
                    link.inertia = Inertia(
                        ixx=inertia_values[0],
                        iyy=inertia_values[1],
                        izz=inertia_values[2],
                        ixy=inertia_values[3],
                        ixz=inertia_values[4],
                        iyz=inertia_values[5],
                    )
            except ValueError:
                context.add_error(f"Invalid full inertia values for link {link.name}")

    def _parse_mjcf_geometry(
        self,
        geom_elem: ET.Element,
        link: Link,
        context: ParseContext,
    ) -> None:
        """Parse MJCF geometry element for visual and collision."""
        geom_type = geom_elem.get("type", "sphere")
        size_str = geom_elem.get("size")

        if geom_type != "mesh" and not size_str:
            context.add_warning(f"Geometry missing size for link {link.name}")
            return

        size_values = []
        if size_str:
            try:
                size_values = [float(x) for x in size_str.split()]
            except ValueError:
                context.add_warning(f"Invalid size values for geometry in link {link.name}")
                return

        geometry = None

        if geom_type == "sphere" and len(size_values) >= 1:
            geometry = Geometry(type=GeometryType.SPHERE, radius=size_values[0])
        elif geom_type == "box" and len(size_values) >= 3:
            geometry = Geometry(
                type=GeometryType.BOX,
                size=Vector3(size_values[0] * 2, size_values[1] * 2, size_values[2] * 2),
            )
        elif geom_type == "cylinder" and len(size_values) >= 2:
            geometry = Geometry(
                type=GeometryType.CYLINDER,
                radius=size_values[0],
                length=size_values[1] * 2,
            )
        elif geom_type == "capsule" and len(size_values) >= 2:
            geometry = Geometry(
                type=GeometryType.CAPSULE,
                radius=size_values[0],
                length=size_values[1] * 2,
            )
        elif geom_type == "plane":
            if len(size_values) >= 2:
                geometry = Geometry(
                    type=GeometryType.PLANE,
                    size=Vector3(
                        size_values[0],
                        size_values[1],
                        size_values[2] if len(size_values) > 2 else 0.0,
                    ),
                )
        elif geom_type == "mesh":
            mesh_name = geom_elem.get("mesh")
            if mesh_name and mesh_name in context.meshes:
                mesh_info = context.meshes[mesh_name]
                geometry = Geometry(
                    type=GeometryType.MESH,
                    filename=mesh_info.get("filename"),
                    scale=mesh_info.get("scale"),
                )
                geom_scale = geom_elem.get("scale")
                if geom_scale:
                    try:
                        scale_vals = [float(x) for x in geom_scale.split()]
                        if len(scale_vals) == 3:
                            geometry.scale = Vector3(scale_vals[0], scale_vals[1], scale_vals[2])
                    except ValueError:
                        context.add_warning(f"Invalid geom scale for mesh in link {link.name}")
                if geometry.scale is None:
                    geometry.scale = Vector3(1.0, 1.0, 1.0)

        if geometry:
            material_name = geom_elem.get("material")
            material = None
            if material_name and material_name in context.materials:
                material = context.materials[material_name]
            else:
                rgba_attr = geom_elem.get("rgba")
                if rgba_attr:
                    try:
                        rgba_vals = [float(x) for x in rgba_attr.split()]
                        if len(rgba_vals) == 4:
                            material = Material(color=rgba_vals)
                    except ValueError:
                        context.add_warning(f"Invalid rgba for geom in link {link.name}")

            pose = Pose()
            pos_str = geom_elem.get("pos")
            quat_str = geom_elem.get("quat")

            if pos_str:
                try:
                    pos_values = [float(x) for x in pos_str.split()]
                    if len(pos_values) == 3:
                        position = Vector3(pos_values[0], pos_values[1], pos_values[2])

                        orientation = Quaternion(0, 0, 0, 1)
                        if quat_str:
                            quat_values = [float(x) for x in quat_str.split()]
                            if len(quat_values) == 4:
                                orientation = Quaternion(
                                    quat_values[1],
                                    quat_values[2],
                                    quat_values[3],
                                    quat_values[0],
                                )

                        pose = Pose(position=position, orientation=orientation)

                except ValueError:
                    context.add_warning(f"Invalid geometry position for link {link.name}")

            group = geom_elem.get("group")
            if group is None:
                add_visual = True
                add_collision = True
            elif group == "2":
                add_visual = True
                add_collision = False
            elif group == "3":
                add_visual = False
                add_collision = True
            else:
                add_visual = True
                add_collision = True

            if add_visual:
                visual = Visual(
                    name=f"{link.name}_visual_{len(link.visuals)}",
                    geometry=geometry,
                    material=material,
                    pose=pose,
                )
                link.visuals.append(visual)

            if add_collision:
                collision = Collision(
                    name=f"{link.name}_collision_{len(link.collisions)}",
                    geometry=geometry,
                    pose=pose,
                )

                friction_attr = geom_elem.get("friction")
                if friction_attr:
                    try:
                        friction_vals = [float(x) for x in friction_attr.split()]
                        if friction_vals:
                            collision.mu_dynamic = friction_vals[0]
                            collision.mu_static = friction_vals[0]
                            collision.extensions["friction"] = friction_vals
                    except ValueError:
                        context.add_warning(f"Invalid friction values for geom in link {link.name}")
                elif self._default_geom_friction:
                    friction_vals = self._default_geom_friction
                    collision.mu_dynamic = friction_vals[0]
                    collision.mu_static = friction_vals[0]
                    collision.extensions["friction"] = friction_vals

                if material is None and self._default_geom_material:
                    mat_name = self._default_geom_material
                    if mat_name in context.materials:
                        material = context.materials[mat_name]

                for attr in ["contype", "conaffinity", "margin", "solref", "solimp"]:
                    val = geom_elem.get(attr)
                    if val:
                        collision.extensions[attr] = val

                link.collisions.append(collision)

    def _parse_mjcf_joint(
        self,
        joint_elem: ET.Element,
        parent_link: str,
        child_link: str,
        context: ParseContext,
        body_pose: Pose | None = None,
    ) -> Joint | None:
        """Parse MJCF joint element with enhanced validation."""
        joint_name = joint_elem.get("name")
        if not joint_name:
            joint_name = f"{child_link}_joint"

        joint_name = sanitize_name(joint_name)

        joint_type_str = joint_elem.get("type", "hinge")
        type_mapping = {
            "hinge": JointType.REVOLUTE,
            "slide": JointType.PRISMATIC,
            "ball": JointType.SPHERICAL,
            "free": JointType.FLOATING,
        }

        joint_type = type_mapping.get(joint_type_str, JointType.REVOLUTE)

        joint = Joint(
            name=joint_name,
            type=joint_type,
            parent_link=parent_link,
            child_link=child_link,
        )

        pos_str = joint_elem.get("pos")
        quat_str = joint_elem.get("quat")
        if pos_str:
            try:
                pos_vals = [float(x) for x in pos_str.split()]
                if len(pos_vals) == 3:
                    joint.pose.position = Vector3(pos_vals[0], pos_vals[1], pos_vals[2])
            except ValueError:
                context.add_warning(f"Invalid joint pos for {joint_name}")
        if quat_str:
            try:
                quat_vals = [float(x) for x in quat_str.split()]
                if len(quat_vals) == 4:
                    joint.pose.orientation = Quaternion(quat_vals[1], quat_vals[2], quat_vals[3], quat_vals[0])
            except ValueError:
                context.add_warning(f"Invalid joint quat for {joint_name}")

        if body_pose:
            # Simple pose composition for translation (assuming small/no rotation for basic cases)
            # TODO: Implement full pose composition in math_utils
            if body_pose.position:
                joint.pose.position.x += body_pose.position.x
                joint.pose.position.y += body_pose.position.y
                joint.pose.position.z += body_pose.position.z
            if body_pose.orientation:
                # If body has orientation, ideally we should compose quaternions
                # For now, if joint has identity orientation, take body's orientation
                if (
                    joint.pose.orientation.w == 1.0
                    and joint.pose.orientation.x == 0.0
                    and joint.pose.orientation.y == 0.0
                    and joint.pose.orientation.z == 0.0
                ):
                    joint.pose.orientation = Quaternion(
                        body_pose.orientation.x,
                        body_pose.orientation.y,
                        body_pose.orientation.z,
                        body_pose.orientation.w,
                    )

        axis_str = joint_elem.get("axis")
        if axis_str:
            try:
                axis_values = [float(x) for x in axis_str.split()]
                if len(axis_values) == 3:
                    magnitude = math.sqrt(sum(x**2 for x in axis_values))
                    if magnitude > 1e-6:
                        joint.axis = Vector3(
                            axis_values[0] / magnitude,
                            axis_values[1] / magnitude,
                            axis_values[2] / magnitude,
                        )
                    else:
                        context.add_warning(f"Zero-magnitude joint axis for {joint_name}")
            except ValueError:
                context.add_warning(f"Invalid joint axis for {joint_name}")

        range_str = joint_elem.get("range")
        if range_str:
            try:
                range_values = [float(x) for x in range_str.split()]
                if len(range_values) == 2:
                    joint.limits = JointLimits(lower=range_values[0], upper=range_values[1])
            except ValueError:
                context.add_warning(f"Invalid joint range for {joint_name}")

        damping = self._parse_float_attr(joint_elem, "damping")
        frictionloss = self._parse_float_attr(joint_elem, "frictionloss")
        armature = self._parse_float_attr(joint_elem, "armature")
        stiffness = self._parse_float_attr(joint_elem, "stiffness")

        if any(v is not None for v in [damping, frictionloss, armature, stiffness]):
            joint.dynamics = JointDynamics(
                damping=damping or 0.0,
                friction=frictionloss or 0.0,
                armature=armature,
                spring_stiffness=stiffness or 0.0,
            )

        return joint

    def _parse_materials(
        self,
        asset_elem: ET.Element,
        context: ParseContext,
    ) -> dict[str, Material]:
        """Parse materials from asset section with validation."""
        materials = {}

        for mat_elem in asset_elem.findall("material"):
            name = mat_elem.get("name")
            if not name:
                context.add_warning("Material missing name attribute")
                continue

            material = Material(name=name)

            rgba_str = mat_elem.get("rgba")
            if rgba_str:
                try:
                    rgba = [float(x) for x in rgba_str.split()]
                    if len(rgba) == 4:
                        material.color = rgba
                    else:
                        context.add_warning(f"Invalid RGBA format for material {name}")
                except ValueError:
                    context.add_warning(f"Invalid RGBA values for material {name}")

            specular_str = mat_elem.get("specular")
            if specular_str:
                try:
                    specular = [float(x) for x in specular_str.split()]
                    material.specular = specular
                except ValueError:
                    context.add_warning(f"Invalid specular values for material {name}")

            shininess = self._parse_float_attr(mat_elem, "shininess")
            if shininess is not None:
                material.shininess = shininess

            reflectance = self._parse_float_attr(mat_elem, "reflectance")
            if reflectance is not None:
                material.extensions["reflectance"] = reflectance

            texture = mat_elem.get("texture")
            if texture:
                material.texture = texture

            emission = self._parse_float_attr(mat_elem, "emission")
            if emission is not None:
                material.emissive = [emission] * 3

            materials[name] = material

        return materials

    def _parse_meshes(
        self,
        asset_elem: ET.Element,
        context: ParseContext,
    ) -> dict[str, dict]:
        """Parse mesh assets with path resolution."""
        meshes = {}

        for mesh_elem in asset_elem.findall("mesh"):
            name = mesh_elem.get("name")
            filename = mesh_elem.get("file")
            scale_attr = mesh_elem.get("scale")

            if name and filename:
                mesh_info = {}
                if not Path(filename).is_absolute():
                    full_path = context.base_dir / filename
                    if full_path.exists():
                        mesh_info["filename"] = str(full_path)
                    else:
                        context.add_warning(f"Mesh file not found: {filename}")
                        mesh_info["filename"] = filename
                else:
                    mesh_info["filename"] = filename

                if scale_attr:
                    try:
                        scale_vals = [float(x) for x in scale_attr.split()]
                        if len(scale_vals) == 3:
                            mesh_info["scale"] = Vector3(scale_vals[0], scale_vals[1], scale_vals[2])
                    except ValueError:
                        context.add_warning(f"Invalid mesh scale for {name}")

                meshes[name] = mesh_info

        return meshes

    def _parse_actuators(
        self,
        actuator_elem: ET.Element,
        context: ParseContext,
    ) -> list[Actuator]:
        """Parse actuator definitions with validation."""
        actuators = []

        for motor_elem in actuator_elem.findall("motor"):
            name = motor_elem.get("name")
            joint = motor_elem.get("joint")

            if not name or not joint:
                context.add_warning("Motor missing name or joint reference")
                continue

            actuator = Actuator(
                name=sanitize_name(name),
                joint=sanitize_name(joint),
                type=ActuatorType.DC_MOTOR,
            )

            gear_str = motor_elem.get("gear")
            if gear_str:
                try:
                    gear_values = [float(x) for x in gear_str.split()]
                    if gear_values:
                        actuator.gear_ratio = gear_values[0]
                except ValueError:
                    context.add_warning(f"Invalid gear values for motor {name}")

            ctrlrange_str = motor_elem.get("ctrlrange")
            if ctrlrange_str:
                try:
                    ctrl_range = [float(x) for x in ctrlrange_str.split()]
                    if len(ctrl_range) == 2:
                        actuator.control_range = (ctrl_range[0], ctrl_range[1])
                except ValueError:
                    context.add_warning(f"Invalid control range for motor {name}")

            actuators.append(actuator)

        return actuators

    def _parse_sensors(
        self,
        sensor_elem: ET.Element,
        context: ParseContext,
    ) -> list[Sensor]:
        """Parse sensor definitions with comprehensive type support."""
        sensors = []

        basic_sensor_types = [
            "accelerometer",
            "gyro",
            "force",
            "torque",
            "magnetometer",
            "rangefinder",
            "camera",
            "touch",
        ]

        for sensor_type in basic_sensor_types:
            for sens_elem in sensor_elem.findall(sensor_type):
                name = sens_elem.get("name")
                site = sens_elem.get("site")

                if not name:
                    context.add_warning(f"Sensor {sensor_type} missing name")
                    continue

                sensor = Sensor(
                    name=sanitize_name(name),
                    type=sensor_type,
                    parent_link=site or "world",
                )

                if sensor_type == "camera":
                    resolution = sens_elem.get("resolution")
                    if resolution:
                        try:
                            res_values = [int(x) for x in resolution.split()]
                            sensor.parameters["resolution"] = res_values
                        except ValueError:
                            context.add_warning(f"Invalid resolution for camera {name}")

                sensors.append(sensor)

        joint_frame_defs = [
            ("jointpos", "joint"),
            ("jointvel", "joint"),
            ("framepos", "site"),
            ("framequat", "site"),
            ("framelinvel", "site"),
            ("frameangvel", "site"),
            ("framelinacc", "site"),
            ("frameangacc", "site"),
        ]

        for tag, attr in joint_frame_defs:
            for sens_elem in sensor_elem.findall(tag):
                name = sens_elem.get("name")
                target = sens_elem.get(attr)
                if not name:
                    context.add_warning(f"Sensor {tag} missing name")
                    continue
                if not target:
                    context.add_warning(f"Sensor {tag} missing {attr}")
                    continue

                parent_link = sanitize_name(target) if attr == "site" else "world"
                sensor = Sensor(
                    name=sanitize_name(name),
                    type=tag,
                    parent_link=parent_link,
                )
                if attr == "joint":
                    sensor.parameters["joint"] = sanitize_name(target)
                else:
                    sensor.parameters["site"] = sanitize_name(target)

                sensors.append(sensor)

        return sensors

    def _parse_float_attr(self, elem: ET.Element, attr: str) -> float | None:
        """Parse float attribute with error handling."""
        value = elem.get(attr)
        if value is None:
            return None
        try:
            return float(value)
        except ValueError:
            return None

    def _parse_contact_pairs(
        self,
        contact_elem: ET.Element,
        context: ParseContext,
    ) -> list[Contact]:
        """Parse contact pairs into schema contacts."""
        contacts: list[Contact] = []
        for pair in contact_elem.findall("pair"):
            geom1 = pair.get("geom1")
            geom2 = pair.get("geom2")
            if not geom1 or not geom2:
                context.add_warning("Contact pair missing geom1 or geom2")
                continue
            name = pair.get("name", f"pair_{geom1}_{geom2}")
            contact = Contact(name=sanitize_name(name), link="world")
            contact.extensions["geom1"] = geom1
            contact.extensions["geom2"] = geom2
            for attr in ["solref", "solimp", "friction", "condim"]:
                val = pair.get(attr)
                if val is not None:
                    contact.extensions[attr] = val
            contacts.append(contact)
        return contacts

    def _parse_mjcf_physics(
        self,
        option_elem: ET.Element,
        context: ParseContext,
    ) -> Physics | None:
        """Parse MJCF physics from <option> element."""
        physics = Physics()

        gravity_str = option_elem.get("gravity")
        if gravity_str:
            try:
                gravity_vals = [float(x) for x in gravity_str.split()]
                if len(gravity_vals) >= 3:
                    physics.gravity = Vector3(gravity_vals[0], gravity_vals[1], gravity_vals[2])
            except ValueError:
                context.add_warning("Invalid gravity value in MJCF option")

        timestep_str = option_elem.get("timestep")
        if timestep_str:
            try:
                physics.timestep = float(timestep_str)
            except ValueError:
                context.add_warning("Invalid timestep value in MJCF option")

        integrator = option_elem.get("integrator")
        if integrator:
            solver = PhysicsSolver()
            integrator_map = {
                "euler": "euler",
                "rk4": "rk4",
                "implicit": "implicit",
                "implicitfast": "implicitfast",
            }
            solver.type = integrator_map.get(integrator.lower(), "quick")
            physics.solver = solver

        iterations_str = option_elem.get("iterations")
        if iterations_str:
            try:
                if physics.solver is None:
                    physics.solver = PhysicsSolver()
                physics.solver.iterations = int(float(iterations_str))
            except ValueError:
                context.add_warning("Invalid iterations value in MJCF option")

        tolerance_str = option_elem.get("tolerance")
        if tolerance_str:
            try:
                if physics.solver is None:
                    physics.solver = PhysicsSolver()
                physics.solver.tolerance = float(tolerance_str)
            except ValueError:
                context.add_warning("Invalid tolerance value in MJCF option")

        return physics if physics.gravity or physics.timestep or physics.solver else None


__all__ = ["MJCFParser"]
