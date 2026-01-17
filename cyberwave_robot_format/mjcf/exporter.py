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

"""MJCF exporter for converting common schema to MuJoCo XML format."""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom

from cyberwave_robot_format.core import BaseExporter
from cyberwave_robot_format.mesh import convert_dae_to_obj
from cyberwave_robot_format.mjcf.utils import sanitize_inertia
from cyberwave_robot_format.schema import (
    Actuator,
    ActuatorType,
    Collision,
    CommonSchema,
    Contact,
    Geometry,
    GeometryType,
    Joint,
    JointType,
    Link,
    Material,
    Pose,
    Sensor,
    Vector3,
)

logger = logging.getLogger(__name__)


class MJCFExporter(BaseExporter):
    """Exporter for MJCF (MuJoCo Model Format) files."""

    def get_extension(self) -> str:
        """Return file extension for MJCF format."""
        return "xml"

    def export(self, schema: CommonSchema, output_path: str | Path) -> None:
        """Export common schema to MJCF format."""
        mujoco = ET.Element("mujoco", model=schema.metadata.name)

        ET.SubElement(mujoco, "compiler", angle="radian", autolimits="true")

        option_attrs = {}
        if schema.physics is not None:
            if schema.physics.gravity is not None:
                gravity = schema.physics.gravity
                option_attrs["gravity"] = f"{gravity.x} {gravity.y} {gravity.z}"
            if schema.physics.timestep is not None:
                option_attrs["timestep"] = str(schema.physics.timestep)
            if schema.physics.solver is not None:
                integrator_map = {
                    "euler": "euler",
                    "rk4": "rk4",
                    "implicit": "implicit",
                    "implicitfast": "implicitfast",
                    "quick": "implicitfast",
                }
                integrator = integrator_map.get(schema.physics.solver.type, "implicitfast")
                option_attrs["integrator"] = integrator
                if schema.physics.solver.iterations is not None:
                    option_attrs["iterations"] = str(schema.physics.solver.iterations)
                if schema.physics.solver.tolerance is not None:
                    option_attrs["tolerance"] = str(schema.physics.solver.tolerance)

        if not option_attrs:
            option_attrs["integrator"] = "implicitfast"

        ET.SubElement(mujoco, "option", **option_attrs)

        size_attrs = schema.extensions.get("mjcf_size") if hasattr(schema, "extensions") else None
        if size_attrs:
            ET.SubElement(mujoco, "size", **{k: str(v) for k, v in size_attrs.items()})

        default_elem = ET.SubElement(mujoco, "default")
        d_robot = ET.SubElement(default_elem, "default", {"class": "robot"})

        d_motor = ET.SubElement(d_robot, "default", {"class": "motor"})
        ET.SubElement(d_motor, "joint")
        ET.SubElement(d_motor, "motor", {"gear": "1"})

        d_visual = ET.SubElement(d_robot, "default", {"class": "visual"})
        ET.SubElement(d_visual, "geom", {"contype": "0", "conaffinity": "0", "group": "2"})

        d_collision = ET.SubElement(d_robot, "default", {"class": "collision"})
        ET.SubElement(
            d_collision,
            "geom",
            {
                "condim": "3",
                # contype/conaffinity now come from CollisionFilter on each collision
                "priority": "1",
                "group": "3",
                "solref": "0.005 1",
                "solimp": "0.99 0.999 1e-05",
                "friction": "1 0.01 0.01",
            },
        )

        defaults = schema.extensions.get("mjcf_defaults") if hasattr(schema, "extensions") else None
        if defaults and defaults.get("geom"):
            geom_defaults = defaults["geom"]
            geom_attrs = {}
            if geom_defaults.get("friction"):
                geom_attrs["friction"] = " ".join(str(x) for x in geom_defaults["friction"])
            if geom_defaults.get("material"):
                geom_attrs["material"] = geom_defaults["material"]
            if geom_attrs:
                ET.SubElement(default_elem, "geom", **geom_attrs)

        asset = ET.SubElement(mujoco, "asset")
        materials_used = set()

        for link in schema.links:
            for visual in link.visuals:
                if visual.material and visual.material.name:
                    materials_used.add(visual.material.name)

        for link in schema.links:
            for visual in link.visuals:
                if visual.material and visual.material.name in materials_used:
                    self._add_material(asset, visual.material)
                    materials_used.remove(visual.material.name)

        mesh_name_map = self._add_mesh_assets(asset, schema)

        worldbody = ET.SubElement(mujoco, "worldbody")

        child_links = {joint.child_link for joint in schema.joints}
        world_children = {joint.child_link for joint in schema.joints if joint.parent_link == "world"}
        root_names = {link.name for link in schema.links if link.name not in child_links and link.name != "world"}
        root_links = [link for link in schema.links if link.name in root_names]

        if not root_links and schema.links:
            root_links = [schema.links[0]]

        link_to_children: dict[str, list[tuple[Joint, str]]] = {}
        for joint in schema.joints:
            parent = joint.parent_link
            child = joint.child_link
            if parent not in link_to_children:
                link_to_children[parent] = []
            link_to_children[parent].append((joint, child))

        if "world" in link_to_children:
            for joint, child_link_name in link_to_children["world"]:
                child_link = next((link for link in schema.links if link.name == child_link_name), None)
                if child_link:
                    self._add_body_with_joint(
                        worldbody,
                        child_link,
                        joint,
                        schema,
                        link_to_children,
                        mesh_name_map,
                    )

        for root_link in root_links:
            if root_link.name != "world" and root_link.name not in world_children:
                self._add_body_hierarchy(worldbody, root_link, schema, link_to_children, mesh_name_map)

        # Add collision configuration (excludes and pairs)
        self._add_contact_section(mujoco, schema)

        actuator_elem = ET.SubElement(mujoco, "actuator")

        if schema.actuators:
            for actuator in schema.actuators:
                self._add_actuator(actuator_elem, actuator)
        else:
            for joint in schema.joints:
                if joint.type != JointType.FIXED and joint.type != JointType.FLOATING:
                    motor = ET.SubElement(actuator_elem, "motor")
                    motor.set("name", f"{joint.name}_ctrl")
                    motor.set("joint", joint.name)
                    motor.set("class", "motor")

                    if joint.limits and joint.limits.effort:
                        motor.set(
                            "forcerange",
                            f"-{joint.limits.effort} {joint.limits.effort}",
                        )

                    if joint.limits and joint.limits.lower is not None and joint.limits.upper is not None:
                        motor.set("ctrlrange", f"{joint.limits.lower} {joint.limits.upper}")

        if schema.sensors:
            sensor_elem = ET.SubElement(mujoco, "sensor")
            for sensor in schema.sensors:
                if sensor.type == "camera":
                    pass
                else:
                    self._add_sensor(sensor_elem, sensor)

        if schema.contacts:
            contact_elem = ET.SubElement(mujoco, "contact")
            for contact in schema.contacts:
                self._add_contact_pair(contact_elem, contact)

        self._write_pretty_xml(mujoco, output_path)
        logger.info("Exported MJCF to: %s", output_path)

    def _add_material(self, asset: ET.Element, material: Material) -> None:
        """Add material to asset section."""
        mat_elem = ET.SubElement(asset, "material", name=material.name)

        if material.color:
            rgba_str = " ".join(str(c) for c in material.color[:4])
            mat_elem.set("rgba", rgba_str)

        if hasattr(material, "specular") and material.specular:
            mat_elem.set("specular", str(material.specular))

        if hasattr(material, "shininess") and material.shininess:
            mat_elem.set("shininess", str(material.shininess))

        if material.texture:
            mat_elem.set("texture", material.texture)

        if material.emissive:
            mat_elem.set("emission", str(material.emissive[0]))

        reflectance = material.extensions.get("reflectance") if hasattr(material, "extensions") else None
        if reflectance is not None:
            mat_elem.set("reflectance", str(reflectance))

    def _add_mesh_assets(
        self, asset: ET.Element, schema: CommonSchema
    ) -> dict[tuple[str, tuple[float, float, float]], str]:
        """Collect mesh geometries and add mesh assets."""
        mesh_lookup: dict[tuple[str, tuple[float, float, float]], str] = {}
        used_names: dict[str, int] = {}

        def register_mesh(filename: str, scale: Vector3 | None) -> None:
            if not filename:
                return

            scale_tuple = (scale.x, scale.y, scale.z) if scale else (1.0, 1.0, 1.0)
            lookup_key = (filename, scale_tuple)

            if lookup_key in mesh_lookup:
                return

            original_filename = filename
            actual_filename = filename

            if filename.lower().endswith(".dae"):
                try:
                    actual_filename = convert_dae_to_obj(filename, scale if scale else None)
                except Exception as e:
                    logger.warning("Failed to convert DAE mesh %s: %s", filename, e)

            base_name = Path(original_filename).stem or "mesh"
            count = used_names.get(base_name, 0)
            if count:
                mesh_name = f"{base_name}_{count}"
            else:
                mesh_name = base_name
            used_names[base_name] = count + 1

            mesh_lookup[lookup_key] = mesh_name

            mesh_scale = "1 1 1"
            if actual_filename != original_filename:
                pass
            elif scale:
                mesh_scale = f"{scale.x} {scale.y} {scale.z}"

            mesh_attrs = {
                "name": mesh_name,
                "file": actual_filename,
                "scale": mesh_scale,
            }
            ET.SubElement(asset, "mesh", **mesh_attrs)

        for link in schema.links:
            for geom_container in list(link.visuals) + list(link.collisions):
                if geom_container.geometry and geom_container.geometry.type == GeometryType.MESH:
                    register_mesh(
                        geom_container.geometry.filename,
                        geom_container.geometry.scale,
                    )

        return mesh_lookup

    def _add_inertial(self, body: ET.Element, link: Link) -> None:
        """Add inertial element to body, ensuring validity."""
        if link.mass <= 0:
            return

        inertial = ET.SubElement(body, "inertial")
        inertial.set("mass", str(link.mass))

        if link.center_of_mass:
            pos = f"{link.center_of_mass.x} {link.center_of_mass.y} " f"{link.center_of_mass.z}"
            inertial.set("pos", pos)

        ixx, iyy, izz, ixy, ixz, iyz = sanitize_inertia(link)

        has_off_diag = any([ixy != 0, ixz != 0, iyz != 0])

        if has_off_diag:
            full = f"{ixx} {iyy} {izz} {ixy} {ixz} {iyz}"
            inertial.set("fullinertia", full)
        else:
            diag = f"{ixx} {iyy} {izz}"
            inertial.set("diaginertia", diag)

    def _add_body_hierarchy(
        self,
        parent_elem: ET.Element,
        link: Link,
        schema: CommonSchema,
        link_to_children: dict[str, list[tuple[Joint, str]]],
        mesh_name_map: dict[tuple[str, tuple[float, float, float]], str],
    ) -> None:
        """Add body and its children recursively."""
        body = ET.SubElement(parent_elem, "body", name=link.name)

        self._add_inertial(body, link)

        for i, visual in enumerate(link.visuals):
            self._add_geom(
                body,
                visual.geometry,
                visual.material,
                visual.pose,
                f"visual_{i}",
                mesh_name_map,
                group="2",
                schema=schema,
            )

        for i, collision in enumerate(link.collisions):
            self._add_geom(
                body,
                collision.geometry,
                None,
                collision.pose,
                f"collision_{i}",
                mesh_name_map,
                group="3",
                collision=collision,
                schema=schema,
            )

        if schema.sensors:
            for sensor in schema.sensors:
                if sensor.type == "camera" and sensor.parent_link == link.name:
                    self._add_camera(body, sensor)

        if link.name in link_to_children:
            for joint, child_link_name in link_to_children[link.name]:
                child_link = next((link for link in schema.links if link.name == child_link_name), None)
                if child_link:
                    self._add_body_with_joint(
                        body,
                        child_link,
                        joint,
                        schema,
                        link_to_children,
                        mesh_name_map,
                    )

    def _add_camera(self, body: ET.Element, sensor: Sensor) -> None:
        """Add camera element to body."""
        cam = ET.SubElement(body, "camera", name=sensor.name)

        if sensor.pose:
            if sensor.pose.position:
                pos = sensor.pose.position
                cam.set("pos", f"{pos.x} {pos.y} {pos.z}")
            if sensor.pose.orientation:
                quat = sensor.pose.orientation
                cam.set("quat", f"{quat.w} {quat.x} {quat.y} {quat.z}")

        if "hfov" in sensor.parameters:
            fovy = sensor.parameters.get("fovy", sensor.parameters["hfov"])
            cam.set("fovy", str(fovy))

    def _add_body_with_joint(
        self,
        parent_elem: ET.Element,
        link: Link,
        joint: Joint,
        schema: CommonSchema,
        link_to_children: dict[str, list[tuple[Joint, str]]],
        mesh_name_map: dict[tuple[str, tuple[float, float, float]], str],
    ) -> ET.Element:
        """Add body with joint connection."""
        body = ET.SubElement(parent_elem, "body", name=link.name)

        if joint.pose and joint.pose.position:
            pos = joint.pose.position
            pos_str = f"{pos.x} {pos.y} {pos.z}"
            body.set("pos", pos_str)

        if joint.pose and joint.pose.orientation:
            quat = joint.pose.orientation
            quat_str = f"{quat.w} {quat.x} {quat.y} {quat.z}"
            body.set("quat", quat_str)

        if joint.type != JointType.FIXED:
            joint_type_map = {
                JointType.REVOLUTE: "hinge",
                JointType.PRISMATIC: "slide",
                JointType.SPHERICAL: "ball",
                JointType.FLOATING: "free",
                JointType.CONTINUOUS: "hinge",
                JointType.PLANAR: "slide",
            }
            joint_attrs = {
                "name": joint.name,
                "type": joint_type_map.get(joint.type, "hinge"),
            }

            joint_elem = ET.SubElement(body, "joint", **joint_attrs)

            if joint.axis:
                axis_str = f"{joint.axis.x} {joint.axis.y} {joint.axis.z}"
                joint_elem.set("axis", axis_str)

            if joint.limits:
                if joint.limits.lower is not None and joint.limits.upper is not None:
                    range_str = f"{joint.limits.lower} {joint.limits.upper}"
                    joint_elem.set("range", range_str)

            if joint.dynamics:
                if joint.dynamics.damping is not None:
                    joint_elem.set("damping", str(joint.dynamics.damping))
                if joint.dynamics.friction is not None:
                    joint_elem.set("frictionloss", str(joint.dynamics.friction))
                if joint.dynamics.armature is not None:
                    joint_elem.set("armature", str(joint.dynamics.armature))
                if joint.dynamics.spring_stiffness is not None and joint.dynamics.spring_stiffness != 0.0:
                    joint_elem.set("stiffness", str(joint.dynamics.spring_stiffness))

        self._add_inertial(body, link)

        for i, visual in enumerate(link.visuals):
            self._add_geom(
                body,
                visual.geometry,
                visual.material,
                visual.pose,
                f"visual_{i}",
                mesh_name_map,
                group="2",
                schema=schema,
            )

        for i, collision in enumerate(link.collisions):
            self._add_geom(
                body,
                collision.geometry,
                None,
                collision.pose,
                f"collision_{i}",
                mesh_name_map,
                group="3",
                collision=collision,
                schema=schema,
            )

        if schema.sensors:
            for sensor in schema.sensors:
                if sensor.type == "camera" and sensor.parent_link == link.name:
                    self._add_camera(body, sensor)
                else:
                    logger.warning("Sensor %s with type %s is not supported.", sensor.name, sensor.type)

        if link.name in link_to_children:
            for child_joint, child_link_name in link_to_children[link.name]:
                child_link = next((link for link in schema.links if link.name == child_link_name), None)
                if child_link:
                    self._add_body_with_joint(
                        body,
                        child_link,
                        child_joint,
                        schema,
                        link_to_children,
                        mesh_name_map,
                    )

        return body

    def _add_geom(
        self,
        body: ET.Element,
        geometry: Geometry,
        material: Material | None,
        pose: Pose | None,
        name_suffix: str,
        mesh_name_map: dict[tuple[str, tuple[float, float, float]], str],
        group: str = "0",
        collision: Collision | None = None,
        schema: CommonSchema | None = None,
    ) -> None:
        """Add geometry element to body."""
        geom = ET.SubElement(body, "geom")
        geom.set("group", group)

        if pose and pose.position:
            pos = pose.position
            pos_str = f"{pos.x} {pos.y} {pos.z}"
            geom.set("pos", pos_str)

        if pose and pose.orientation:
            quat = pose.orientation
            quat_str = f"{quat.w} {quat.x} {quat.y} {quat.z}"
            geom.set("quat", quat_str)

        # Visual geoms: disable collision
        if group == "2":
            geom.set("class", "visual")
            geom.set("contype", "0")
            geom.set("conaffinity", "0")
        elif group == "3":
            geom.set("class", "collision")

        if geometry.type == GeometryType.BOX:
            geom.set("type", "box")
            if geometry.size:
                size_str = f"{geometry.size.x / 2} {geometry.size.y / 2} " f"{geometry.size.z / 2}"
                geom.set("size", size_str)

        elif geometry.type == GeometryType.SPHERE:
            geom.set("type", "sphere")
            if geometry.radius is not None:
                geom.set("size", str(geometry.radius))

        elif geometry.type == GeometryType.CYLINDER:
            geom.set("type", "cylinder")
            if geometry.radius is not None and geometry.length is not None:
                size_str = f"{geometry.radius} {geometry.length / 2}"
                geom.set("size", size_str)

        elif geometry.type == GeometryType.CAPSULE:
            geom.set("type", "capsule")
            if geometry.radius is not None and geometry.length is not None:
                size_str = f"{geometry.radius} {geometry.length / 2}"
                geom.set("size", size_str)

        elif geometry.type == GeometryType.PLANE:
            geom.set("type", "plane")
            if geometry.size:
                size_str = f"{geometry.size.x} {geometry.size.y} {geometry.size.z}"
                geom.set("size", size_str)

        elif geometry.type == GeometryType.MESH:
            geom.set("type", "mesh")
            if geometry.filename:
                scale_tuple = (
                    (geometry.scale.x, geometry.scale.y, geometry.scale.z) if geometry.scale else (1.0, 1.0, 1.0)
                )
                key = (geometry.filename, scale_tuple)

                if key in mesh_name_map:
                    mesh_name = mesh_name_map[key]
                    geom.set("mesh", mesh_name)

        else:
            raise ValueError(f"Unsupported geometry type for MJCF export: {geometry.type}")

        if material and material.name:
            geom.set("material", material.name)

        # Apply collision filter if this is a collision geom
        if collision:
            group_name = collision.group
            
            # "default" group doesn't require collision_config - use default MuJoCo behavior
            if group_name == "default" and (not schema or not schema.collision_config):
                # Use default MuJoCo collision behavior (all collisions enabled)
                # MuJoCo defaults: contype=1, conaffinity=1 (all collisions enabled)
                # We don't need to set these explicitly - MuJoCo will use defaults
                pass
            elif schema and schema.collision_config:
                # Look up collision group from schema.collision_config.groups
                if group_name not in schema.collision_config.groups:
                    raise ValueError(
                        f"Collision references group '{group_name}' which is not defined in "
                        f"collision_config.groups. Available groups: {list(schema.collision_config.groups.keys())}"
                    )
                
                collision_group = schema.collision_config.groups[group_name]
                
                # Apply contype/conaffinity from the group
                geom.set("contype", str(collision_group.contype))
                geom.set("conaffinity", str(collision_group.conaffinity))
            else:
                # Non-default group requires collision_config
                raise ValueError(
                    f"Collision references group '{group_name}' but schema has no collision_config"
                )
            
            # Apply contact properties (friction, etc.)
            if collision.mu_dynamic is not None:
                mu_static = collision.mu_static if collision.mu_static is not None else collision.mu_dynamic
                geom.set("friction", f"{collision.mu_dynamic} {mu_static} 0.01")
            
            # Check for MuJoCo-specific overrides in extensions (highest priority)
            if collision.extensions:
                for key in ["contype", "conaffinity", "margin", "solref", "solimp", "condim"]:
                    val = collision.extensions.get(key)
                    if val is not None:
                        geom.set(key, str(val))

    def _add_actuator(self, actuator_elem: ET.Element, actuator: Actuator) -> None:
        """Add actuator to actuator section."""
        raw_type = getattr(actuator, "type", None)
        norm = None
        if isinstance(raw_type, str):
            norm = raw_type.lower()
        type_map = {
            ActuatorType.DC_MOTOR: "motor",
            ActuatorType.POSITION: "position",
            ActuatorType.VELOCITY: "velocity",
            ActuatorType.TORQUE: "motor",
            ActuatorType.SERVO: "position",
            ActuatorType.MUSCLE: "muscle",
            "dc_motor": "motor",
            "motor": "motor",
            "position": "position",
            "velocity": "velocity",
            "torque": "motor",
            "servo": "position",
            "muscle": "muscle",
            "general": "general",
        }
        act_tag = type_map.get(raw_type) or type_map.get(norm, "general")
        act = ET.SubElement(actuator_elem, act_tag, name=actuator.name, joint=actuator.joint)

        if hasattr(actuator, "control_range") and actuator.control_range:
            ctrl_min, ctrl_max = actuator.control_range
            ctrl_range = f"{ctrl_min} {ctrl_max}"
            act.set("ctrlrange", ctrl_range)

        if hasattr(actuator, "force_range") and actuator.force_range:
            force_min, force_max = actuator.force_range
            force_range = f"{force_min} {force_max}"
            act.set("forcerange", force_range)
        elif hasattr(actuator, "max_torque") and actuator.max_torque is not None:
            # If max_torque is specified but no force_range, use symmetric range
            max_torque = actuator.max_torque
            act.set("forcerange", f"-{max_torque} {max_torque}")

        if actuator.gear_ratio is not None:
            act.set("gear", str(actuator.gear_ratio))

        if actuator.kp is not None and act_tag == "position":
            act.set("kp", str(actuator.kp))
        
        if actuator.kd is not None and act_tag in ["position", "velocity"]:
            act.set("kv", str(actuator.kd))

    def _add_sensor(self, sensor_elem: ET.Element, sensor: Sensor) -> None:
        """Add sensor entry."""
        sensor_type = sensor.type.lower()
        allowed_types = {
            "accelerometer": "accelerometer",
            "gyro": "gyro",
            "force": "force",
            "torque": "torque",
            "magnetometer": "magnetometer",
            "rangefinder": "rangefinder",
            "camera": "camera",
            "touch": "touch",
            "jointpos": "jointpos",
            "jointvel": "jointvel",
            "framepos": "framepos",
            "framequat": "framequat",
            "framelinvel": "framelinvel",
            "frameangvel": "frameangvel",
            "framelinacc": "framelinacc",
            "frameangacc": "frameangacc",
        }
        tag = allowed_types.get(sensor_type)
        if not tag:
            return
        attrs = {"name": sensor.name}

        if sensor_type in ("jointpos", "jointvel"):
            joint_name = sensor.parameters.get("joint") if hasattr(sensor, "parameters") else None
            if joint_name:
                attrs["joint"] = joint_name
        else:
            if sensor.parent_link and sensor.parent_link != "world":
                attrs["site"] = sensor.parent_link
        if sensor.pose and sensor.pose.position:
            pos = sensor.pose.position
            attrs["pos"] = f"{pos.x} {pos.y} {pos.z}"
        if sensor.pose and sensor.pose.orientation:
            quat = sensor.pose.orientation
            attrs["quat"] = f"{quat.w} {quat.x} {quat.y} {quat.z}"
        sens_elem = ET.SubElement(sensor_elem, tag, **attrs)
        if sensor.update_rate is not None:
            sens_elem.set("cutoff", str(sensor.update_rate))

    def _add_contact_pair(self, contact_elem: ET.Element, contact: Contact) -> None:
        """Add contact pair entries."""
        geom1 = contact.extensions.get("geom1") if contact.extensions else None
        geom2 = contact.extensions.get("geom2") if contact.extensions else None
        if not geom1 or not geom2:
            return
        attrs = {"geom1": geom1, "geom2": geom2, "name": contact.name}
        for key in ["solref", "solimp", "friction", "condim"]:
            val = contact.extensions.get(key) if contact.extensions else None
            if val is not None:
                attrs[key] = str(val)
        ET.SubElement(contact_elem, "pair", **attrs)

    def _add_contact_section(self, mujoco: ET.Element, schema: CommonSchema) -> None:
        """Add <contact> section with excludes and pairs."""
        if not schema.collision_config:
            return
        
        config = schema.collision_config
        
        # Only create <contact> if we have rules
        if not config.excludes and not config.pairs:
            return
        
        contact = ET.SubElement(mujoco, "contact")
        
        # Add body-level excludes
        for exclude in config.excludes:
            ET.SubElement(contact, "exclude", {
                "body1": exclude.body1,
                "body2": exclude.body2
            })
        
        # Add explicit geom pairs
        for pair in config.pairs:
            pair_attrs = {
                "geom1": pair.geom1,
                "geom2": pair.geom2
            }
            
            # Add optional contact overrides
            if pair.friction is not None:
                pair_attrs["friction"] = " ".join(str(f) for f in pair.friction)
            if pair.solref is not None:
                pair_attrs["solref"] = " ".join(str(s) for s in pair.solref)
            if pair.solimp is not None:
                pair_attrs["solimp"] = " ".join(str(s) for s in pair.solimp)
            if pair.condim is not None:
                pair_attrs["condim"] = str(pair.condim)
            if pair.margin is not None:
                pair_attrs["margin"] = str(pair.margin)
            if pair.gap is not None:
                pair_attrs["gap"] = str(pair.gap)
            
            # Add any extension attributes
            if pair.extensions:
                pair_attrs.update({k: str(v) for k, v in pair.extensions.items()})
            
            ET.SubElement(contact, "pair", pair_attrs)

    def _write_pretty_xml(self, root: ET.Element, output_path: str | Path) -> None:
        """Write XML with pretty formatting."""
        rough_string = ET.tostring(root, "utf-8")
        reparsed = minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent="  ")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(pretty_xml)


__all__ = ["MJCFExporter"]
