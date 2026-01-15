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

"""URDF exporter for converting common schema to URDF format."""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom

from cyberwave_robot_format.core import BaseExporter
from cyberwave_robot_format.schema import Collision, CommonSchema, GeometryType, Joint, JointType, Link, Visual

logger = logging.getLogger(__name__)


class URDFExporter(BaseExporter):
    """Exporter for URDF (Unified Robot Description Format) files."""

    def get_extension(self) -> str:
        """Return file extension for URDF format."""
        return "urdf"

    def export(self, schema: CommonSchema, output_path: str | Path) -> None:
        """Export common schema to URDF format."""
        # Validate that all joint parent/child links exist
        link_names = {link.name for link in schema.links}
        for joint in schema.joints:
            if joint.parent_link not in link_names:
                raise ValueError(
                    f"Joint '{joint.name}' references non-existent parent link '{joint.parent_link}'"
                )
            if joint.child_link not in link_names:
                raise ValueError(
                    f"Joint '{joint.name}' references non-existent child link '{joint.child_link}'"
                )

        robot = ET.Element("robot", name=schema.metadata.name)

        for link in schema.links:
            self._add_link(robot, link)

        for joint in schema.joints:
            self._add_joint(robot, joint)

        self._write_pretty_xml(robot, output_path)
        logger.info("Exported URDF to: %s", output_path)

    def _add_link(self, robot: ET.Element, link: Link) -> None:
        """Add link element to URDF."""
        link_elem = ET.SubElement(robot, "link", name=link.name)

        if link.mass > 0 or any([link.inertia.ixx, link.inertia.iyy, link.inertia.izz]):
            inertial = ET.SubElement(link_elem, "inertial")

            ET.SubElement(inertial, "mass", value=str(link.mass))

            com = link.center_of_mass
            ET.SubElement(inertial, "origin", xyz=f"{com.x} {com.y} {com.z}", rpy="0 0 0")

            inertia = link.inertia
            ET.SubElement(
                inertial,
                "inertia",
                ixx=str(inertia.ixx),
                iyy=str(inertia.iyy),
                izz=str(inertia.izz),
                ixy=str(inertia.ixy),
                ixz=str(inertia.ixz),
                iyz=str(inertia.iyz),
            )

        for visual in link.visuals:
            self._add_visual(link_elem, visual)

        for collision in link.collisions:
            self._add_collision(link_elem, collision)

    def _add_joint(self, robot: ET.Element, joint: Joint) -> None:
        """Add joint element to URDF."""
        type_mapping = {
            JointType.REVOLUTE: "revolute",
            JointType.CONTINUOUS: "continuous",
            JointType.PRISMATIC: "prismatic",
            JointType.FIXED: "fixed",
            JointType.FLOATING: "floating",
            JointType.PLANAR: "planar",
            JointType.SPHERICAL: "continuous",  # URDF approximation
            JointType.UNIVERSAL: "continuous",  # URDF approximation
        }

        urdf_type = type_mapping.get(joint.type, "fixed")
        joint_elem = ET.SubElement(robot, "joint", name=joint.name, type=urdf_type)

        ET.SubElement(joint_elem, "parent", link=joint.parent_link)
        ET.SubElement(joint_elem, "child", link=joint.child_link)

        pos = joint.pose.position
        ET.SubElement(joint_elem, "origin", xyz=f"{pos.x} {pos.y} {pos.z}", rpy="0 0 0")

        axis = joint.axis
        ET.SubElement(joint_elem, "axis", xyz=f"{axis.x} {axis.y} {axis.z}")

        if joint.limits:
            limits = joint.limits
            limit_elem = ET.SubElement(joint_elem, "limit")
            if limits.lower is not None:
                limit_elem.set("lower", str(limits.lower))
            if limits.upper is not None:
                limit_elem.set("upper", str(limits.upper))
            if limits.effort is not None:
                limit_elem.set("effort", str(limits.effort))
            if limits.velocity is not None:
                limit_elem.set("velocity", str(limits.velocity))

        if joint.dynamics:
            dyn = joint.dynamics
            ET.SubElement(
                joint_elem,
                "dynamics",
                damping=str(dyn.damping),
                friction=str(dyn.friction),
            )

    def _add_visual(self, link_elem: ET.Element, visual: Visual) -> None:
        """Add visual element to link."""
        visual_elem = ET.SubElement(link_elem, "visual")
        if visual.name:
            visual_elem.set("name", visual.name)

        pos = visual.pose.position
        ET.SubElement(visual_elem, "origin", xyz=f"{pos.x} {pos.y} {pos.z}", rpy="0 0 0")

        if visual.geometry:
            self._add_geometry(visual_elem, visual.geometry)

        if visual.material:
            self._add_material(visual_elem, visual.material)

    def _add_collision(self, link_elem: ET.Element, collision: Collision) -> None:
        """Add collision element to link."""
        collision_elem = ET.SubElement(link_elem, "collision")
        if collision.name:
            collision_elem.set("name", collision.name)

        pos = collision.pose.position
        ET.SubElement(collision_elem, "origin", xyz=f"{pos.x} {pos.y} {pos.z}", rpy="0 0 0")

        if collision.geometry:
            self._add_geometry(collision_elem, collision.geometry)

    def _add_geometry(self, parent: ET.Element, geometry) -> None:
        """Add geometry element."""
        geom_elem = ET.SubElement(parent, "geometry")

        if geometry.type == GeometryType.BOX:
            if geometry.size:
                size = geometry.size
                ET.SubElement(geom_elem, "box", size=f"{size.x} {size.y} {size.z}")

        elif geometry.type == GeometryType.CYLINDER:
            ET.SubElement(
                geom_elem,
                "cylinder",
                radius=str(geometry.radius or 1.0),
                length=str(geometry.length or 1.0),
            )

        elif geometry.type == GeometryType.SPHERE:
            ET.SubElement(geom_elem, "sphere", radius=str(geometry.radius or 1.0))

        elif geometry.type == GeometryType.MESH:
            mesh_attrs = {}
            if geometry.filename:
                mesh_attrs["filename"] = geometry.filename
            if geometry.scale:
                scale = geometry.scale
                mesh_attrs["scale"] = f"{scale.x} {scale.y} {scale.z}"
            ET.SubElement(geom_elem, "mesh", **mesh_attrs)

    def _add_material(self, parent: ET.Element, material) -> None:
        """Add material element."""
        mat_elem = ET.SubElement(parent, "material")
        if material.name:
            mat_elem.set("name", material.name)

        if material.color:
            color = material.color
            rgba_str = " ".join(str(c) for c in color)
            ET.SubElement(mat_elem, "color", rgba=rgba_str)

        if material.texture:
            ET.SubElement(mat_elem, "texture", filename=material.texture)

    def _write_pretty_xml(self, root: ET.Element, output_path: str | Path) -> None:
        """Write XML with pretty formatting."""
        rough_string = ET.tostring(root, "utf-8")
        reparsed = minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent="  ")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(pretty_xml)


__all__ = ["URDFExporter"]
