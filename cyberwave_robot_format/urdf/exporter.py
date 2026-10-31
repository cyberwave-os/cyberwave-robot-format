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
from cyberwave_robot_format.schema import (
    Collision,
    CommonSchema,
    GeometryType,
    Joint,
    JointType,
    Link,
    Material,
    Pose,
    Visual,
)
from cyberwave_robot_format.urdf.conventions import (
    AXIS_JOINT_TYPES,
    BOUNDED_JOINT_TYPES,
)
from cyberwave_robot_format.urdf.passthrough import (
    passthrough_from_extensions,
    render_urdf_passthrough,
)

logger = logging.getLogger(__name__)

_IDENTITY_EPS = 1e-12

# Written for a ``<limit>`` attribute URDF requires (``effort``, ``velocity``) when
# the schema has none. It re-imports as a real ``0.0``, not ``None``: the parser
# keeps what the file says, and CAD exporters (e.g. SolidWorks) already write
# ``effort="0" velocity="0"`` as the same placeholder. So a URDF-derived ``0`` in
# ``JointLimits.effort``/``velocity`` means "unspecified", and every reader of
# those fields in this package treats it so: the MJCF fallback motor
# (``mjcf/exporter.py``), the PhysX max joint velocity
# (``usd/physx.py::derived_joint_physx``) and the scene drive report
# (``usd/scene.py::_drive_entry``). A new reader must do the same, or it clamps
# the joint to zero torque/speed. Spelled ``"0.0"``, as ``str(0.0)`` is, so
# re-exporting the re-parsed schema writes the same text.
_UNSPECIFIED_LIMIT = "0.0"


def _material_has_content(material: Material) -> bool:
    """True when ``material`` defines an appearance, not just a name to look up."""
    return bool(
        material.color is not None
        or material.texture
        or material.specular is not None
        or material.emissive is not None
        or material.shininess is not None
        or material.extensions
    )


def _rpy_str(pose: Pose) -> str:
    """Return the pose orientation as a URDF ``rpy`` string (Fixed XYZ, radians)."""
    roll, pitch, yaw = pose.orientation.to_rpy()
    return f"{roll} {pitch} {yaw}"


def _is_identity_pose(pose: Pose) -> bool:
    """True when ``pose`` is the URDF default, i.e. an ``<origin>`` may be omitted."""
    position = pose.position
    roll, pitch, yaw = pose.orientation.to_rpy()
    return all(
        abs(component) <= _IDENTITY_EPS
        for component in (position.x, position.y, position.z, roll, pitch, yaw)
    )


def _add_origin(parent: ET.Element, pose: Pose) -> None:
    """Emit ``<origin>`` unless the pose is identity (which is what URDF defaults to).

    Omitting the identity case matters for round-trip equality: most authored URDFs
    leave the element out entirely, and materializing it everywhere would make every
    exported file differ from its original for no semantic reason.
    """
    if _is_identity_pose(pose):
        return
    position = pose.position
    ET.SubElement(
        parent,
        "origin",
        xyz=f"{position.x} {position.y} {position.z}",
        rpy=_rpy_str(pose),
    )


class URDFExporter(BaseExporter):
    """Exporter for URDF (Unified Robot Description Format) files."""

    def get_extension(self) -> str:
        """Return file extension for URDF format."""
        return "urdf"

    def export(self, schema: CommonSchema, output_path: str | Path) -> None:
        """Export common schema to URDF format."""
        self._write_pretty_xml(self.build_tree(schema), output_path)
        logger.info("Exported URDF to: %s", output_path)

    def export_string(self, schema: CommonSchema) -> str:
        """Return the URDF document as text, without touching the filesystem.

        The platform regenerates a served URDF straight into object storage, so it
        needs the bytes rather than a path. Same tree, same formatting as
        :meth:`export`.
        """
        return self._pretty_xml(self.build_tree(schema))

    def build_tree(self, schema: CommonSchema) -> ET.Element:
        """Build the ``<robot>`` element for ``schema``.

        Handles special cases:
        - If any joint has parent_link='world', a world link is automatically injected
        - JointType.SPHERICAL/UNIVERSAL are approximated (URDF has no native element)
        """
        robot = ET.Element("robot", name=schema.metadata.name)

        # Check if we need to inject a world link
        needs_world_link = any(j.parent_link == "world" for j in schema.joints)
        has_world_link = any(link.name == "world" for link in schema.links)

        if needs_world_link and not has_world_link:
            # Inject an empty world link
            ET.SubElement(robot, "link", name="world")
            logger.info("Injected world link for joints with parent_link='world'")

        # Document-level materials first: a visual may reference one by name, and a
        # reference that resolves to nothing renders untextured.
        document_materials = {
            material.name: material for material in schema.materials if material.name
        }
        for material in schema.materials:
            self._add_material(robot, material)

        for link in schema.links:
            self._add_link(robot, link, document_materials)

        declared_joint_names = {joint.name for joint in schema.joints if joint.name}
        for joint in schema.joints:
            self._add_joint(robot, joint, declared_joint_names)

        # Verbatim replay of everything the schema cannot type. Last, so the
        # kinematics a human reads first stay at the top of the file.
        emitted = render_urdf_passthrough(
            robot, passthrough_from_extensions(schema.extensions)
        )
        if emitted:
            logger.info("Re-emitted %d URDF passthrough block(s)", emitted)

        return robot

    def _add_link(
        self,
        robot: ET.Element,
        link: Link,
        document_materials: dict[str, Material] | None = None,
    ) -> None:
        """Add link element to URDF."""
        link_elem = ET.SubElement(robot, "link", name=link.name)

        if link.mass > 0 or any([link.inertia.ixx, link.inertia.iyy, link.inertia.izz]):
            inertial = ET.SubElement(link_elem, "inertial")

            ET.SubElement(inertial, "mass", value=str(link.mass))

            # The inertial frame's rotation lives in ``inertial_orientation``; it used
            # to be hardcoded to "0 0 0" here, which silently reoriented the inertia
            # tensor of every link whose URDF declared an rpy.
            com = link.center_of_mass
            _add_origin(
                inertial, Pose(position=com, orientation=link.inertial_orientation)
            )

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
            self._add_visual(link_elem, visual, document_materials or {})

        for collision in link.collisions:
            self._add_collision(link_elem, collision)

    def _add_joint(
        self,
        robot: ET.Element,
        joint: Joint,
        declared_joint_names: set[str],
    ) -> None:
        """Add joint element to URDF.

        JointType.FLOATING maps to URDF 'floating' type for mobile/floating base robots.

        ``declared_joint_names`` is required, not defaulted: it gates whether a
        ``<mimic>`` is emitted at all, and a default of ``None`` meant any future second
        caller silently opted out of that check and went back to emitting couplings that
        make the URDF unloadable.
        """
        type_mapping = {
            JointType.REVOLUTE: "revolute",
            JointType.CONTINUOUS: "continuous",
            JointType.PRISMATIC: "prismatic",
            JointType.FIXED: "fixed",
            JointType.FLOATING: "floating",
            JointType.PLANAR: "planar",
            JointType.SPHERICAL: "continuous",  # Approximate spherical as continuous
            JointType.UNIVERSAL: "continuous",  # Approximate universal as continuous
        }

        urdf_type = type_mapping.get(joint.type, "fixed")
        if joint.type in (JointType.SPHERICAL, JointType.UNIVERSAL):
            logger.warning(
                "Joint %r is %s, which URDF cannot express; exporting as %r",
                joint.name,
                joint.type.value,
                urdf_type,
            )
        joint_elem = ET.SubElement(robot, "joint", name=joint.name, type=urdf_type)

        ET.SubElement(joint_elem, "parent", link=joint.parent_link)
        ET.SubElement(joint_elem, "child", link=joint.child_link)

        _add_origin(joint_elem, joint.pose)

        # An <axis> is meaningless on a fixed/floating joint, and emitting one makes
        # every exported file differ from its original. See conventions.AXIS_JOINT_TYPES.
        if urdf_type in AXIS_JOINT_TYPES:
            axis = joint.axis
            ET.SubElement(joint_elem, "axis", xyz=f"{axis.x} {axis.y} {axis.z}")

        self._add_limit(joint_elem, joint, urdf_type)

        if joint.dynamics:
            dyn = joint.dynamics
            ET.SubElement(
                joint_elem,
                "dynamics",
                damping=str(dyn.damping),
                friction=str(dyn.friction),
            )

        for tag, values in (
            ("safety_controller", joint.safety_controller),
            ("calibration", joint.calibration),
        ):
            if not values:
                continue
            if tag == "safety_controller" and "k_velocity" not in values:
                # URDF requires `k_velocity`, and urdfdom rejects the joint (and so
                # the document) without it. No placeholder is safe: the safety
                # controller bounds effort by `k_velocity * (v_limit - v)`, so 0
                # caps the joint at zero effort. Omitting the element is what this
                # exporter did before it wrote safety controllers at all.
                logger.warning(
                    "Joint %r has a safety_controller without k_velocity, which URDF "
                    "requires; omitting <safety_controller> so the URDF stays "
                    "loadable (schema values: %r).",
                    joint.name,
                    values,
                )
                continue
            ET.SubElement(
                joint_elem,
                tag,
                {key: str(value) for key, value in values.items()},
            )

        if joint.mimic:
            # Skip a coupling whose driver the schema does not declare, matching the
            # MJCF exporter (`_add_equality_section`) and the backend
            # (`schema_mimic.mimic_by_joint_name`). Emitting it unconditionally meant
            # one schema produced a coupled gripper in the served URDF and a free,
            # undamped, uncommandable hinge in MJCF — and a `<mimic>` naming a joint
            # that does not exist makes the URDF itself unloadable for most parsers.
            driver = joint.mimic.joint
            if driver not in declared_joint_names:
                logger.warning(
                    "Joint %r mimics unknown joint %r; omitting <mimic> so the URDF "
                    "stays loadable. The joint exports as free, which is what every "
                    "other consumer already does with a dangling driver.",
                    joint.name,
                    driver,
                )
            else:
                ET.SubElement(
                    joint_elem,
                    "mimic",
                    joint=driver,
                    multiplier=str(joint.mimic.multiplier),
                    offset=str(joint.mimic.offset),
                )

    def _add_limit(self, joint_elem: ET.Element, joint: Joint, urdf_type: str) -> None:
        """Emit ``<limit>`` with the bounds URDF gives meaning to for this joint type.

        A ``continuous`` joint must not carry ``lower``/``upper``: every unlimited-hinge
        detector downstream reads the presence of bounds as "this hinge is limited", so
        emitting them converts the joint. It may still carry ``effort``/``velocity``.

        A ``revolute``/``prismatic`` joint *requires* a ``<limit>`` element; its
        ``lower``/``upper`` are optional and default to 0. When the schema has no
        bounds (reachable by flipping a continuous joint to revolute, which drops
        them) the joint either exports locked at 0 or, with no ``<limit>`` at all,
        makes the document invalid. Either way, say so rather than ship it quietly.

        ``effort`` and ``velocity`` are required on every ``<limit>``; urdfdom rejects
        the whole document when one is missing. The MJCF importer never fills them,
        so a missing one is written as :data:`_UNSPECIFIED_LIMIT`, with a warning.
        """
        limits = joint.limits
        attributes: dict[str, str] = {}

        if limits is not None:
            if urdf_type in BOUNDED_JOINT_TYPES:
                if limits.lower is not None:
                    attributes["lower"] = str(limits.lower)
                if limits.upper is not None:
                    attributes["upper"] = str(limits.upper)
            if limits.effort is not None:
                attributes["effort"] = str(limits.effort)
            if limits.velocity is not None:
                attributes["velocity"] = str(limits.velocity)

        if attributes:
            missing = [key for key in ("effort", "velocity") if key not in attributes]
            for key in missing:
                attributes[key] = _UNSPECIFIED_LIMIT
            if missing:
                logger.warning(
                    "Joint %r has no %s in the schema, which URDF requires on every "
                    "<limit>; writing %s=%r (unspecified).",
                    joint.name,
                    " or ".join(missing),
                    "/".join(missing),
                    _UNSPECIFIED_LIMIT,
                )

        if urdf_type in BOUNDED_JOINT_TYPES and (
            "lower" not in attributes or "upper" not in attributes
        ):
            consequence = (
                "URDF defaults a missing bound to 0, so the joint exports LOCKED at 0"
                if attributes
                else "with no <limit> at all, the exported URDF will be INVALID"
            )
            logger.warning(
                "Joint %r is type=%r but the schema has no positional bounds "
                "(lower=%r, upper=%r); %s. Patch limits.lower/limits.upper, or set "
                "the type back to 'continuous'.",
                joint.name,
                urdf_type,
                getattr(limits, "lower", None),
                getattr(limits, "upper", None),
                consequence,
            )

        if attributes:
            ET.SubElement(joint_elem, "limit", attributes)

    def _add_visual(
        self,
        link_elem: ET.Element,
        visual: Visual,
        document_materials: dict[str, Material],
    ) -> None:
        """Add visual element to link."""
        visual_elem = ET.SubElement(link_elem, "visual")
        if visual.name:
            visual_elem.set("name", visual.name)

        _add_origin(visual_elem, visual.pose)

        if visual.geometry:
            self._add_geometry(visual_elem, visual.geometry)

        if visual.material:
            # The parser resolves a by-name reference to the full document-level
            # Material, so inlining it here would duplicate every colour definition
            # once per visual. Emit the reference the source URDF actually had.
            #
            # The document copy is the source of truth for a named material (as in
            # urdfdom, where a global material wins over a visual's inline one of
            # the same name). After a save/reload the visual holds its own copy, so
            # an edit made only there is not exported; say so instead of dropping it
            # quietly. A name-only copy is just a reference and is not a conflict.
            document_material = document_materials.get(visual.material.name or "")
            if document_material is not None:
                if (
                    _material_has_content(visual.material)
                    and visual.material != document_material
                ):
                    logger.warning(
                        "Visual material %r differs from the document material of "
                        "the same name; exporting the document copy. Edit "
                        "schema.materials to change it.",
                        visual.material.name,
                    )
                ET.SubElement(visual_elem, "material", name=document_material.name)
            else:
                self._add_material(visual_elem, visual.material)

    def _add_collision(self, link_elem: ET.Element, collision: Collision) -> None:
        """Add collision element to link."""
        collision_elem = ET.SubElement(link_elem, "collision")
        if collision.name:
            collision_elem.set("name", collision.name)

        _add_origin(collision_elem, collision.pose)

        if collision.geometry:
            self._add_geometry(collision_elem, collision.geometry)

    def _add_geometry(self, parent: ET.Element, geometry) -> None:
        """Add geometry element."""
        geom_elem = ET.SubElement(parent, "geometry")

        if geometry.type == GeometryType.BOX:
            if geometry.size:
                size = geometry.size
                ET.SubElement(geom_elem, "box", size=f"{size.x} {size.y} {size.z}")
            else:
                # The one branch that could still leave ``<geometry/>`` empty, which
                # the fallback at the bottom of this method exists to prevent: URDF
                # requires exactly one shape child and most parsers reject the whole
                # document, so a single sizeless box took the entire robot down. Same
                # placeholder and same loud warning as the unhandled-type case.
                logger.warning(
                    "Box geometry has no size; exporting a 1m placeholder box. The "
                    "shape is WRONG and must be replaced."
                )
                ET.SubElement(geom_elem, "box", size="1 1 1")

        elif geometry.type == GeometryType.CYLINDER:
            ET.SubElement(
                geom_elem,
                "cylinder",
                radius=str(geometry.radius or 1.0),
                length=str(geometry.length or 1.0),
            )

        elif geometry.type == GeometryType.SPHERE:
            ET.SubElement(geom_elem, "sphere", radius=str(geometry.radius or 1.0))

        elif geometry.type == GeometryType.PLANE:
            # URDF doesn't have native plane support, represent as thin box
            if geometry.size:
                size = geometry.size
                ET.SubElement(geom_elem, "box", size=f"{size.x} {size.y} {size.z}")
            else:
                # Default 10x10m plane with 1cm thickness
                ET.SubElement(geom_elem, "box", size="10 10 0.01")

        elif geometry.type == GeometryType.MESH:
            mesh_attrs = {}
            if geometry.filename:
                mesh_attrs["filename"] = geometry.filename
            if geometry.scale:
                scale = geometry.scale
                mesh_attrs["scale"] = f"{scale.x} {scale.y} {scale.z}"
            ET.SubElement(geom_elem, "mesh", **mesh_attrs)

        elif geometry.type == GeometryType.CAPSULE:
            # URDF has no capsule. Approximate with the cylindrical section, the
            # conventional substitution: it keeps the radius and the axis exact and
            # only loses the two hemispherical caps.
            radius = geometry.radius or 1.0
            logger.warning(
                "Geometry is a capsule, which URDF cannot express; exporting as a "
                "cylinder (radius=%s length=%s), losing the hemispherical caps.",
                radius,
                geometry.length or 1.0,
            )
            ET.SubElement(
                geom_elem,
                "cylinder",
                radius=str(radius),
                length=str(geometry.length or 1.0),
            )

        elif geometry.type == GeometryType.ELLIPSOID:
            # ``size`` carries the three semi-axes (see the MJCF exporter); URDF has
            # no ellipsoid, so emit its bounding box.
            semi_axes = geometry.size
            logger.warning(
                "Geometry is an ellipsoid, which URDF cannot express; exporting as "
                "its bounding box."
            )
            if semi_axes:
                ET.SubElement(
                    geom_elem,
                    "box",
                    size=(
                        f"{semi_axes.x * 2} {semi_axes.y * 2} {semi_axes.z * 2}"
                    ),
                )
            else:
                radius = geometry.radius or 1.0
                ET.SubElement(geom_elem, "sphere", radius=str(radius))

        else:
            # Never leave ``<geometry/>`` empty: URDF requires exactly one shape child
            # and most parsers reject the document outright, so an unhandled type used
            # to take the whole robot down rather than one link. A unit box is visibly
            # wrong, which is the point — it is a placeholder, not a silent omission.
            logger.warning(
                "Geometry type %r has no URDF representation; exporting a 1m "
                "placeholder box. The shape is WRONG and must be replaced.",
                getattr(geometry.type, "value", geometry.type),
            )
            ET.SubElement(geom_elem, "box", size="1 1 1")

    def _add_material(self, parent: ET.Element, material: Material) -> None:
        """Add material element."""
        mat_elem = ET.SubElement(parent, "material")
        if material.name:
            mat_elem.set("name", material.name)

        if material.color:
            color = material.color
            rgba_str = " ".join(str(c) for c in color)
            ET.SubElement(mat_elem, "color", rgba=rgba_str)

        if material.texture:
            # MJCF texture is an asset identifier; URDF needs its image path.
            texture = material.extensions.get("mjcf_texture", {})
            ET.SubElement(mat_elem, "texture", filename=texture.get("file", material.texture))

    def _pretty_xml(self, root: ET.Element) -> str:
        """Render the tree as an indented XML document."""
        rough_string = ET.tostring(root, "utf-8")
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")

    def _write_pretty_xml(self, root: ET.Element, output_path: str | Path) -> None:
        """Write XML with pretty formatting."""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(self._pretty_xml(root))


__all__ = ["URDFExporter"]
