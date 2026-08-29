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
import math
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

# Default dynamics applied to a URDF ``<mimic>`` slave joint that carries no
# authored dynamics. The ``<equality>`` coupling constrains the slave's
# *position*, but between solver iterations an undamped gram-scale finger link
# can still ring; a touch of damping/armature dissipates it. Defined here as the
# single source of truth so the value cannot drift from other producers.
MIMIC_SLAVE_DEFAULT_DAMPING = 0.2
MIMIC_SLAVE_DEFAULT_ARMATURE = 0.001

# A MuJoCo ``<equality><joint>`` couples two *scalar* (1-DOF hinge/slide) joints.
# These joint types cannot participate: FIXED emits no ``<joint>`` element,
# FLOATING emits a 6-DOF ``<free>``, and SPHERICAL emits a 3-DOF ``<ball>`` —
# none is a valid ``joint1``/``joint2`` for a scalar polycoef constraint.
_EQUALITY_INELIGIBLE_JOINT_TYPES = frozenset(
    {JointType.FIXED, JointType.FLOATING, JointType.SPHERICAL}
)

# Half-range (rad) of the ctrlrange given to a *continuous* joint's position /
# velocity actuator. A continuous joint has no positional limits (it exports as
# a ``limited="false"`` free hinge), but MuJoCo position actuators still need a
# finite ``ctrlrange`` — otherwise it defaults to ``[0, 0]`` and freezes the
# joint. The finite band therefore lives on the *actuator* (``home ± span``),
# not on the joint. Matches the RL ``continuous_span_rad`` default so training
# and export agree.
CONTINUOUS_ACTUATOR_CTRL_SPAN = math.pi


# Schema sensor type -> the MuJoCo ``<sensor>`` tags it expands to.
#
# The two vocabularies are not the same shape. MuJoCo has no "IMU" — it has a
# gyro and an accelerometer sharing a site — so a device-level schema type maps
# to several primitives. Raw MuJoCo tag names are accepted verbatim too, so a
# schema authored directly against MJCF keeps working.
_MJCF_SENSOR_TAGS: dict[str, tuple[str, ...]] = {
    # Device-level types used by the universal schema / asset editor.
    "imu": ("gyro", "accelerometer"),
    "contact": ("touch",),
    "touch_sensor": ("touch",),
    "ultrasound": ("rangefinder",),
    "sonar": ("rangefinder",),
    "force_torque": ("force", "torque"),
    # Raw MuJoCo primitives, passed through unchanged.
    "accelerometer": ("accelerometer",),
    "gyro": ("gyro",),
    "velocimeter": ("velocimeter",),
    "force": ("force",),
    "torque": ("torque",),
    "magnetometer": ("magnetometer",),
    "rangefinder": ("rangefinder",),
    "touch": ("touch",),
    "jointpos": ("jointpos",),
    "jointvel": ("jointvel",),
    "framepos": ("framepos",),
    "framequat": ("framequat",),
    "framelinvel": ("framelinvel",),
    "frameangvel": ("frameangvel",),
    "framelinacc": ("framelinacc",),
    "frameangacc": ("frameangacc",),
}

# Sensors MuJoCo cannot express as a ``<sensor>`` element. They are sampled at
# runtime by casting ``mj_ray`` (see ``cyberwave-sim/sensors/range.py``), so their
# absence from the MJCF is correct rather than a gap — hence no warning.
_RUNTIME_SAMPLED_SENSOR_TYPES = frozenset(
    {
        "lidar",
        "lidar_2d",
        "lidar_3d",
        "lidar_4d",
        "laser",
        "laser_scan",
        "ray",
        "radar",
        "radar_2d",
        "radar_3d",
        "pointcloud",
        "point_cloud",
        "depth",
        "rgb",
        "map",
    }
)

# MuJoCo sensors that measure at a frame and therefore need a ``<site>``.
# ``jointpos``/``jointvel`` reference a joint instead, and are excluded.
_SITE_ATTACHED_TAGS = frozenset(
    {
        "accelerometer",
        "gyro",
        "velocimeter",
        "magnetometer",
        "rangefinder",
        "touch",
        "force",
        "torque",
    }
)

# Sites exist only as sensor mount frames; group 4 keeps them out of the way of
# the visual (2) / collision (3) / camera-housing (5) groups the exporter uses.
_SENSOR_SITE_GROUP = "4"


def _needs_site(sensor_type: str | None) -> bool:
    """True when a schema sensor maps to at least one site-attached primitive."""
    tags = _MJCF_SENSOR_TAGS.get((sensor_type or "").lower())
    if not tags:
        return False
    return any(tag in _SITE_ATTACHED_TAGS for tag in tags)


def _rgba_attribute(color: list[float]) -> str | None:
    """Format a schema colour as MuJoCo's ``rgba``, which is strictly four reals.

    A schema colour is not guaranteed to carry alpha: ``USDParser`` builds a
    three-component colour whenever a ``UsdPreviewSurface`` omits its ``opacity``
    input. MuJoCo rejects a short ``rgba`` outright ("attribute 'rgba' does not
    have enough data") and refuses to compile the model, so a missing alpha
    defaults to opaque instead. Returns ``None`` for a colour too short to be one.
    """
    if len(color) < 3:
        return None
    rgba = list(color[:4])
    if len(rgba) == 3:
        rgba.append(1.0)
    return " ".join(str(c) for c in rgba)


def _is_passive_joint(joint: Joint) -> bool:
    """True when the schema explicitly marks a joint as undriven.

    Opt-in via ``extensions.passive`` rather than inferred: there is no reliable
    structural signal for "undriven" in an imported URDF (``<transmission>``
    declares the driven set, but nothing converts it), and guessing from joint
    names would be exactly the kind of heuristic this codebase forbids. Assets
    that declare their own ``actuators`` bypass this path entirely, which is the
    preferred fix.
    """
    extensions = getattr(joint, "extensions", None)
    if not isinstance(extensions, dict):
        return False
    return bool(extensions.get("passive"))


def site_name_for_sensor(sensor: Sensor) -> str:
    """Name of the ``<site>`` that carries this sensor.

    Derived from the sensor name, which the scene composer has already namespaced
    per twin, so two twins of the same asset cannot collide.
    """
    return f"{sensor.name}_site"


class MJCFExporter(BaseExporter):
    """Exporter for MJCF (MuJoCo Model Format) files."""

    def get_extension(self) -> str:
        """Return file extension for MJCF format."""
        return "xml"

    def export(self, schema: CommonSchema, output_path: str | Path) -> None:
        """Export common schema to MJCF format."""
        # Names of the <site> elements actually emitted during body construction.
        # ``_add_sensor`` consults it before writing a site reference: MuJoCo fails
        # to compile a model whose sensor names a site that does not exist, so a
        # sensor whose parent_link matched no exported body must be skipped rather
        # than left dangling. Reset per export so a reused exporter cannot carry
        # names across models.
        self._emitted_sites: set[str] = set()
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
                integrator = integrator_map.get(
                    schema.physics.solver.type, "implicitfast"
                )
                option_attrs["integrator"] = integrator
                if schema.physics.solver.iterations is not None:
                    option_attrs["iterations"] = str(schema.physics.solver.iterations)
                if schema.physics.solver.tolerance is not None:
                    option_attrs["tolerance"] = str(schema.physics.solver.tolerance)

        if not option_attrs:
            option_attrs["integrator"] = "implicitfast"

        option_attrs.setdefault("noslip_iterations", "10")

        ET.SubElement(mujoco, "option", **option_attrs)

        size_attrs = (
            schema.extensions.get("mjcf_size")
            if hasattr(schema, "extensions")
            else None
        )
        if size_attrs:
            ET.SubElement(mujoco, "size", **{k: str(v) for k, v in size_attrs.items()})

        default_elem = ET.SubElement(mujoco, "default")
        d_robot = ET.SubElement(default_elem, "default", {"class": "robot"})

        d_motor = ET.SubElement(d_robot, "default", {"class": "motor"})
        ET.SubElement(d_motor, "joint")
        ET.SubElement(d_motor, "motor", {"gear": "1"})

        d_visual = ET.SubElement(d_robot, "default", {"class": "visual"})
        ET.SubElement(
            d_visual, "geom", {"contype": "0", "conaffinity": "0", "group": "2"}
        )

        d_collision = ET.SubElement(d_robot, "default", {"class": "collision"})
        ET.SubElement(
            d_collision,
            "geom",
            {
                "condim": "3",
                "contype": "1",
                "conaffinity": "1",
                "priority": "1",
                "group": "3",
                "solref": "0.02 1",
                "solimp": "0.7 0.9 0.02",
                "friction": "0.6 0.005 0.005",
            },
        )

        defaults = (
            schema.extensions.get("mjcf_defaults")
            if hasattr(schema, "extensions")
            else None
        )
        if defaults and defaults.get("geom"):
            geom_defaults = defaults["geom"]
            geom_attrs = {}
            if geom_defaults.get("friction"):
                geom_attrs["friction"] = " ".join(
                    str(x) for x in geom_defaults["friction"]
                )
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
        world_children = {
            joint.child_link for joint in schema.joints if joint.parent_link == "world"
        }
        root_names = {
            link.name
            for link in schema.links
            if link.name not in child_links and link.name != "world"
        }
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
                child_link = next(
                    (link for link in schema.links if link.name == child_link_name),
                    None,
                )
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
                self._add_body_hierarchy(
                    worldbody, root_link, schema, link_to_children, mesh_name_map
                )

        # Add collision configuration (excludes and pairs)
        self._add_contact_section(mujoco, schema)

        self._add_equality_section(mujoco, schema)

        actuator_elem = ET.SubElement(mujoco, "actuator")

        if schema.actuators:
            # Build a joint-name → Joint lookup so _add_actuator can resolve the
            # ctrlrange: from the joint's limits for a bounded joint, or from
            # ``home ± span`` for a continuous joint (which carries no limits).
            joints_by_name = {j.name: j for j in schema.joints}
            for actuator in schema.actuators:
                target_joint = joints_by_name.get(actuator.joint)
                if target_joint is not None and target_joint.mimic is not None:
                    # A mimic slave is already position-constrained by its
                    # <equality>; an actuator (especially a position servo) on
                    # the same joint fights the coupling. This is an authoring
                    # error, but SKIP the offending actuator rather than raising:
                    # export runs per *scene*, so raising here fails every scene
                    # containing the asset — one bad asset patch takes down
                    # unrelated environments. Skipping yields a physically correct
                    # model (the coupling drives the joint) and leaves a loud log
                    # for whoever owns the patch. The conflict is better caught at
                    # patch-apply time, where it is actionable.
                    logger.error(
                        "Actuator %r targets mimic slave joint %r, which is "
                        "position-constrained by an <equality> coupling. An "
                        "actuator on a mimic slave fights the coupling; skipping "
                        "it. Remove the actuator or the joint's <mimic>.",
                        actuator.name,
                        target_joint.name,
                    )
                    continue
                self._add_actuator(actuator_elem, actuator, joints_by_name)
        else:
            # No declared actuators: fall back to one motor per movable joint so an
            # imported URDF is still controllable. Joints the schema marks passive
            # are skipped — a caster swivel or free-rolling wheel must not be
            # driven, and motorizing it both fights the physics and adds phantom
            # entries to the control vector.
            for joint in schema.joints:
                if _is_passive_joint(joint):
                    logger.debug(
                        "Joint %s is marked passive; no actuator emitted.", joint.name
                    )
                    continue
                # Mimic slaves are driven by the <equality> coupling, not an
                # independent actuator — emitting a fallback motor for them would
                # double-drive the joint.
                if (
                    joint.type != JointType.FIXED
                    and joint.type != JointType.FLOATING
                    and joint.mimic is None
                ):
                    motor = ET.SubElement(actuator_elem, "motor")
                    motor.set("name", f"{joint.name}_ctrl")
                    motor.set("joint", joint.name)
                    motor.set("class", "motor")

                    if joint.limits and joint.limits.effort:
                        motor.set(
                            "forcerange",
                            f"-{joint.limits.effort} {joint.limits.effort}",
                        )

                    if (
                        joint.limits
                        and joint.limits.lower is not None
                        and joint.limits.upper is not None
                    ):
                        motor.set(
                            "ctrlrange", f"{joint.limits.lower} {joint.limits.upper}"
                        )

        if schema.sensors:
            sensor_elem = ET.SubElement(mujoco, "sensor")
            for sensor in schema.sensors:
                if sensor.type in ("camera", "depth_camera"):
                    pass  # cameras become <camera> elements in the body, not <sensor>
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
            rgba_str = _rgba_attribute(material.color)
            if rgba_str:
                mat_elem.set("rgba", rgba_str)

        if hasattr(material, "specular") and material.specular:
            mat_elem.set("specular", str(material.specular))

        if hasattr(material, "shininess") and material.shininess:
            mat_elem.set("shininess", str(material.shininess))

        if material.texture:
            mat_elem.set("texture", material.texture)

        if material.emissive:
            mat_elem.set("emission", str(material.emissive[0]))

        reflectance = material.extensions.get("reflectance")
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
                    actual_filename = convert_dae_to_obj(
                        filename, scale if scale else None
                    )
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
                if (
                    geom_container.geometry
                    and geom_container.geometry.type == GeometryType.MESH
                ):
                    register_mesh(
                        geom_container.geometry.filename,
                        geom_container.geometry.scale,
                    )

        return mesh_lookup

    # Fallback for movable bodies (frames, tool sites) that the source schema
    # leaves with zero mass. MuJoCo refuses to load any moving body whose mass
    # or inertia is below mjMINVAL, so the exporter has to emit *something*.
    _FALLBACK_MASS = 1e-6
    _FALLBACK_DIAGINERTIA = "1e-9 1e-9 1e-9"

    def _add_inertial(
        self, body: ET.Element, link: Link, *, ensure_minimum: bool = False
    ) -> None:
        """Add inertial element to body, ensuring validity.

        ``ensure_minimum=True`` guarantees that the body has a non-zero
        inertial even when ``link.mass`` is missing or non-positive. Callers
        that know the body will have a joint underneath should pass it; this
        is the case for every body emitted via :meth:`_add_body_with_joint`.
        """
        has_mass = link.mass and link.mass > 0
        if not has_mass:
            if ensure_minimum:
                inertial = ET.SubElement(body, "inertial")
                inertial.set("mass", str(self._FALLBACK_MASS))
                inertial.set("pos", "0 0 0")
                inertial.set("diaginertia", self._FALLBACK_DIAGINERTIA)
                logger.debug(
                    "Emitted fallback inertial for movable body '%s' (link.mass=%s).",
                    link.name,
                    link.mass,
                )
            return

        inertial = ET.SubElement(body, "inertial")
        inertial.set("mass", str(link.mass))

        if link.center_of_mass:
            pos = (
                f"{link.center_of_mass.x} {link.center_of_mass.y} "
                f"{link.center_of_mass.z}"
            )
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

        # Links that host camera sensors get group=5 so the wrist-depth renderer
        # (which already disables geomgroup[5]) does not see the camera housing.
        _camera_parent_links: frozenset[str] = frozenset(
            s.parent_link
            for s in (schema.sensors or [])
            if s.type in ("camera", "depth_camera")
        )
        visual_group = "5" if link.name in _camera_parent_links else "2"
        collision_group = "5" if link.name in _camera_parent_links else "3"

        for i, visual in enumerate(link.visuals):
            self._add_geom(
                body,
                visual.geometry,
                visual.material,
                visual.pose,
                f"visual_{i}",
                mesh_name_map,
                group=visual_group,
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
                group=collision_group,
                collision=collision,
                schema=schema,
            )

        if schema.sensors:
            for sensor in schema.sensors:
                if sensor.parent_link != link.name:
                    continue
                if sensor.type in ("camera", "depth_camera"):
                    self._add_camera(body, sensor)
                else:
                    # Root links come through this path, and a site-attached sensor
                    # mounted here (an IMU on base_link is the common case) needs
                    # its <site> on this body. Omitting it left <gyro site="..."/>
                    # pointing at nothing, which MuJoCo rejects for the whole model
                    # — so one root-mounted IMU broke the entire scene export.
                    self._add_sensor_site(body, sensor)

        if link.name in link_to_children:
            for joint, child_link_name in link_to_children[link.name]:
                child_link = next(
                    (link for link in schema.links if link.name == child_link_name),
                    None,
                )
                if child_link:
                    self._add_body_with_joint(
                        body,
                        child_link,
                        joint,
                        schema,
                        link_to_children,
                        mesh_name_map,
                    )

    def _add_sensor_site(self, body: ET.Element, sensor: Sensor) -> None:
        """Create the ``<site>`` a site-attached sensor will reference.

        ``_add_sensor`` writes ``site="..."`` on gyro/accelerometer/rangefinder
        and friends, but nothing ever created those sites — and MuJoCo fails to
        *compile* a model whose sensor references a missing site. Declaring such a
        sensor therefore used to break the whole scene rather than being ignored.

        The site also gives the runtime raycaster an exact mount frame to cast
        from (see ``cyberwave-sim/sensors/range.py``), which is what a
        corner-mounted scanner needs.
        """
        if not _needs_site(sensor.type):
            return
        site_name = site_name_for_sensor(sensor)
        site = ET.SubElement(body, "site", name=site_name)
        site.set("group", _SENSOR_SITE_GROUP)
        self._emitted_sites.add(site_name)
        if sensor.pose:
            if sensor.pose.position:
                pos = sensor.pose.position
                site.set("pos", f"{pos.x} {pos.y} {pos.z}")
            if sensor.pose.orientation:
                quat = sensor.pose.orientation
                if not (
                    abs(quat.w - 1.0) < 1e-9
                    and abs(quat.x) < 1e-9
                    and abs(quat.y) < 1e-9
                    and abs(quat.z) < 1e-9
                ):
                    site.set("quat", f"{quat.w} {quat.x} {quat.y} {quat.z}")

    def _add_camera(self, body: ET.Element, sensor: Sensor) -> None:
        """Add camera element to body."""
        cam = ET.SubElement(body, "camera", name=sensor.name)

        if sensor.pose:
            if sensor.pose.position:
                pos = sensor.pose.position
                cam.set("pos", f"{pos.x} {pos.y} {pos.z}")
            if sensor.pose.orientation:
                quat = sensor.pose.orientation
                # Skip identity to keep MJCF compact; non-identity encodes optical offset.
                if not (
                    abs(quat.w - 1.0) < 1e-9
                    and abs(quat.x) < 1e-9
                    and abs(quat.y) < 1e-9
                    and abs(quat.z) < 1e-9
                ):
                    cam.set("quat", f"{quat.w} {quat.x} {quat.y} {quat.z}")

        params = sensor.parameters or {}
        vfov_rad = params.get("vfov_rad")
        has_valid_vfov_rad = (
            isinstance(vfov_rad, int | float)
            and not isinstance(vfov_rad, bool)
            and vfov_rad > 0
        )
        # vfov_rad is canonical; fovy is a legacy fallback.
        if has_valid_vfov_rad:
            fovy_deg = math.degrees(float(vfov_rad))
            cam.set("fovy", str(fovy_deg))
        elif "hfov" in params or "fovy" in params or "fov_degrees" in params:
            if "fovy" in params:
                fovy_deg = float(params["fovy"])
            elif "fov_degrees" in params:
                fovy_deg = float(params["fov_degrees"])
            else:
                # hfov is stored in radians (SI).  Convert to vertical FOV in
                # degrees for MuJoCo, accounting for the sensor's aspect ratio.
                hfov_rad = float(params["hfov"])
                width = float(params.get("width", 640))
                height = float(params.get("height", 480))
                # hfov → vfov: tan(vfov/2) = tan(hfov/2) * (height/width)
                vfov_rad = 2.0 * math.atan(math.tan(hfov_rad / 2.0) * height / width)
                fovy_deg = math.degrees(vfov_rad)
            cam.set("fovy", str(fovy_deg))

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

            if joint.type == JointType.CONTINUOUS:
                # A continuous joint rotates without limit. Never emit a positional
                # ``range`` even if the schema carries limits, and set
                # ``limited="false"`` explicitly so neither a stray range nor the
                # ``autolimits`` inference can turn it into a limited hinge. The
                # finite command band a position actuator needs lives on the
                # actuator's ``ctrlrange`` instead (see ``_add_actuator``).
                joint_elem.set("limited", "false")
            elif (
                joint.limits
                and joint.limits.lower is not None
                and joint.limits.upper is not None
            ):
                range_str = f"{joint.limits.lower} {joint.limits.upper}"
                joint_elem.set("range", range_str)
            elif joint.type in (JointType.REVOLUTE, JointType.PRISMATIC):
                # A bounded joint type with no usable positional bounds. With
                # ``autolimits="true"`` and no ``range`` this compiles to
                # ``jnt_limited == 0``, and every unlimited-hinge detector downstream
                # (cyberwave-sim/control/actuation.py, cyberwave-sim/mqtt/state_egress.py,
                # controller-sim-adapter/joint_mapping.py) gates purely on
                # ``hinge && !jnt_limited`` — so a schema-``revolute`` hinge is silently
                # reinterpreted as CONTINUOUS: driven shortest-path, allowed to wind freely,
                # and angle-wrapped in telemetry, while the schema and frontend still treat it
                # as bounded (physicalPositionRange returns null, uiControlRange falls back to
                # a fixed [-pi, pi], formatContinuousAngle refuses to unwrap it).
                #
                # We do NOT invent a range here — fabricating physical limits is worse than
                # the ambiguity. Surface it instead; the schema layer warns on the same
                # condition in ``urdf_schema_sync._sync_limit``.
                logger.warning(
                    "Joint %r is type=%s but has no positional bounds; MJCF will compile it "
                    "as an UNLIMITED hinge, which the sim and deploy adapter classify as "
                    "continuous. Patch limits.lower/limits.upper, or set the type to "
                    "'continuous' if it really rotates without limit.",
                    joint.name,
                    joint.type.value if hasattr(joint.type, "value") else joint.type,
                )

            if joint.dynamics:
                if joint.dynamics.damping is not None:
                    joint_elem.set("damping", str(joint.dynamics.damping))
                if joint.dynamics.friction is not None:
                    joint_elem.set("frictionloss", str(joint.dynamics.friction))
                if joint.dynamics.armature is not None:
                    joint_elem.set("armature", str(joint.dynamics.armature))
                if (
                    joint.dynamics.spring_stiffness is not None
                    and joint.dynamics.spring_stiffness != 0.0
                ):
                    joint_elem.set("stiffness", str(joint.dynamics.spring_stiffness))
            elif joint.mimic is not None:
                # Mimic slaves without authored dynamics get a light default
                # damping/armature (see MIMIC_SLAVE_DEFAULT_* for the rationale).
                joint_elem.set("damping", str(MIMIC_SLAVE_DEFAULT_DAMPING))
                joint_elem.set("armature", str(MIMIC_SLAVE_DEFAULT_ARMATURE))

        # A movable body must always have a valid inertial.
        is_movable_joint = joint.type not in (JointType.FIXED,)
        self._add_inertial(body, link, ensure_minimum=is_movable_joint)

        # Links that host camera sensors get group=5 so the wrist-depth renderer
        # (which already disables geomgroup[5]) does not see the camera housing.
        _camera_parent_links: frozenset[str] = frozenset(
            s.parent_link
            for s in (schema.sensors or [])
            if s.type in ("camera", "depth_camera")
        )
        visual_group = "5" if link.name in _camera_parent_links else "2"
        collision_group = "5" if link.name in _camera_parent_links else "3"

        for i, visual in enumerate(link.visuals):
            self._add_geom(
                body,
                visual.geometry,
                visual.material,
                visual.pose,
                f"visual_{i}",
                mesh_name_map,
                group=visual_group,
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
                group=collision_group,
                collision=collision,
                schema=schema,
            )

        if schema.sensors:
            for sensor in schema.sensors:
                if sensor.parent_link != link.name:
                    continue
                if sensor.type in ("camera", "depth_camera"):
                    self._add_camera(body, sensor)
                else:
                    # Every site-attached MuJoCo sensor (gyro, accelerometer,
                    # rangefinder, touch, ...) needs a <site> to reference. The
                    # site must be created on the body it is mounted to, which is
                    # only reachable here.
                    self._add_sensor_site(body, sensor)

        if link.name in link_to_children:
            for child_joint, child_link_name in link_to_children[link.name]:
                child_link = next(
                    (link for link in schema.links if link.name == child_link_name),
                    None,
                )
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

        if collision is not None and collision.name:
            geom.set("name", collision.name)

        if pose and pose.position:
            pos = pose.position
            pos_str = f"{pos.x} {pos.y} {pos.z}"
            geom.set("pos", pos_str)

        if pose and pose.orientation:
            quat = pose.orientation
            quat_str = f"{quat.w} {quat.x} {quat.y} {quat.z}"
            geom.set("quat", quat_str)

        # Visual geoms: disable collision.
        # Group "5" is the camera-mount convention: same class/collision rules
        # as groups 2/3 but hidden from the depth-camera renderer.
        if group == "2" or (group == "5" and collision is None):
            geom.set("class", "visual")
            geom.set("contype", "0")
            geom.set("conaffinity", "0")
        elif group == "3" or (group == "5" and collision is not None):
            geom.set("class", "collision")

        if geometry.type == GeometryType.BOX:
            geom.set("type", "box")
            if geometry.size:
                size_str = (
                    f"{geometry.size.x / 2} {geometry.size.y / 2} {geometry.size.z / 2}"
                )
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

        elif geometry.type == GeometryType.ELLIPSOID:
            geom.set("type", "ellipsoid")
            if geometry.size:
                # ``size`` carries the three semi-axes (rx, ry, rz).
                size_str = f"{geometry.size.x} {geometry.size.y} {geometry.size.z}"
                geom.set("size", size_str)

        elif geometry.type == GeometryType.MESH:
            geom.set("type", "mesh")
            if geometry.filename:
                scale_tuple = (
                    (geometry.scale.x, geometry.scale.y, geometry.scale.z)
                    if geometry.scale
                    else (1.0, 1.0, 1.0)
                )
                key = (geometry.filename, scale_tuple)

                if key in mesh_name_map:
                    mesh_name = mesh_name_map[key]
                    geom.set("mesh", mesh_name)

        else:
            raise ValueError(
                f"Unsupported geometry type for MJCF export: {geometry.type}"
            )

        if material and material.name:
            geom.set("material", material.name)
        elif material and material.color:
            # Anonymous material: no <asset><material> is emitted for it, so the
            # colour has to ride on the geom itself. This is the inverse of the
            # parser reading a bare `rgba` geom attribute into a nameless Material.
            # Only the colour survives here — a geom `rgba` cannot carry a texture
            # or specular, which MuJoCo can express only through a named material.
            rgba_str = _rgba_attribute(material.color)
            if rgba_str:
                geom.set("rgba", rgba_str)

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
                mu_static = (
                    collision.mu_static
                    if collision.mu_static is not None
                    else collision.mu_dynamic
                )
                geom.set("friction", f"{collision.mu_dynamic} {mu_static} 0.01")

            # Check for MuJoCo-specific overrides in extensions (highest priority)
            if collision.extensions:
                for key in [
                    "contype",
                    "conaffinity",
                    "margin",
                    "solref",
                    "solimp",
                    "condim",
                    "friction",
                ]:
                    val = collision.extensions.get(key)
                    if val is not None:
                        geom.set(key, str(val))

    def _add_actuator(
        self,
        actuator_elem: ET.Element,
        actuator: Actuator,
        joints_by_name: dict | None = None,
    ) -> None:
        """Add actuator to actuator section.

        Args:
            actuator_elem: Parent ``<actuator>`` XML element.
            actuator: Actuator definition from the universal schema.
            joints_by_name: Optional mapping of joint name → ``Joint``. Used to
                set ``ctrlrange`` on *position* actuators when the actuator has no
                explicit ``control_range``: from the joint's limits for a bounded
                joint, or from ``home ± span`` for a continuous joint (which
                carries no limits). Without a ``ctrlrange`` MuJoCo defaults it to
                ``[0, 0]`` which freezes position-actuated joints and collapses
                the viewer Ctrl-panel slider to zero width.

                Velocity actuators deliberately do NOT consult this: their
                ``ctrl`` is a rad/s setpoint, so a joint-angle band is the wrong
                quantity entirely (see the velocity branch below).
        """
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
        act = ET.SubElement(
            actuator_elem, act_tag, name=actuator.name, joint=actuator.joint
        )

        if hasattr(actuator, "control_range") and actuator.control_range:
            ctrl_min, ctrl_max = actuator.control_range
            act.set("ctrlrange", f"{ctrl_min} {ctrl_max}")
        elif act_tag == "velocity":
            # A velocity servo's ``ctrl`` is a rad/s setpoint, so its ctrlrange is a
            # SPEED band; the joint's positional limits say nothing about it. Sharing
            # the position branch below handed a drive wheel its continuous joint's
            # ``home ± π`` band, capping it at π rad/s — 0.17 m/s on a 53 mm wheel,
            # with the robot merely crawling and nothing logged anywhere.
            if actuator.max_velocity:
                act.set(
                    "ctrlrange", f"{-actuator.max_velocity} {actuator.max_velocity}"
                )
            else:
                # Left unlimited rather than invented: ``compiler autolimits="true"``
                # (set above) reads an absent ctrlrange as "no limit", so this is a
                # loose actuator, not a frozen one. Warned because an unbounded wheel
                # command reaches the contact solver as a skid.
                logger.warning(
                    "Velocity actuator %r on joint %r declares neither "
                    "control_range nor max_velocity; exporting it without a "
                    "ctrlrange (unlimited rad/s). Declare max_velocity to bound it.",
                    actuator.name,
                    actuator.joint,
                )
        elif act_tag == "position" and joints_by_name:
            # Resolve a finite ctrlrange so MuJoCo does not default it to [0, 0]
            # (which clips every ctrl command to zero and freezes the joint).
            joint = joints_by_name.get(actuator.joint)
            limits = joint.limits if joint is not None else None
            if joint is not None and joint.type == JointType.CONTINUOUS:
                # Continuous joint: the joint itself is unlimited, so give the
                # actuator a finite band centered on the home pose
                # (``home ± span``). Checked BEFORE the limits branch so any
                # stray positional limits a continuous joint happens to carry are
                # ignored — the joint must not be lied about.
                home = joint.home_position or 0.0
                span = CONTINUOUS_ACTUATOR_CTRL_SPAN
                act.set("ctrlrange", f"{home - span} {home + span}")
            elif (
                limits is not None
                and limits.lower is not None
                and limits.upper is not None
            ):
                # Bounded joint: match the actuator's control range to the
                # joint's allowed range.
                act.set("ctrlrange", f"{limits.lower} {limits.upper}")

        if hasattr(actuator, "force_range") and actuator.force_range:
            force_min, force_max = actuator.force_range
            force_range = f"{force_min} {force_max}"
            act.set("forcerange", force_range)
        elif hasattr(actuator, "max_torque") and actuator.max_torque is not None:
            # If max_torque is specified but no force_range, use symmetric range
            max_torque = actuator.max_torque
            act.set("forcerange", f"-{max_torque} {max_torque}")

        if actuator.gear_ratio is not None:
            if act_tag in ("position", "velocity"):
                import warnings

                warnings.warn(
                    f"Actuator '{actuator.name}' has gear_ratio={actuator.gear_ratio} and "
                    f"act_tag='{act_tag}'. Emitting gear={actuator.gear_ratio}, but note that for "
                    "position/velocity actuators this divides the controllable joint range by the "
                    "gear ratio (effective max qpos = ctrlrange_max / gear). Ensure kp/kv/forcerange "
                    "are in motor-shaft space, or set gear_ratio=None if they are in joint "
                    "(output-shaft) space.",
                    UserWarning,
                    stacklevel=2,
                )
            act.set("gear", str(actuator.gear_ratio))

        if actuator.kp is not None and act_tag == "position":
            act.set("kp", str(actuator.kp))

        ext = getattr(actuator, "extensions", None) or {}
        timeconst = ext.get("timeconst")
        dampratio = ext.get("dampratio")
        if timeconst is not None:
            act.set("timeconst", str(timeconst))
        if dampratio is not None:
            act.set("dampratio", str(dampratio))

        if actuator.kd is not None and act_tag in ["position", "velocity"]:
            if timeconst is None and dampratio is None:
                act.set("kv", str(actuator.kd))

    def _add_sensor(self, sensor_elem: ET.Element, sensor: Sensor) -> None:
        """Add one or more ``<sensor>`` entries for a schema sensor.

        A schema sensor and a MuJoCo sensor are not 1:1. ``imu`` is a *device*
        that MuJoCo models as several primitives on one site, so it expands to
        ``gyro`` + ``accelerometer`` — which is exactly the pair
        ``cyberwave-sim/sensors/imu.py`` looks for in ``sensordata``. Types that
        MuJoCo has no primitive for (lidar, radar, point clouds) are sampled at
        runtime with ``mj_ray`` instead and are skipped here without a warning:
        they are not broken, just not MJCF-native.
        """
        sensor_type = (sensor.type or "").lower()

        if sensor_type in _RUNTIME_SAMPLED_SENSOR_TYPES:
            logger.debug(
                "Sensor %s (%s) is sampled at runtime via mj_ray; no MJCF sensor "
                "element is emitted.",
                sensor.name,
                sensor.type,
            )
            return

        tags = _MJCF_SENSOR_TAGS.get(sensor_type)
        if not tags:
            logger.warning(
                "Sensor %s with type %s is not supported.", sensor.name, sensor.type
            )
            return

        # A site-attached sensor whose site was never emitted would compile to a
        # dangling reference and MuJoCo would reject the entire model, so drop the
        # sensor instead of taking the whole scene down with it. Reachable when
        # parent_link names no exported link at all (a typo, or "world").
        # ``None`` means the caller reached this helper without going through
        # ``export``, so there is nothing to check against.
        emitted_sites = getattr(self, "_emitted_sites", None)
        if (
            emitted_sites is not None
            and _needs_site(sensor.type)
            and site_name_for_sensor(sensor) not in emitted_sites
        ):
            logger.warning(
                "Sensor %s declares parent_link %r, which matched no exported body; "
                "skipping it. A sensor referencing a missing site makes MuJoCo "
                "reject the whole model.",
                sensor.name,
                sensor.parent_link,
            )
            return

        for tag in tags:
            # Multi-primitive devices (imu -> gyro + accelerometer) need distinct
            # names; a single-primitive sensor keeps its own name so downstream
            # lookups by sensor id keep working.
            name = sensor.name if len(tags) == 1 else f"{sensor.name}_{tag}"
            attrs = {"name": name}

            if tag in ("jointpos", "jointvel"):
                joint_name = (
                    sensor.parameters.get("joint")
                    if hasattr(sensor, "parameters")
                    else None
                )
                if joint_name:
                    attrs["joint"] = joint_name
            elif _needs_site(sensor.type):
                attrs["site"] = site_name_for_sensor(sensor)

            sens_elem = ET.SubElement(sensor_elem, tag, **attrs)
            self._set_sensor_cutoff(sens_elem, sensor, tag)
            self._set_sensor_noise(sens_elem, sensor)

    def _set_sensor_cutoff(
        self, sens_elem: ET.Element, sensor: Sensor, tag: str
    ) -> None:
        """Set MuJoCo's ``cutoff``, which clamps the sensor *value*.

        It is not a sample rate. ``cutoff`` was previously set from
        ``update_rate``, so a 12.5 Hz scanner told MuJoCo to clamp its readings at
        12.5 (metres, for a rangefinder). Sampling cadence has no MJCF
        representation at all — it is honoured by the simulator's egress budget
        (``range_stream_frequency_hz``) from the schema's own ``update_rate``.
        """
        params = sensor.parameters or {}
        explicit = params.get("cutoff")
        if isinstance(explicit, (int, float)) and not isinstance(explicit, bool):
            sens_elem.set("cutoff", str(float(explicit)))
            return
        if tag == "rangefinder":
            # For a ranged sensor the meaningful clamp is its maximum range.
            for key in ("max_range", "range_max"):
                value = params.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    sens_elem.set("cutoff", str(float(value)))
                    return

    def _set_sensor_noise(self, sens_elem: ET.Element, sensor: Sensor) -> None:
        """Carry a declared noise stddev onto the MJCF sensor.

        Lives in ``parameters`` rather than the schema's ``noise`` field because
        that field cannot be deserialized: it is typed
        ``dict[str, float | list[float]]`` and cattrs has no structure hook for
        that union, so any populated value makes ``CommonSchema.from_dict`` raise.
        """
        params = sensor.parameters or {}
        for key in ("noise_stddev", "noise_std_m"):
            value = params.get(key)
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and value > 0
            ):
                sens_elem.set("noise", str(float(value)))
                return

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
            ET.SubElement(
                contact, "exclude", {"body1": exclude.body1, "body2": exclude.body2}
            )

        # Add explicit geom pairs
        for pair in config.pairs:
            pair_attrs = {"geom1": pair.geom1, "geom2": pair.geom2}

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

    def _add_equality_section(self, mujoco: ET.Element, schema: CommonSchema) -> None:
        """Add <equality> joint couplings for mimic joints.

        A URDF ``<mimic>`` (``Joint.mimic``) declares ``slave = offset +
        multiplier * driver``. Without the corresponding MJCF
        ``<equality><joint .../>`` element the slave joint is left free and
        undamped — e.g. the Robotiq 2F-85 finger linkage joints swing as
        undamped pendulums, never settle, and accumulate energy that shakes the
        whole arm (and its wrist camera). MJCF expresses the same coupling as
        ``q_joint1 = polycoef(q_joint2)`` with ``polycoef = offset multiplier
        0 0 0``. Solver params are left at MuJoCo defaults so the constraint is
        stable across the exported timesteps (plant 0.002 s, RL 0.005 s).
        """
        # Only scalar (hinge/slide) joints can appear in an <equality><joint>.
        scalar_joint_names = {
            j.name
            for j in schema.joints
            if j.type not in _EQUALITY_INELIGIBLE_JOINT_TYPES
        }
        mimic_joints = []
        for j in schema.joints:
            if j.mimic is None or j.type in _EQUALITY_INELIGIBLE_JOINT_TYPES:
                continue
            if j.mimic.joint == j.name:
                # Self-referential mimic (a slave that drives itself) would emit
                # a degenerate <equality joint1==joint2> that over-constrains the
                # joint. Drop it loudly.
                logger.warning(
                    "Skipping self-referential mimic on joint %r "
                    "(mimic.joint points at itself).",
                    j.name,
                )
                continue
            if j.mimic.joint not in scalar_joint_names:
                # The driver reference does not resolve to a scalar joint: it is
                # missing, pruned/renamed, or a FIXED/FLOATING/SPHERICAL joint
                # that has no scalar DOF. Emitting the equality anyway would
                # dangle-reference a non-existent <joint> and fail MuJoCo
                # compilation, so drop it and surface the miswiring instead.
                logger.warning(
                    "Skipping mimic coupling for joint %r: driver joint %r is "
                    "missing or not a scalar (hinge/slide) joint; the slave is "
                    "left uncoupled.",
                    j.name,
                    j.mimic.joint,
                )
                continue
            mimic_joints.append(j)
        if not mimic_joints:
            return

        equality = ET.SubElement(mujoco, "equality")
        for joint in mimic_joints:
            ET.SubElement(
                equality,
                "joint",
                {
                    "joint1": joint.name,
                    "joint2": joint.mimic.joint,
                    "polycoef": (
                        f"{joint.mimic.offset} {joint.mimic.multiplier} 0 0 0"
                    ),
                },
            )

    def _write_pretty_xml(self, root: ET.Element, output_path: str | Path) -> None:
        """Write XML with pretty formatting."""
        rough_string = ET.tostring(root, "utf-8")
        reparsed = minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent="  ")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(pretty_xml)


__all__ = ["MJCFExporter"]
