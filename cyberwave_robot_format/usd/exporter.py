# Copyright [2025] Cyberwave
# Licensed under the Apache License, Version 2.0

"""Exporter from :class:`CommonSchema` to OpenUSD (``.usda``/``.usd``/``.usdc``/``.usdz``).

Three principles govern the mapping; they are what make the round trip both
lossless *and* useful to third-party USD consumers (Isaac Sim, Omniverse,
usdview). See :mod:`cyberwave_robot_format.usd` for the full statement.

1. Where UsdPhysics has a faithful home for a field, that is where it goes.
   Where the native encoding is lossy (almost all of UsdPhysics is ``float32``),
   ambiguous (a limit-less ``PhysicsRevoluteJoint`` could be ``REVOLUTE`` or
   ``CONTINUOUS``) or simply absent (mimic, armature, jerkmax), a
   ``cyberwave:``-namespaced double-precision sidecar is written *as well*, and
   the parser treats the sidecar as authoritative.
2. ``None`` means unauthored, in both directions.
3. Prim names are sanitized forward only; a ``cyberwave:name`` sidecar carries
   the original whenever sanitization changed it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from cyberwave_robot_format.core import BaseExporter
from cyberwave_robot_format.schema import (
    Actuator,
    Collision,
    CollisionConfig,
    CommonSchema,
    ContactSurface,
    Geometry,
    GeometryType,
    Joint,
    JointType,
    Link,
    Material,
    Physics,
    Pose,
    Quaternion,
    Scene,
    Sensor,
    Visual,
)
from cyberwave_robot_format.usd import common as c
from cyberwave_robot_format.usd import frames as f
from cyberwave_robot_format.usd import physx as px
from cyberwave_robot_format.usd import semantics as sem
from cyberwave_robot_format.usd import tokens as t

logger = logging.getLogger(__name__)

_IDENTITY = np.eye(4)


class USDExporter(BaseExporter):
    """Exporter for OpenUSD stages carrying full physics and articulation data.

    Args:
        bake_meshes: When ``True``, mesh geometry referenced by
            ``Geometry.filename`` is resolved from disk and its points authored
            into the ``UsdGeomMesh`` prim. Off by default so that export stays
            pure, deterministic and filesystem-free — ``package://`` URIs are
            frequently unresolvable at export time, and the reference itself is
            preserved either way.
        mesh_search_paths: Extra directories handed to
            :func:`cyberwave_robot_format.mesh.resolution.resolve_mesh_uri` when
            baking.
    """

    def __init__(
        self,
        bake_meshes: bool = False,
        mesh_search_paths: list[str] | None = None,
    ) -> None:
        self.bake_meshes = bake_meshes
        self.mesh_search_paths = mesh_search_paths or []

    def get_extension(self) -> str:
        """Return the default file extension (the human-readable encoding)."""
        return "usda"

    def export(self, schema: CommonSchema, output_path: str | Path) -> None:
        """Export ``schema`` to ``output_path``.

        The encoding follows the suffix, using USD's own defaults: ``.usda`` is
        ASCII, ``.usdc`` is crate (binary), ``.usd`` is crate, and ``.usdz`` is a
        zero-compression package built from a temporary ``.usda``.
        """
        pxr = c.require_pxr()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        stage = self._build_stage(schema)

        if output_path.suffix.lower() == ".usdz":
            import tempfile

            with tempfile.TemporaryDirectory() as tmp_dir:
                staged = Path(tmp_dir) / f"{output_path.stem}.usda"
                stage.GetRootLayer().Export(str(staged))
                packaged = pxr.UsdUtils.CreateNewUsdzPackage(
                    pxr.Sdf.AssetPath(str(staged)), str(output_path)
                )
                if not packaged and not output_path.exists():
                    raise OSError(f"Failed to write USDZ package to {output_path}")
                if not packaged:
                    # A usdz is meant to be self-contained, so USD reports failure
                    # when a referenced asset cannot be bundled — which is the norm
                    # for robot models whose meshes are `package://` URIs. The stage
                    # itself is written and complete, and the schema data lives in
                    # attributes rather than in the unresolved files, so this is a
                    # warning rather than a hard error.
                    logger.warning(
                        "USDZ package %s was written but some referenced assets "
                        "(meshes/textures) could not be resolved and bundled; the "
                        "schema data is intact but those files are missing from the "
                        "package.",
                        output_path,
                    )
        else:
            stage.GetRootLayer().Export(str(output_path))

        logger.info("Exported USD to: %s", output_path)

    def export_to_string(self, schema: CommonSchema) -> str:
        """Export ``schema`` as ``.usda`` text.

        The primary entry point for tests and for callers that want the bytes
        without touching the filesystem.
        """
        # The stage must stay alive in a local for the duration of the export:
        # a layer handle obtained from a temporary is a dangling TfWeakPtr once
        # the stage is released, which crashes the process rather than raising.
        stage = self._build_stage(schema)
        return stage.GetRootLayer().ExportToString()

    def build_stage(self, schema: CommonSchema) -> tuple[Any, AssetPaths]:
        """Build the in-memory stage and report where each entity was authored.

        For callers composing assets into a larger stage (see
        :func:`cyberwave_robot_format.usd.scene.export_scene_usd`),
        which need the prim paths without re-deriving the exporter's naming.
        """
        builder = self._builder(schema)
        return builder.stage, builder.asset_paths()

    # ------------------------------------------------------------------
    # Stage construction
    # ------------------------------------------------------------------

    def _build_stage(self, schema: CommonSchema) -> Any:
        return self._builder(schema).stage

    def _builder(self, schema: CommonSchema) -> _StageBuilder:
        pxr = c.require_pxr()
        stage = pxr.Usd.Stage.CreateInMemory("cyberwave.usda")

        # Authored explicitly rather than left to USD's defaults: USD falls back to
        # centimetres and Y-up when a stage says nothing, so leaving these unset
        # would make our own output ambiguous about units and orientation.
        pxr.UsdGeom.SetStageUpAxis(stage, pxr.UsdGeom.Tokens.z)
        pxr.UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        pxr.UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)

        if schema.metadata.units != "SI":
            # The stage is authored as metres and kilogrammes unconditionally, since
            # that is what every number in CommonSchema is documented to be. A schema
            # claiming otherwise is not converted — say so rather than emitting a
            # stage whose declared units silently disagree with its data.
            logger.warning(
                "Schema %r declares units=%r, but the USD stage is authored as SI "
                "(metersPerUnit=1, kilogramsPerUnit=1) and the values are written "
                "unconverted. The declared value is preserved in the metadata.",
                schema.metadata.name,
                schema.metadata.units,
            )

        # Imported here, not at module scope: cyberwave_robot_format.__init__
        # imports this module, and __version__ is only bound after those imports.
        from cyberwave_robot_format import __version__

        stage.GetRootLayer().comment = (
            f"Generated by cyberwave-robot-format {__version__} from CommonSchema"
        )

        builder = _StageBuilder(
            stage=stage,
            schema=schema,
            bake_meshes=self.bake_meshes,
            mesh_search_paths=self.mesh_search_paths,
        )
        builder.build()
        return builder


class AssetPaths:
    """Prim paths an export authored, relative to the asset's root prim.

    Plain strings (``"Links/base"``), so they can be re-rooted under any prim a
    composing stage references the asset into.
    """

    def __init__(self) -> None:
        self.root_name: str = ""
        self.links: dict[str, str] = {}
        self.joints: dict[str, str] = {}
        self.sensors: dict[str, str] = {}
        self.rotors: dict[str, str] = {}
        self.materials: dict[str, str] = {}
        # ``(subject, message)`` per ``extensions["physx"]`` entry that was skipped.
        self.physx_problems: list[tuple[str, str]] = []


class _StageBuilder:
    """Per-export mutable state, so :class:`USDExporter` itself stays reusable."""

    def __init__(
        self,
        stage: Any,
        schema: CommonSchema,
        bake_meshes: bool,
        mesh_search_paths: list[str],
    ) -> None:
        self.pxr = c.require_pxr()
        self.stage = stage
        self.schema = schema
        self.bake_meshes = bake_meshes
        self.mesh_search_paths = mesh_search_paths
        self.namer = c.PrimNamer()

        self.root_path: Any = None
        self.links_scope: Any = None
        self.joints_scope: Any = None
        self.physics_materials_scope: Any = None
        self.link_paths: dict[str, Any] = {}
        self.joint_paths: dict[str, Any] = {}
        self.material_paths: dict[str, Any] = {}
        self._physics_materials: dict[tuple, Any] = {}
        self._named_materials: dict[str, Material] = {}
        self._document_material_names: frozenset[str] = frozenset()
        self._shadowed_warned: set[str] = set()
        self._group_members: dict[str, list[Any]] = {}
        self.sensor_paths: dict[str, Any] = {}
        self.rotor_paths: dict[str, Any] = {}
        self.dof_indices = sem.dof_indices(schema)
        self.physx_problems: list[tuple[str, str]] = []
        # Whether a zero link mass means "massless" rather than "unknown";
        # see _write_mass.
        self._schema_authors_mass = any(link.mass > 0.0 for link in schema.links)

    def asset_paths(self) -> AssetPaths:
        paths = AssetPaths()
        paths.root_name = self.root_path.name
        for target, source in (
            (paths.links, self.link_paths),
            (paths.joints, self.joint_paths),
            (paths.sensors, self.sensor_paths),
            (paths.rotors, self.rotor_paths),
            (paths.materials, self.material_paths),
        ):
            for name, path in source.items():
                target[name] = self._relative(path, self.root_path)
        paths.physx_problems = list(self.physx_problems)
        return paths

    # -- helpers -------------------------------------------------------

    def _define(self, schema_type: Any, path: Any) -> Any:
        return schema_type.Define(self.stage, path)

    def _scope(self, parent: Any, name: str) -> Any:
        return self.pxr.UsdGeom.Scope.Define(self.stage, parent.AppendChild(name)).GetPrim().GetPath()

    def _child_path(self, parent: Any, schema_name: str) -> tuple[Any, str]:
        prim_name = self.namer.name_for(str(parent), schema_name)
        return parent.AppendChild(prim_name), prim_name

    def _author_name_sidecar(self, prim: Any, schema_name: str) -> None:
        """Author ``cyberwave:name`` only when the prim name lost information.

        For entities whose schema name is mandatory (links, joints, actuators,
        sensors, contacts, groups) the prim name is a perfectly good carrier when
        it survived sanitization unchanged, so the sidecar stays out of the
        ``.usda`` in the common case.
        """
        if self.namer.needs_name_sidecar(prim.GetName(), schema_name):
            c.write_string(prim, t.A_NAME, schema_name)

    def _author_optional_name(self, prim: Any, schema_name: str | None) -> None:
        """Author ``cyberwave:name`` for an entity whose name may legitimately be ``None``.

        ``Visual.name`` and ``Collision.name`` are optional, so the parser has to
        read absence of the sidecar as ``None`` — which means a *named* visual must
        always author it, even when the prim name already matches. Falling back to
        the prim name instead would turn every unnamed visual into one called
        ``visual_0``.
        """
        if schema_name is not None:
            c.write_string(prim, t.A_NAME, schema_name)

    def _relative(self, prim_path: Any, base_path: Any) -> str:
        return str(prim_path.MakeRelativePath(base_path))

    def _set_transform(self, prim: Any, matrix: np.ndarray) -> None:
        """Author ``xformOp:transform`` unless the transform is the identity."""
        if np.allclose(matrix, _IDENTITY, rtol=0.0, atol=0.0):
            return
        self.pxr.UsdGeom.Xformable(prim).AddTransformOp().Set(
            c.np_to_gf_matrix(matrix)
        )

    def _set_local_pose(self, prim: Any, pose: Pose) -> None:
        """Retain authored components as well as the native USD transform."""
        self._set_transform(prim, f.pose_to_matrix(pose))
        # Pure translations already round-trip exactly. Keep those stages lean
        # and byte-compatible; only rotations need their source components.
        if pose.orientation != Quaternion():
            c.write_vector3(prim, t.A_POSE_POSITION, pose.position)
            c.write_quaternion(prim, t.A_POSE_ORIENTATION, pose.orientation)

    def _set_scale(self, prim: Any, scale: tuple[float, float, float]) -> None:
        """Author ``xformOp:scale`` after any transform op.

        Op order matters: with ``["xformOp:transform", "xformOp:scale"]`` USD
        applies the scale to the geometry *first* and then places it by the
        transform, which is the nesting a visual/collision pose implies. The
        reverse order would scale the offset too.
        """
        self.pxr.UsdGeom.Xformable(prim).AddScaleOp(
            self.pxr.UsdGeom.XformOp.PrecisionDouble
        ).Set(self.pxr.Gf.Vec3d(*[float(v) for v in scale]))

    # -- top level -----------------------------------------------------

    def build(self) -> None:
        pxr = self.pxr
        schema = self.schema

        root_name = c.sanitize_prim_name(self.schema.metadata.name)
        self.root_path = pxr.Sdf.Path(f"/{root_name}")
        root = self._define(pxr.UsdGeom.Xform, self.root_path)
        root_prim = root.GetPrim()
        self.stage.SetDefaultPrim(root_prim)

        # The whole schema is one reduced-coordinate articulation.
        pxr.UsdPhysics.ArticulationRootAPI.Apply(root_prim)
        self._write_physx(
            root_prim,
            px.ARTICULATION_APIS,
            px.physx_extension(schema.extensions),
            schema.metadata.name,
        )

        if root_name != schema.metadata.name:
            c.write_string(root_prim, t.A_NAME, schema.metadata.name)
        c.write_custom_data_json(
            root_prim, t.CD_METADATA, _metadata_to_dict(schema.metadata)
        )
        c.write_custom_data_json(root_prim, t.CD_EXTENSIONS, schema.extensions)

        self.links_scope = self._scope(self.root_path, t.SCOPE_LINKS)
        self.joints_scope = self._scope(self.root_path, t.SCOPE_JOINTS)

        document_materials = self._write_named_materials()
        order = self._write_links()
        order["materials"] = document_materials
        order["joints"] = self._write_joints()
        order["actuators"], driven_joints = self._write_actuators()
        self._write_passive_drives(driven_joints)
        self._write_rotors()
        order["contacts"] = self._write_contacts()
        c.write_custom_data_json(root_prim, t.CD_ORDER, _pruned(order))

        self._write_collision_config()
        self._write_render_scope(schema.scene)
        self._write_physics_scene(schema.physics)

    # -- links ---------------------------------------------------------

    def _write_links(self) -> dict[str, list[str]]:
        world_transforms = f.accumulated_world_transforms(self.schema)
        sensors_by_link: dict[str, list[Sensor]] = {}
        for sensor in self.schema.sensors:
            sensors_by_link.setdefault(sensor.parent_link, []).append(sensor)

        link_order: list[str] = []
        # Sensors are written while walking the links, but their recorded order has
        # to follow `schema.sensors`, not link iteration — otherwise a world-parented
        # sensor listed first in the schema would come back last.
        sensor_paths: dict[int, Any] = {}

        for link in self.schema.links:
            path, _ = self._child_path(self.links_scope, link.name)
            prim = self._define(self.pxr.UsdGeom.Xform, path).GetPrim()
            self.link_paths[link.name] = path
            link_order.append(self._relative(path, self.root_path))

            self._author_name_sidecar(prim, link.name)
            c.write_custom_data_value(prim, t.CD_LINK_NAME, link.name)
            self.pxr.UsdPhysics.RigidBodyAPI.Apply(prim)
            self._write_mass(prim, link)
            self._set_transform(prim, world_transforms.get(link.name, _IDENTITY))
            c.write_custom_data_json(prim, t.CD_EXTENSIONS, link.extensions)
            c.write_usd_passthrough(prim, t.USD_PASSTHROUGH_LINK, link.extensions)
            self._write_physx(
                prim, px.LINK_APIS, px.physx_extension(link.extensions), link.name
            )

            member_order = {
                "visuals": self._write_visuals(path, link),
                "collisions": self._write_collisions(path, link),
            }
            c.write_custom_data_json(prim, t.CD_ORDER, _pruned(member_order))

            link_sensors = sensors_by_link.get(link.name, [])
            if link_sensors:
                # A dedicated scope, rather than parenting sensors straight to the
                # link: it keeps them from colliding with the Visuals/Collisions
                # scope names, and mirrors how every other collection is laid out.
                sensors_scope = self._scope(path, t.SCOPE_SENSORS)
                for sensor in link_sensors:
                    sensor_paths[id(sensor)] = self._write_sensor(
                        sensors_scope, sensor, nested_under_link=True
                    )

        # Sensors whose parent_link is "world" (or dangling) have no link prim to
        # live under, so they get a top-level scope.
        orphans = [
            sensor
            for name, sensors in sensors_by_link.items()
            if name not in self.link_paths
            for sensor in sensors
        ]
        if orphans:
            scope = self._scope(self.root_path, t.SCOPE_SENSORS)
            for sensor in orphans:
                sensor_paths[id(sensor)] = self._write_sensor(
                    scope, sensor, nested_under_link=False
                )

        sensor_order = [
            self._relative(sensor_paths[id(sensor)], self.root_path)
            for sensor in self.schema.sensors
            if id(sensor) in sensor_paths
        ]
        return {"links": link_order, "sensors": sensor_order}

    def _write_mass(self, prim: Any, link: Link) -> None:
        """Apply ``UsdPhysicsMassAPI`` only when the link has inertial data.

        Skipping it for a default-inertial link matters: ``physics:mass = 0`` is
        UsdPhysics' request to *derive* mass from density and geometry, the
        opposite of what a schema mass of ``0.0`` means. Only a schema that
        authors no inertial anywhere leaves a link with collision geometry to
        that derivation -- there is nothing better to go on. Once another link
        carries mass, a zero-mass link is a deliberate massless one (Spot's
        ``body``, whose 32 kg sit on the fixed ``base_link``; a frame such as
        a camera optical frame or a tool tip): deriving it from its collider
        added 31 kg to Spot in PhysX, which MuJoCo -- whose exporter gives
        every zero-mass body a stand-in -- never sees. Those links get the small
        stand-in from :meth:`_write_massless_fallback` instead.
        """
        orientation = link.inertial_orientation
        if (orientation.x, orientation.y, orientation.z, orientation.w) != (
            0.0,
            0.0,
            0.0,
            1.0,
        ):
            # Sidecar only, and only when rotated, so unrotated links stay
            # byte-identical: the native MassAPI below gets the link-frame tensor.
            c.write_quaternion(prim, t.A_LINK_INERTIAL_ORIENTATION, orientation)
        inertia = link.inertia
        has_inertia = any(
            value != 0.0
            for value in (
                inertia.ixx,
                inertia.iyy,
                inertia.izz,
                inertia.ixy,
                inertia.ixz,
                inertia.iyz,
            )
        )
        has_com = (
            link.center_of_mass.x != 0.0
            or link.center_of_mass.y != 0.0
            or link.center_of_mass.z != 0.0
        )
        if link.mass == 0.0 and not has_inertia and not has_com:
            if not link.collisions or self._schema_authors_mass:
                self._write_massless_fallback(prim, link)
            return

        mass_api = self.pxr.UsdPhysics.MassAPI.Apply(prim)
        if link.mass != 0.0:
            mass_api.CreateMassAttr(float(link.mass))
            c.write_double(prim, t.A_LINK_MASS, link.mass)
        if has_com:
            mass_api.CreateCenterOfMassAttr(
                c.vector3_to_gf(link.center_of_mass, precision="float")
            )
            c.write_vector3(prim, t.A_LINK_CENTER_OF_MASS, link.center_of_mass)
        if has_inertia:
            # physics:principalAxes is relative to the link (prim) frame, so a
            # rotated inertial frame is folded in before the diagonalisation.
            diagonal, principal_axes, _is_exact = f.inertia_to_principal(
                link.inertia_in_link_frame()
            )
            mass_api.CreateDiagonalInertiaAttr(self.pxr.Gf.Vec3f(*diagonal))
            mass_api.CreatePrincipalAxesAttr(
                c.quaternion_to_gf(principal_axes, precision="float")
            )
            # The sidecar is written unconditionally, not only when a
            # decomposition was needed: physics:diagonalInertia is float3 either
            # way, so even a diagonal tensor loses precision natively.
            c.write_float_list(
                prim,
                t.A_LINK_INERTIA,
                [
                    inertia.ixx,
                    inertia.iyy,
                    inertia.izz,
                    inertia.ixy,
                    inertia.ixz,
                    inertia.iyz,
                ],
            )

    def _write_massless_fallback(self, prim: Any, link: Link) -> None:
        """Give a link with nothing to weigh a tiny, explicit inertial.

        Every link is a rigid body here, and UsdPhysics cannot express a
        massless one: with no mass and no collider PhysX invents an inertial
        and misplaces the frame. The native ``MassAPI`` carries the stand-in;
        the sidecars keep the schema's zeros, so import and re-export are
        unchanged, and :data:`t.A_LINK_MASS_FALLBACK` marks the substitution.
        """
        mass_api = self.pxr.UsdPhysics.MassAPI.Apply(prim)
        mass_api.CreateMassAttr(float(t.MASSLESS_LINK_FALLBACK_MASS_KG))
        inertia = float(t.MASSLESS_LINK_FALLBACK_INERTIA_KG_M2)
        mass_api.CreateDiagonalInertiaAttr(self.pxr.Gf.Vec3f(inertia, inertia, inertia))
        mass_api.CreateCenterOfMassAttr(self.pxr.Gf.Vec3f(0.0, 0.0, 0.0))
        c.write_double(prim, t.A_LINK_MASS, link.mass)
        c.write_float_list(prim, t.A_LINK_INERTIA, [0.0] * 6)
        c.write_bool(prim, t.A_LINK_MASS_FALLBACK, True)
        logger.debug(
            "Link %r has no mass, inertia or collision geometry; authored a "
            "%g kg stand-in inertial.",
            link.name,
            t.MASSLESS_LINK_FALLBACK_MASS_KG,
        )

    # -- visuals / collisions ------------------------------------------

    def _write_visuals(self, link_path: Any, link: Link) -> list[str]:
        if not link.visuals:
            return []
        scope = self._scope(link_path, t.SCOPE_VISUALS)
        order: list[str] = []
        for index, visual in enumerate(link.visuals):
            schema_name = visual.name or f"{t.UNNAMED_VISUAL_PREFIX}{index}"
            path, _ = self._child_path(scope, schema_name)
            textured = bool(visual.material and visual.material.texture)
            prim = self._write_geometry_prim(
                path, visual.geometry, visual.pose, textured=textured
            )
            order.append(self._relative(path, link_path))
            self._author_optional_name(prim, visual.name)
            c.write_custom_data_json(prim, t.CD_EXTENSIONS, visual.extensions)
            self._bind_visual_material(prim, visual)
        return order

    def _write_collisions(self, link_path: Any, link: Link) -> list[str]:
        if not link.collisions:
            return []
        scope = self._scope(link_path, t.SCOPE_COLLISIONS)
        order: list[str] = []
        for index, collision in enumerate(link.collisions):
            schema_name = collision.name or f"{t.UNNAMED_COLLISION_PREFIX}{index}"
            path, _ = self._child_path(scope, schema_name)
            prim = self._write_geometry_prim(path, collision.geometry, collision.pose)
            order.append(self._relative(path, link_path))
            self._author_optional_name(prim, collision.name)
            c.write_custom_data_json(prim, t.CD_EXTENSIONS, collision.extensions)
            c.write_usd_passthrough(
                prim, t.USD_PASSTHROUGH_COLLISION, collision.extensions
            )
            self._write_collision_physics(prim, collision)
            self._write_physx(
                prim,
                px.COLLISION_APIS,
                px.physx_extension(collision.extensions),
                f"{link.name}/{schema_name}",
            )
            self._group_members.setdefault(collision.group, []).append(path)
        return order

    def _write_collision_physics(self, prim: Any, collision: Collision) -> None:
        pxr = self.pxr
        pxr.UsdPhysics.CollisionAPI.Apply(prim)
        pxr.UsdGeom.Imageable(prim).CreatePurposeAttr(pxr.UsdGeom.Tokens.guide)

        if collision.geometry and collision.geometry.type == GeometryType.MESH:
            mesh_api = pxr.UsdPhysics.MeshCollisionAPI.Apply(prim)
            approximation = collision.extensions.get(
                "approximation", t.TOK_APPROX_CONVEX_HULL
            )
            mesh_api.CreateApproximationAttr(str(approximation))

        if collision.group != "default":
            c.write_string(prim, t.A_COLLISION_GROUP, collision.group)
        c.write_double(prim, t.A_COLLISION_MU_STATIC, collision.mu_static)
        c.write_double(prim, t.A_COLLISION_MU_DYNAMIC, collision.mu_dynamic)
        c.write_double(prim, t.A_COLLISION_RESTITUTION, collision.restitution)
        # No UsdPhysics equivalent, and per-collision rather than per-material.
        c.write_double(prim, t.A_COLLISION_STIFFNESS, collision.stiffness)
        c.write_double(prim, t.A_COLLISION_DAMPING, collision.damping)

        physics_material = self._physics_material_for(
            collision.mu_static, collision.mu_dynamic, collision.restitution
        )
        if physics_material is not None:
            self._bind_material(prim, physics_material, purpose=t.TOK_MATERIAL_PURPOSE_PHYSICS)

    def _write_geometry_prim(
        self, path: Any, geometry: Geometry | None, pose: Pose, *, textured: bool = False
    ) -> Any:
        pxr = self.pxr
        geometry_type = geometry.type if geometry else GeometryType.UNKNOWN
        prim_type_name = t.GEOMETRY_TYPE_TO_PRIM_TYPE.get(geometry_type, t.PRIM_XFORM)
        prim = self._define(getattr(pxr.UsdGeom, prim_type_name), path).GetPrim()

        self._set_local_pose(prim, pose)
        if geometry is None:
            # No type sidecar at all: its absence is what tells the parser that
            # Visual.geometry / Collision.geometry was None rather than UNKNOWN.
            return prim

        scale = self._write_native_geometry(
            prim, geometry, geometry_type, textured=textured
        )
        if scale is not None:
            self._set_scale(prim, scale)

        # Sidecars are the import authority: they are double precision, they
        # distinguish ELLIPSOID from SPHERE and UNKNOWN from an empty Xform, and
        # they preserve `None` vs a defaulted value.
        c.write_token(prim, t.A_GEOM_TYPE, geometry_type.value)
        c.write_vector3(prim, t.A_GEOM_SIZE, geometry.size)
        c.write_double(prim, t.A_GEOM_RADIUS, geometry.radius)
        c.write_double(prim, t.A_GEOM_LENGTH, geometry.length)
        c.write_string(prim, t.A_GEOM_MESH_FILENAME, geometry.filename)
        c.write_vector3(prim, t.A_GEOM_SCALE, geometry.scale)
        # A separate key from CD_EXTENSIONS: the owning Visual/Collision writes its
        # own extensions to this same prim and would otherwise overwrite these.
        c.write_custom_data_json(prim, t.CD_GEOMETRY_EXTENSIONS, geometry.extensions)
        return prim

    def _write_native_geometry(
        self,
        prim: Any,
        geometry: Geometry,
        geometry_type: GeometryType,
        *,
        textured: bool = False,
    ) -> tuple[float, float, float] | None:
        """Author the native ``UsdGeom`` attributes; return an ``xformOp:scale``."""
        pxr = self.pxr
        size = geometry.size

        if geometry_type == GeometryType.BOX:
            # UsdGeomCube has a single uniform `size`, so a non-uniform box has to
            # carry its dimensions in xformOp:scale over a unit cube.
            pxr.UsdGeom.Cube(prim).CreateSizeAttr(1.0)
            return (size.x, size.y, size.z) if size else None

        if geometry_type == GeometryType.SPHERE:
            pxr.UsdGeom.Sphere(prim).CreateRadiusAttr(float(geometry.radius or 1.0))
            return None

        if geometry_type == GeometryType.ELLIPSOID:
            # A scaled unit sphere is USD's ellipsoid; `size` holds the semi-axes,
            # matching how MJCFExporter reads it.
            pxr.UsdGeom.Sphere(prim).CreateRadiusAttr(1.0)
            return (size.x, size.y, size.z) if size else None

        if geometry_type in (GeometryType.CYLINDER, GeometryType.CAPSULE):
            shape = (
                pxr.UsdGeom.Cylinder(prim)
                if geometry_type == GeometryType.CYLINDER
                else pxr.UsdGeom.Capsule(prim)
            )
            shape.CreateRadiusAttr(float(geometry.radius or 1.0))
            # For a capsule, USD's `height` is the cylindrical section only
            # (total = height + 2*radius) — exactly what `length` means here.
            shape.CreateHeightAttr(float(geometry.length or 1.0))
            # Z-aligned, matching URDF and MJCF, so no corrective rotation.
            shape.CreateAxisAttr(pxr.UsdGeom.Tokens.z)
            return None

        if geometry_type == GeometryType.PLANE:
            plane = pxr.UsdGeom.Plane(prim)
            plane.CreateAxisAttr(pxr.UsdGeom.Tokens.z)
            plane.CreateWidthAttr(float(size.x) if size else t.DEFAULT_PLANE_EXTENT)
            plane.CreateLengthAttr(float(size.y) if size else t.DEFAULT_PLANE_EXTENT)
            plane.CreateDoubleSidedAttr(True)
            return None

        if geometry_type == GeometryType.MESH:
            if self.bake_meshes and geometry.filename:
                self._bake_mesh_points(prim, geometry.filename, uvs=textured)
            if geometry.scale:
                return (geometry.scale.x, geometry.scale.y, geometry.scale.z)
            return None

        return None

    def _resolve_mesh(self, filename: str) -> str | None:
        """Resolve a mesh reference to a readable file, or ``None``.

        ``resolve_mesh_uri`` takes a *single* base directory, while
        ``mesh_search_paths`` is a list, so each entry is tried in turn. It also
        returns its input unchanged when it cannot resolve anything, so the result
        is confirmed to be a real file rather than trusted.

        A filename that already points at a readable file — absolute, or relative
        to the working directory — is used directly and short-circuits the search.
        That is not just an optimization: ``resolve_mesh_uri`` falls back to
        ``rglob`` over the base directory *and its parent*, which would be a
        recursive walk of a large tree for every mesh.
        """
        direct = Path(filename)
        if direct.is_file():
            return str(direct)

        from cyberwave_robot_format.mesh.resolution import resolve_mesh_uri

        for search_path in self.mesh_search_paths:
            resolved = resolve_mesh_uri(filename, Path(search_path))
            if resolved and Path(resolved).is_file():
                return resolved
        return None

    def _bake_mesh_points(self, prim: Any, filename: str, *, uvs: bool = False) -> None:
        """Author mesh points into the prim, best-effort.

        Unresolvable or unloadable meshes degrade to a reference-only placeholder
        rather than failing the export: the asset reference itself is already
        preserved in ``cyberwave:geometry:meshFilename``.

        With *uvs* (a textured visual) the mesh's per-vertex texture coordinates
        are authored as the ``st`` primvar the material's ``UsdPrimvarReader``
        reads; a mesh without them keeps the material's base color.
        """
        resolved = self._resolve_mesh(filename)
        if resolved is None:
            logger.warning(
                "Could not bake mesh %r into USD (not found in %s); the prim keeps "
                "only the asset reference.",
                filename,
                self.mesh_search_paths or "the working directory",
            )
            return
        try:
            import trimesh

            loaded: Any = trimesh.load(resolved, force="mesh")
            vertices = np.asarray(loaded.vertices, dtype=np.float64)
            faces = np.asarray(loaded.faces, dtype=np.int64)
            uv = getattr(loaded.visual, "uv", None) if uvs else None
            uv = np.asarray(uv, dtype=np.float32) if uv is not None else None
        except Exception as exc:
            logger.warning(
                "Could not bake mesh %r into USD (%s); the prim keeps only the "
                "asset reference.",
                filename,
                exc,
            )
            return

        pxr = self.pxr
        mesh = pxr.UsdGeom.Mesh(prim)
        mesh.CreatePointsAttr(pxr.Vt.Vec3fArray([pxr.Gf.Vec3f(*v) for v in vertices]))
        mesh.CreateFaceVertexCountsAttr(pxr.Vt.IntArray([3] * len(faces)))
        mesh.CreateFaceVertexIndicesAttr(
            pxr.Vt.IntArray([int(i) for i in faces.flatten()])
        )
        mesh.CreateSubdivisionSchemeAttr(pxr.UsdGeom.Tokens.none)
        if uv is not None and uv.shape == (len(vertices), 2):
            pxr.UsdGeom.PrimvarsAPI(prim).CreatePrimvar(
                "st", pxr.Sdf.ValueTypeNames.TexCoord2fArray, pxr.UsdGeom.Tokens.vertex
            ).Set(pxr.Vt.Vec2fArray.FromNumpy(uv))
        elif uvs:
            logger.warning(
                "Textured mesh %r has no per-vertex UVs; it renders with the "
                "material's base color.",
                filename,
            )

    # -- materials -----------------------------------------------------

    def _write_named_materials(self) -> list[str]:
        """Define every *named* visual material once, up front, so visuals share them."""
        # A named material's document-level copy is the source of truth, as in the
        # URDF and MJCF exporters; otherwise the first visual's copy defines it.
        document = {
            material.name: material
            for material in getattr(self.schema, "materials", None) or []
            if material.name
        }
        named: dict[str, Material] = {}
        for link in self.schema.links:
            for visual in link.visuals:
                material = visual.material
                if material and material.name and material.name not in named:
                    named[material.name] = document.get(material.name, material)
        self._named_materials = named
        self._document_material_names = frozenset(document)
        # Document materials no visual references are still written (after the
        # referenced ones, so existing output keeps its prim order): the
        # root's cyberwave:order "materials" list is what rebuilds
        # CommonSchema.materials on import.
        to_write = dict(named)
        for name, material in document.items():
            to_write.setdefault(name, material)
        if not to_write:
            return []

        scope = self._scope(self.root_path, t.SCOPE_MATERIALS)
        for name, material in to_write.items():
            path, _ = self._child_path(scope, name)
            prim = self._write_preview_surface(path, material)
            self._author_optional_name(prim, name)
            self.material_paths[name] = path
        return [
            self._relative(self.material_paths[name], self.root_path) for name in document
        ]

    def _bind_visual_material(self, prim: Any, visual: Visual) -> None:
        material = visual.material
        if material is None:
            return
        if material.name and (
            material.name in self._document_material_names
            or self._named_materials.get(material.name) == material
        ):
            self._bind_material(prim, self.material_paths[material.name])
            return
        if material.name and material.name not in self._shadowed_warned:
            # Two distinct materials sharing one name: only the first can own the
            # shared prim, so the rest are authored inline. The name sidecar keeps
            # them named, so the round trip stays exact.
            self._shadowed_warned.add(material.name)
            logger.warning(
                "Material %r is defined more than once with different properties; "
                "the extra definitions are authored inline per visual.",
                material.name,
            )
        # An unnamed (or shadowed) material cannot be shared, so it is authored
        # inline under the gprim.
        inline_path = prim.GetPath().AppendChild(t.INLINE_MATERIAL_NAME)
        inline_prim = self._write_preview_surface(inline_path, material)
        self._author_optional_name(inline_prim, material.name)
        self._bind_material(prim, inline_path)

    def _write_preview_surface(self, path: Any, material: Material) -> Any:
        pxr = self.pxr
        usd_material = self._define(pxr.UsdShade.Material, path)
        prim = usd_material.GetPrim()
        shader = self._define(pxr.UsdShade.Shader, path.AppendChild(t.SHADER_NAME))
        shader.CreateIdAttr("UsdPreviewSurface")

        color = material.color
        if color and len(color) >= 3:
            shader.CreateInput(
                "diffuseColor", pxr.Sdf.ValueTypeNames.Color3f
            ).Set(pxr.Gf.Vec3f(float(color[0]), float(color[1]), float(color[2])))
        if color and len(color) >= 4:
            shader.CreateInput("opacity", pxr.Sdf.ValueTypeNames.Float).Set(
                float(color[3])
            )
        if material.specular and len(material.specular) >= 3:
            shader.CreateInput(
                "specularColor", pxr.Sdf.ValueTypeNames.Color3f
            ).Set(pxr.Gf.Vec3f(*[float(v) for v in material.specular[:3]]))
            shader.CreateInput(
                "useSpecularWorkflow", pxr.Sdf.ValueTypeNames.Int
            ).Set(1)
        if material.emissive and len(material.emissive) >= 3:
            shader.CreateInput(
                "emissiveColor", pxr.Sdf.ValueTypeNames.Color3f
            ).Set(pxr.Gf.Vec3f(*[float(v) for v in material.emissive[:3]]))
        if material.shininess is not None:
            # UsdPreviewSurface has roughness, not shininess. Derived one-way for
            # renderers; cyberwave:material:shininess is the authority.
            roughness = 1.0 - min(1.0, max(0.0, float(material.shininess)))
            shader.CreateInput("roughness", pxr.Sdf.ValueTypeNames.Float).Set(roughness)

        usd_material.CreateSurfaceOutput().ConnectToSource(
            shader.ConnectableAPI(), "surface"
        )

        if material.texture:
            self._write_texture_chain(path, shader, material.texture)

        # Sidecars: `color` is a variable-length list that cannot be rebuilt from
        # a color3f + float pair, and every USD colour input is float32.
        c.write_float_list(prim, t.A_MATERIAL_COLOR, material.color)
        c.write_float_list(prim, t.A_MATERIAL_SPECULAR, material.specular)
        c.write_float_list(prim, t.A_MATERIAL_EMISSIVE, material.emissive)
        c.write_double(prim, t.A_MATERIAL_SHININESS, material.shininess)
        c.write_string(prim, t.A_MATERIAL_TEXTURE, material.texture)
        c.write_custom_data_json(prim, t.CD_EXTENSIONS, material.extensions)
        return prim

    def _write_texture_chain(self, material_path: Any, shader: Any, texture: str) -> None:
        pxr = self.pxr
        reader = self._define(
            pxr.UsdShade.Shader, material_path.AppendChild(t.UV_READER_NAME)
        )
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", pxr.Sdf.ValueTypeNames.Token).Set("st")
        reader_output = reader.CreateOutput("result", pxr.Sdf.ValueTypeNames.Float2)

        sampler = self._define(
            pxr.UsdShade.Shader, material_path.AppendChild(t.TEXTURE_SHADER_NAME)
        )
        sampler.CreateIdAttr("UsdUVTexture")
        sampler.CreateInput("file", pxr.Sdf.ValueTypeNames.Asset).Set(
            pxr.Sdf.AssetPath(texture)
        )
        sampler.CreateInput("st", pxr.Sdf.ValueTypeNames.Float2).ConnectToSource(
            reader_output
        )
        sampler_output = sampler.CreateOutput("rgb", pxr.Sdf.ValueTypeNames.Float3)
        shader.CreateInput("diffuseColor", pxr.Sdf.ValueTypeNames.Color3f).ConnectToSource(
            sampler_output
        )

    def _bind_material(self, prim: Any, material_path: Any, purpose: str = "") -> None:
        pxr = self.pxr
        pxr.UsdShade.MaterialBindingAPI.Apply(prim)
        material = pxr.UsdShade.Material.Get(self.stage, material_path)
        binding = pxr.UsdShade.MaterialBindingAPI(prim)
        if purpose:
            binding.Bind(
                material, pxr.UsdShade.Tokens.weakerThanDescendants, purpose
            )
        else:
            binding.Bind(material)

    def _physics_material_for(
        self,
        mu_static: float | None,
        mu_dynamic: float | None,
        restitution: float | None,
    ) -> Any | None:
        """Return a deduplicated ``UsdPhysicsMaterialAPI`` material path, or ``None``."""
        if mu_static is None and mu_dynamic is None and restitution is None:
            return None
        key = (mu_static, mu_dynamic, restitution)
        cached = self._physics_materials.get(key)
        if cached is not None:
            return cached

        if not self._physics_materials:
            self.physics_materials_scope = self._scope(
                self.root_path, t.SCOPE_PHYSICS_MATERIALS
            )
        index = len(self._physics_materials)
        path = self.physics_materials_scope.AppendChild(
            f"{t.PHYSICS_MATERIAL_PREFIX}{index}"
        )
        material = self._define(self.pxr.UsdShade.Material, path)
        api = self.pxr.UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        if mu_static is not None:
            api.CreateStaticFrictionAttr(float(mu_static))
        if mu_dynamic is not None:
            api.CreateDynamicFrictionAttr(float(mu_dynamic))
        if restitution is not None:
            api.CreateRestitutionAttr(float(restitution))
        self._physics_materials[key] = path
        return path

    # -- joints --------------------------------------------------------

    def _write_joints(self) -> list[str]:
        order: list[str] = []
        for joint in self.schema.joints:
            path, _ = self._child_path(self.joints_scope, joint.name)
            class_name = t.JOINT_TYPE_TO_USDPHYSICS_CLASS.get(joint.type, "Joint")
            usd_joint = self._define(getattr(self.pxr.UsdPhysics, class_name), path)
            prim = usd_joint.GetPrim()
            self.joint_paths[joint.name] = path
            order.append(self._relative(path, self.root_path))

            self._author_name_sidecar(prim, joint.name)
            c.write_custom_data_value(prim, t.CD_JOINT_NAME, joint.name)
            c.write_custom_data_value(
                prim, t.CD_DOF_INDEX, self.dof_indices.get(joint.name)
            )
            self._write_joint_bodies(usd_joint, joint)
            self._write_joint_frames(usd_joint, joint)
            self._write_joint_limits(usd_joint, joint)
            self._write_joint_dof_locks(prim, joint)
            self._write_joint_sidecars(prim, joint)
            c.write_custom_data_json(prim, t.CD_EXTENSIONS, joint.extensions)
            c.write_usd_passthrough(prim, t.USD_PASSTHROUGH_JOINT, joint.extensions)
            # Derived first, so a valid extensions["physx"] value overrides it and
            # an invalid one is skipped without losing the derived value.
            self._write_physx(
                prim, px.JOINT_APIS, px.derived_joint_physx(joint), joint.name
            )
            self._write_physx(
                prim, px.JOINT_APIS, px.physx_extension(joint.extensions), joint.name
            )
        self._write_native_mimics()
        return order

    def _write_native_mimics(self) -> None:
        """Author each mimic Isaac can simulate as a ``NewtonMimicAPI``.

        A second pass, because the leader can come after its follower. A mimic
        NVIDIA cannot express keeps only its sidecars and is passive in Isaac.
        """
        joints = {joint.name: joint for joint in self.schema.joints}
        for joint in self.schema.joints:
            if joint.mimic is None:
                continue
            leader = joints.get(joint.mimic.joint)
            problem = px.native_mimic_problem(joint, leader)
            if problem is not None:
                logger.warning(
                    "Mimic joint %r is not authored natively: %s.", joint.name, problem
                )
                continue
            px.write_newton_mimic(
                self.stage.GetPrimAtPath(self.joint_paths[joint.name]),
                self.joint_paths[joint.mimic.joint],
                joint.mimic,
                angular=joint.type in t.ANGULAR_JOINT_TYPES,
            )

    def _write_physx(
        self, prim: Any, apis: frozenset[str], values: dict[str, Any], subject: str
    ) -> None:
        for problem in px.write_physx(prim, apis, values):
            logger.warning("Skipping PhysX attribute on %r: %s.", subject, problem)
            self.physx_problems.append((subject, problem))

    def _write_joint_bodies(self, usd_joint: Any, joint: Joint) -> None:
        """Wire ``physics:body0``/``body1``.

        A joint parented to ``"world"`` with no ``world`` link in the schema gets
        an **empty** ``body0`` relationship, which is UsdPhysics' own encoding for
        "attached to the world frame" — and inverts cleanly on import.
        """
        parent_path = self.link_paths.get(joint.parent_link)
        child_path = self.link_paths.get(joint.child_link)

        usd_joint.CreateBody0Rel().SetTargets(
            [parent_path] if parent_path is not None else []
        )
        usd_joint.CreateBody1Rel().SetTargets(
            [child_path] if child_path is not None else []
        )

        prim = usd_joint.GetPrim()
        # An empty body0 already means "world" on both sides, so only a genuinely
        # dangling reference needs the verbatim-name sidecar — which exists because
        # CommonSchema.validate() only *reports* dangling links rather than
        # rejecting them, and export must not lose them.
        if parent_path is None and joint.parent_link != t.WORLD_LINK:
            logger.warning(
                "Joint %r references unknown parent link %r; exporting it as "
                "world-attached with the name preserved in a sidecar.",
                joint.name,
                joint.parent_link,
            )
            c.write_string(prim, t.A_JOINT_PARENT_LINK, joint.parent_link)
        if child_path is None:
            logger.warning(
                "Joint %r references unknown child link %r.",
                joint.name,
                joint.child_link,
            )
            c.write_string(prim, t.A_JOINT_CHILD_LINK, joint.child_link)

    def _write_joint_frames(self, usd_joint: Any, joint: Joint) -> None:
        local_pos0, local_rot0, local_pos1, local_rot1 = f.joint_frames_for_export(
            joint.pose, joint.axis
        )
        usd_joint.CreateLocalPos0Attr(c.vector3_to_gf(local_pos0, precision="float"))
        usd_joint.CreateLocalRot0Attr(c.quaternion_to_gf(local_rot0))
        usd_joint.CreateLocalPos1Attr(c.vector3_to_gf(local_pos1, precision="float"))
        usd_joint.CreateLocalRot1Attr(c.quaternion_to_gf(local_rot1))
        if joint.type in t.JOINT_TYPES_WITH_AXIS_ATTR:
            usd_joint.CreateAxisAttr(t.REF_AXIS_TOKEN)

    def _write_joint_limits(self, usd_joint: Any, joint: Joint) -> None:
        limits = joint.limits
        if limits is None:
            return
        is_angular = joint.type in t.ANGULAR_JOINT_TYPES

        # CONTINUOUS deliberately authors no native limits: an unauthored
        # -inf/+inf pair *is* "unlimited" in UsdPhysics, and cyberwave:joint:type
        # is what tells it apart from a limit-less REVOLUTE.
        if joint.type in (JointType.REVOLUTE, JointType.PRISMATIC):
            if limits.lower is not None:
                usd_joint.CreateLowerLimitAttr(_to_usd_angle(limits.lower, is_angular))
            if limits.upper is not None:
                usd_joint.CreateUpperLimitAttr(_to_usd_angle(limits.upper, is_angular))

    def _write_joint_dof_locks(self, prim: Any, joint: Joint) -> None:
        """Lock the DOFs a ``PLANAR``/``UNIVERSAL`` joint does not have.

        ``FLOATING`` is intentionally absent: a bare ``PhysicsJoint`` with no
        ``LimitAPI`` instance already has all six DOFs free. It gets
        ``physics:excludeFromArticulation`` instead so PhysX simulates the child
        as a free rigid body rather than an articulation link.
        """
        pxr = self.pxr
        if joint.type == JointType.FLOATING:
            pxr.UsdPhysics.Joint(prim).CreateExcludeFromArticulationAttr(True)
            return
        for dof in t.LOCKED_DOFS_FOR_JOINT_TYPE.get(joint.type, ()):
            limit_api = pxr.UsdPhysics.LimitAPI.Apply(prim, dof)
            limit_api.CreateLowAttr(t.LOCK_LOW)
            limit_api.CreateHighAttr(t.LOCK_HIGH)

    def _write_joint_sidecars(self, prim: Any, joint: Joint) -> None:
        c.write_token(prim, t.A_JOINT_TYPE, joint.type.value)
        c.write_vector3(prim, t.A_JOINT_AXIS, joint.axis)
        c.write_vector3(prim, t.A_JOINT_POSE_POSITION, joint.pose.position)
        c.write_quaternion(prim, t.A_JOINT_POSE_ORIENTATION, joint.pose.orientation)
        c.write_double(prim, t.A_JOINT_HOME_POSITION, joint.home_position)
        c.write_json_attr(prim, t.A_JOINT_SAFETY_CONTROLLER, joint.safety_controller)
        c.write_json_attr(prim, t.A_JOINT_CALIBRATION, joint.calibration)

        limits = joint.limits
        # A presence marker is the only way to tell `limits=None` from a
        # JointLimits() whose every field is None.
        c.write_bool(prim, t.A_JOINT_HAS_LIMITS, True if limits is not None else None)
        if limits is not None:
            c.write_double(prim, t.A_JOINT_LIMIT_LOWER, limits.lower)
            c.write_double(prim, t.A_JOINT_LIMIT_UPPER, limits.upper)
            c.write_double(prim, t.A_JOINT_LIMIT_EFFORT, limits.effort)
            c.write_double(prim, t.A_JOINT_LIMIT_VELOCITY, limits.velocity)
            c.write_double(prim, t.A_JOINT_LIMIT_ACCELERATION, limits.acceleration)
            c.write_double(prim, t.A_JOINT_LIMIT_JERKMAX, limits.jerkmax)

        dynamics = joint.dynamics
        c.write_bool(
            prim, t.A_JOINT_HAS_DYNAMICS, True if dynamics is not None else None
        )
        if dynamics is not None:
            c.write_double(prim, t.A_JOINT_DYN_DAMPING, dynamics.damping)
            c.write_double(prim, t.A_JOINT_DYN_FRICTION, dynamics.friction)
            c.write_double(
                prim, t.A_JOINT_DYN_SPRING_REFERENCE, dynamics.spring_reference
            )
            c.write_double(
                prim, t.A_JOINT_DYN_SPRING_STIFFNESS, dynamics.spring_stiffness
            )
            c.write_double(prim, t.A_JOINT_DYN_ARMATURE, dynamics.armature)
            c.write_double(prim, t.A_JOINT_DYN_BACKLASH, dynamics.backlash)

        if joint.mimic is not None:
            # No core UsdPhysics equivalent. The sidecars are authoritative; a
            # NewtonMimicAPI is added on top where Isaac can simulate the coupling
            # (see _write_native_mimics).
            c.write_string(prim, t.A_JOINT_MIMIC_JOINT, joint.mimic.joint)
            c.write_double(prim, t.A_JOINT_MIMIC_MULTIPLIER, joint.mimic.multiplier)
            c.write_double(prim, t.A_JOINT_MIMIC_OFFSET, joint.mimic.offset)

    # -- actuators -----------------------------------------------------

    def _write_actuators(self) -> tuple[list[str], set[str]]:
        """Write the ``/Actuators`` records and their drives.

        Returns the prim order and the set of joint names that received a
        ``UsdPhysicsDriveAPI``, so passive spring-dampers know to stay away.
        """
        if not self.schema.actuators:
            return [], set()
        scope = self._scope(self.root_path, t.SCOPE_ACTUATORS)
        joints_by_name = {joint.name: joint for joint in self.schema.joints}
        driven: set[str] = set()
        order: list[str] = []

        for actuator in self.schema.actuators:
            path, _ = self._child_path(scope, actuator.name)
            prim = self._define(self.pxr.UsdGeom.Scope, path).GetPrim()
            order.append(self._relative(path, self.root_path))
            self._author_name_sidecar(prim, actuator.name)

            c.write_string(prim, t.A_ACTUATOR_JOINT, actuator.joint)
            c.write_token(prim, t.A_ACTUATOR_TYPE, actuator.type.value)
            c.write_double(prim, t.A_ACTUATOR_TORQUE_CONSTANT, actuator.torque_constant)
            c.write_double(prim, t.A_ACTUATOR_GEAR_RATIO, actuator.gear_ratio)
            c.write_double(prim, t.A_ACTUATOR_MAX_CURRENT, actuator.max_current)
            c.write_double(prim, t.A_ACTUATOR_MAX_TORQUE, actuator.max_torque)
            c.write_double(prim, t.A_ACTUATOR_MAX_VELOCITY, actuator.max_velocity)
            c.write_double(prim, t.A_ACTUATOR_KP, actuator.kp)
            c.write_double(prim, t.A_ACTUATOR_KI, actuator.ki)
            c.write_double(prim, t.A_ACTUATOR_KD, actuator.kd)
            c.write_float_pair(prim, t.A_ACTUATOR_CONTROL_RANGE, actuator.control_range)
            c.write_float_pair(prim, t.A_ACTUATOR_FORCE_RANGE, actuator.force_range)
            c.write_double(prim, t.A_ACTUATOR_RESISTANCE, actuator.resistance)
            c.write_double(prim, t.A_ACTUATOR_INDUCTANCE, actuator.inductance)
            c.write_double(prim, t.A_ACTUATOR_EFFICIENCY, actuator.efficiency)
            c.write_custom_data_json(prim, t.CD_EXTENSIONS, actuator.extensions)

            # Only the first actuator on a joint gets the drive: UsdPhysicsDriveAPI
            # is one instance per DOF, so a second one would overwrite the first.
            joint = joints_by_name.get(actuator.joint)
            if (
                joint is not None
                and actuator.joint not in driven
                and self._write_drive(joint, actuator)
            ):
                driven.add(actuator.joint)
        return order, driven

    def _write_passive_drives(self, driven: set[str]) -> None:
        """Express un-actuated joints with spring/damper dynamics as USD drives.

        A ``UsdPhysicsDriveAPI`` whose stiffness pulls towards a rest position and
        whose damping resists motion *is* a passive spring-damper in USD; there is
        no separate schema for one.
        """
        for joint in self.schema.joints:
            if joint.name in driven or joint.dynamics is None:
                continue
            self._write_passive_drive(joint)

    def _drive_target(self, joint: Joint) -> Any | None:
        instance = t.DRIVE_INSTANCE_FOR_JOINT_TYPE.get(joint.type)
        if instance is None:
            return None
        joint_path = self.joint_paths.get(joint.name)
        if joint_path is None:
            return None
        return self.pxr.UsdPhysics.DriveAPI.Apply(
            self.stage.GetPrimAtPath(joint_path), instance
        )

    def _write_drive(self, joint: Joint, actuator: Actuator) -> bool:
        """Author the joint's ``UsdPhysicsDriveAPI`` for its primary actuator.

        The gains follow the actuator's control mode, mirroring what the MJCF
        exporter emits and cyberwave-sim drives: a position servo is a
        stiffness/damping drive towards ``home_position``; a velocity servo is a
        pure damping drive (``kd`` is MJCF ``kv``) towards zero velocity; a motor
        is an effort-only drive with no gains, only ``maxForce``. Angular gains
        are converted to UsdPhysics' per-degree units.
        """
        drive = self._drive_target(joint)
        if drive is None:
            return False
        is_angular = joint.type in t.ANGULAR_JOINT_TYPES
        mode = sem.CONTROL_MODE_FOR_ACTUATOR.get(actuator.type)
        drive.CreateTypeAttr(
            t.DRIVE_TYPE_FOR_ACTUATOR.get(actuator.type, t.TOK_DRIVE_FORCE)
        )
        if mode == "position":
            if actuator.kp is not None:
                drive.CreateStiffnessAttr(
                    sem.angular_gain_to_usd(actuator.kp, is_angular)
                )
            if actuator.kd is not None:
                drive.CreateDampingAttr(sem.angular_gain_to_usd(actuator.kd, is_angular))
            if joint.home_position is not None:
                drive.CreateTargetPositionAttr(
                    _to_usd_angle(joint.home_position, is_angular)
                )
        elif mode == "velocity":
            if actuator.kd is not None:
                drive.CreateDampingAttr(sem.angular_gain_to_usd(actuator.kd, is_angular))
            drive.CreateTargetVelocityAttr(0.0)
        max_force = actuator.max_torque
        if max_force is None and actuator.force_range is not None:
            max_force = max(abs(actuator.force_range[0]), abs(actuator.force_range[1]))
        if max_force is not None:
            drive.CreateMaxForceAttr(float(max_force))
        prim = drive.GetPrim()
        c.write_custom_data_value(prim, t.CD_CONTROL_MODE, mode)
        c.write_custom_data_value(prim, t.CD_ACTUATOR_NAME, actuator.name)
        return True

    def _write_passive_drive(self, joint: Joint) -> None:
        dynamics = joint.dynamics
        if dynamics is None:
            return
        if dynamics.damping == 0.0 and dynamics.spring_stiffness == 0.0:
            return
        drive = self._drive_target(joint)
        if drive is None:
            return
        is_angular = joint.type in t.ANGULAR_JOINT_TYPES
        drive.CreateTypeAttr(t.TOK_DRIVE_FORCE)
        if dynamics.spring_stiffness != 0.0:
            drive.CreateStiffnessAttr(
                sem.angular_gain_to_usd(dynamics.spring_stiffness, is_angular)
            )
        if dynamics.damping != 0.0:
            drive.CreateDampingAttr(
                sem.angular_gain_to_usd(dynamics.damping, is_angular)
            )
        target = (
            joint.home_position
            if joint.home_position is not None
            else dynamics.spring_reference
        )
        drive.CreateTargetPositionAttr(_to_usd_angle(target, is_angular))

    # -- sensors -------------------------------------------------------

    def _write_sensor(
        self, parent_path: Any, sensor: Sensor, nested_under_link: bool
    ) -> Any:
        """Author a sensor as a typed prim plus its raw schema record.

        Cameras (rgb/depth/first-person) become ``UsdGeomCamera`` with the full
        pinhole model; every other sensor is an ``Xform`` mount frame. Either way
        the prim carries the portable ``cyberwave:sensorKind`` /
        ``cyberwave:sensorParams`` customData a runtime instantiates its own
        sensor from (no simulator-specific sensor schema is authored), and the
        verbatim ``cyberwave:sensor:*`` attributes that make the round trip exact.

        Camera frames need no conversion: ``CommonSchema`` camera poses use the
        MuJoCo convention (look along -Z, +Y up), which is also USD's.
        """
        kind = sem.sensor_kind(sensor.type)
        is_camera = kind in sem.CAMERA_KINDS
        path, _ = self._child_path(parent_path, sensor.name)
        schema_type = self.pxr.UsdGeom.Camera if is_camera else self.pxr.UsdGeom.Xform
        prim = self._define(schema_type, path).GetPrim()
        self.sensor_paths.setdefault(sensor.name, path)
        self._author_name_sidecar(prim, sensor.name)
        self._set_local_pose(prim, sensor.pose)
        c.write_custom_data_value(prim, t.CD_SENSOR_NAME, sensor.name)
        c.write_custom_data_value(prim, t.CD_SENSOR_KIND, kind)
        c.write_custom_data_value(
            prim, t.CD_SENSOR_PARAMS, sem.normalized_sensor_params(sensor, kind)
        )
        if is_camera:
            write_camera_intrinsics(
                self.pxr.UsdGeom.Camera(prim),
                sem.camera_intrinsics(sensor.parameters),
            )

        # `type` is free-form in the schema (IMU, camera, lidar, rangefinder, ...)
        # and is written verbatim: mapping it onto a USD vocabulary would need a
        # lossy invention per kind, and USD has no core sensor schema anyway.
        c.write_string(prim, t.A_SENSOR_TYPE, sensor.type)
        c.write_double(prim, t.A_SENSOR_UPDATE_RATE, sensor.update_rate)
        c.write_json_attr(prim, t.A_SENSOR_NOISE, sensor.noise)
        c.write_json_attr(prim, t.A_SENSOR_PARAMETERS, sensor.parameters or None)
        # parent_link is recovered from prim ancestry when the sensor sits under
        # its link; it only needs an attribute when it does not — a "world"-parented
        # sensor, or one whose parent_link does not exist as a link.
        if not nested_under_link:
            c.write_string(prim, t.A_SENSOR_PARENT_LINK, sensor.parent_link)
        c.write_custom_data_json(prim, t.CD_EXTENSIONS, sensor.extensions)
        return path

    # -- rotors --------------------------------------------------------

    def _write_rotors(self) -> None:
        """Author a frame per thrust-producing site motor under its link.

        ``Links/<link>/Rotors/<actuator>`` sits at the MJCF site's pose and
        carries ``cyberwave:rotor`` (thrust axis, gain, reaction torque ratio,
        control range). USD has no thruster schema, so a runtime applies the
        force itself; the ``Actuators`` record keeps the round trip exact.
        """
        sites = sem.sites_by_name(self.schema)
        for actuator in self.schema.actuators:
            rotor = sem.rotor_spec(actuator, sites)
            if rotor is None or rotor.link not in self.link_paths:
                continue
            scope = self._scope(self.link_paths[rotor.link], t.SCOPE_ROTORS)
            path, _ = self._child_path(scope, actuator.name)
            prim = self._define(self.pxr.UsdGeom.Xform, path).GetPrim()
            self.rotor_paths[actuator.name] = path
            self._set_transform(
                prim,
                f.pose_to_matrix(
                    Pose(position=rotor.position, orientation=rotor.orientation)
                ),
            )
            c.write_custom_data_value(prim, t.CD_ACTUATOR_NAME, actuator.name)
            c.write_custom_data_value(prim, t.CD_ROTOR, rotor.to_custom_data())

    # -- contacts ------------------------------------------------------

    def _write_contacts(self) -> list[str]:
        if not self.schema.contacts:
            return []
        scope = self._scope(self.root_path, t.SCOPE_CONTACTS)
        order: list[str] = []
        for contact in self.schema.contacts:
            path, _ = self._child_path(scope, contact.name)
            prim = self._define(self.pxr.UsdGeom.Scope, path).GetPrim()
            order.append(self._relative(path, self.root_path))
            self._author_name_sidecar(prim, contact.name)
            c.write_string(prim, t.A_CONTACT_LINK, contact.link)
            _write_contact_surface(prim, t.CONTACT_SURFACE_PREFIX, contact.surface)
            c.write_custom_data_json(prim, t.CD_EXTENSIONS, contact.extensions)
        return order

    # -- collision config ----------------------------------------------

    def _write_collision_config(self) -> None:
        config = self.schema.collision_config
        if config is None:
            return
        self._write_collision_groups(config)

        path = self.root_path.AppendChild(t.SCOPE_COLLISION_CONFIG)
        prim = self._define(self.pxr.UsdGeom.Scope, path).GetPrim()
        c.write_custom_data_json(prim, t.CD_EXTENSIONS, config.extensions)

        # Excludes are authoritative as customData: it preserves list order and
        # (body1, body2) orientation exactly, and it survives links that do not
        # exist as prims — which CommonSchema.validate() permits.
        if config.excludes:
            c.write_custom_data_json(
                prim,
                t.CD_EXCLUDES,
                [[exclude.body1, exclude.body2] for exclude in config.excludes],
            )
            self._apply_filtered_pairs(config)

        if config.pairs:
            pairs_scope = self._scope(path, t.SCOPE_PAIRS)
            for index, pair in enumerate(config.pairs):
                pair_path = pairs_scope.AppendChild(f"{t.PAIR_PREFIX}{index}")
                pair_prim = self._define(self.pxr.UsdGeom.Scope, pair_path).GetPrim()
                c.write_string(pair_prim, t.A_PAIR_GEOM1, pair.geom1)
                c.write_string(pair_prim, t.A_PAIR_GEOM2, pair.geom2)
                c.write_float_list(pair_prim, t.A_PAIR_FRICTION, pair.friction)
                c.write_float_list(pair_prim, t.A_PAIR_SOLREF, pair.solref)
                c.write_float_list(pair_prim, t.A_PAIR_SOLIMP, pair.solimp)
                c.write_int(pair_prim, t.A_PAIR_CONDIM, pair.condim)
                c.write_double(pair_prim, t.A_PAIR_MARGIN, pair.margin)
                c.write_double(pair_prim, t.A_PAIR_GAP, pair.gap)
                c.write_custom_data_json(pair_prim, t.CD_EXTENSIONS, pair.extensions)

    def _write_collision_groups(self, config: CollisionConfig) -> None:
        if not config.groups:
            return
        scope = self._scope(self.root_path, t.SCOPE_COLLISION_GROUPS)
        group_paths: dict[str, Any] = {}
        for name, group in config.groups.items():
            path, _ = self._child_path(scope, name)
            usd_group = self._define(self.pxr.UsdPhysics.CollisionGroup, path)
            prim = usd_group.GetPrim()
            group_paths[name] = path
            self._author_name_sidecar(prim, name)
            # MuJoCo bitmask semantics have no USD analogue, so they are the
            # authority; physics:filteredGroups below is derived from them.
            c.write_int(prim, t.A_GROUP_CONTYPE, group.contype)
            c.write_int(prim, t.A_GROUP_CONAFFINITY, group.conaffinity)
            # CollisionGroup has no extensions field of its own, so the USD-only
            # group properties ride along in CollisionConfig.extensions, keyed by
            # group name under "usd".
            c.write_usd_passthrough(
                prim, t.USD_PASSTHROUGH_GROUP, _group_extensions(config, name)
            )

            members = self._group_members.get(name, [])
            if members:
                usd_group.GetCollidersCollectionAPI().CreateIncludesRel().SetTargets(
                    members
                )

        # Derive pairwise filtering so PhysX actually honours the bitmasks:
        # two groups collide iff (A.contype & B.conaffinity) or vice versa.
        for name, group in config.groups.items():
            filtered = [
                group_paths[other_name]
                for other_name, other in config.groups.items()
                if not (
                    (group.contype & other.conaffinity)
                    or (other.contype & group.conaffinity)
                )
            ]
            if filtered:
                self.pxr.UsdPhysics.CollisionGroup(
                    self.stage.GetPrimAtPath(group_paths[name])
                ).CreateFilteredGroupsRel().SetTargets(filtered)

    def _apply_filtered_pairs(self, config: CollisionConfig) -> None:
        """Mirror the excludes onto ``UsdPhysicsFilteredPairsAPI`` for consumers."""
        by_body: dict[str, list[Any]] = {}
        for exclude in config.excludes:
            first = self.link_paths.get(exclude.body1)
            second = self.link_paths.get(exclude.body2)
            if first is None or second is None:
                continue
            by_body.setdefault(str(first), []).append(second)
        for body_path, targets in by_body.items():
            prim = self.stage.GetPrimAtPath(self.pxr.Sdf.Path(body_path))
            api = self.pxr.UsdPhysics.FilteredPairsAPI.Apply(prim)
            api.CreateFilteredPairsRel().SetTargets(targets)

    # -- scene / physics -----------------------------------------------

    def _write_render_scope(self, scene: Scene | None) -> None:
        if scene is None:
            return
        path = self.root_path.AppendChild(t.SCOPE_RENDER)
        prim = self._define(self.pxr.UsdGeom.Scope, path).GetPrim()
        # UsdRenderSettings has no notion of a grid or an origin gizmo, and
        # half-mapping only `shadows` would buy nothing, so Scene is sidecar-only.
        c.write_float_list(prim, t.A_SCENE_AMBIENT, scene.ambient)
        c.write_float_list(prim, t.A_SCENE_BACKGROUND, scene.background)
        c.write_bool(prim, t.A_SCENE_SHADOWS, scene.shadows)
        c.write_bool(prim, t.A_SCENE_GRID, scene.grid)
        c.write_bool(prim, t.A_SCENE_ORIGIN_VISUAL, scene.origin_visual)
        c.write_custom_data_json(prim, t.CD_EXTENSIONS, scene.extensions)

    def _physics_scene_path(self) -> Any:
        """Return a free stage path for the ``UsdPhysicsScene``.

        Normally the canonical ``/PhysicsScene`` beside the robot prim. The only
        prim that can occupy it is the robot root itself (everything else the
        exporter writes lives under the root), and a robot whose name sanitizes to
        ``PhysicsScene`` must not have its articulation root silently *redefined* as
        a physics scene — that both corrupts the root's prim type and merges
        world-level physics into the referenceable subtree, which is exactly what
        keeping the scene outside the root is meant to avoid.
        """
        pxr = self.pxr
        path = pxr.Sdf.Path(t.PHYSICS_SCENE_PATH)
        suffix = 2
        while self.stage.GetPrimAtPath(path):
            path = pxr.Sdf.Path(f"{t.PHYSICS_SCENE_PATH}_{suffix}")
            suffix += 1
        return path

    def _write_physics_scene(self, physics: Physics | None) -> None:
        if physics is None:
            return
        pxr = self.pxr
        usd_scene = self._define(pxr.UsdPhysics.Scene, self._physics_scene_path())
        prim = usd_scene.GetPrim()

        gravity = physics.gravity
        magnitude = float(
            np.linalg.norm([gravity.x, gravity.y, gravity.z])
        )
        if magnitude > 0.0:
            usd_scene.CreateGravityDirectionAttr(
                pxr.Gf.Vec3f(
                    gravity.x / magnitude, gravity.y / magnitude, gravity.z / magnitude
                )
            )
            usd_scene.CreateGravityMagnitudeAttr(magnitude)
        # The sidecar is not redundant: it is double precision and it separates a
        # genuine zero-g schema from USD's "(0,0,0) means use the up axis" and
        # "-inf means earth gravity" sentinels.
        c.write_vector3(prim, t.A_PHYSICS_GRAVITY, gravity)
        c.write_double(prim, t.A_PHYSICS_TIMESTEP, physics.timestep)

        solver = physics.solver
        c.write_string(prim, t.A_PHYSICS_SOLVER_TYPE, solver.type)
        c.write_int(prim, t.A_PHYSICS_SOLVER_ITERATIONS, solver.iterations)
        c.write_double(prim, t.A_PHYSICS_SOLVER_SOR, solver.sor)
        c.write_double(prim, t.A_PHYSICS_SOLVER_MIN_STEP, solver.min_step_size)
        c.write_double(prim, t.A_PHYSICS_SOLVER_MAX_STEP, solver.max_step_size)
        c.write_double(prim, t.A_PHYSICS_SOLVER_TOLERANCE, solver.tolerance)
        c.write_custom_data_json(prim, t.CD_SOLVER_EXTENSIONS, solver.extensions)

        _write_contact_surface(prim, t.PHYSICS_CONTACT_PREFIX, physics.contact)
        c.write_custom_data_json(prim, t.CD_EXTENSIONS, physics.extensions)
        self._write_physx(
            prim, px.SCENE_APIS, px.physx_extension(physics.extensions), "physics"
        )


# --------------------------------------------------------------------------
# Module-level helpers
# --------------------------------------------------------------------------


def _group_extensions(config: CollisionConfig, group_name: str) -> dict[str, Any]:
    """USD-only properties for one collision group, if any were preserved.

    ``CollisionGroup`` has no ``extensions`` field, so they live at
    ``collision_config.extensions["usd"]["groups"][<name>]``.
    """
    stored = config.extensions.get(t.USD_EXTENSION_KEY)
    if not isinstance(stored, dict):
        return {}
    groups = stored.get("groups")
    if not isinstance(groups, dict):
        return {}
    per_group = groups.get(group_name)
    if not isinstance(per_group, dict):
        return {}
    return {t.USD_EXTENSION_KEY: per_group}


def _pruned(order: dict[str, list[str]]) -> dict[str, list[str]]:
    """Drop empty entries from a ``cyberwave:order`` dict.

    A missing key and an empty list mean the same thing to the parser (fall back
    to traversal order), so omitting them keeps the ``.usda`` free of
    ``"contacts":[]`` noise on the majority of models.
    """
    return {key: value for key, value in order.items() if value}


def write_camera_intrinsics(camera: Any, intrinsics: sem.CameraIntrinsics) -> None:
    """Author a pinhole model on a ``UsdGeomCamera`` (see ``CameraIntrinsics``)."""
    pxr = c.require_pxr()
    camera.CreateProjectionAttr(pxr.UsdGeom.Tokens.perspective)
    camera.CreateFocalLengthAttr(intrinsics.usd_focal_length)
    camera.CreateHorizontalApertureAttr(intrinsics.usd_horizontal_aperture)
    camera.CreateVerticalApertureAttr(intrinsics.usd_vertical_aperture)
    if intrinsics.usd_horizontal_aperture_offset != 0.0:
        camera.CreateHorizontalApertureOffsetAttr(
            intrinsics.usd_horizontal_aperture_offset
        )
    if intrinsics.usd_vertical_aperture_offset != 0.0:
        camera.CreateVerticalApertureOffsetAttr(
            intrinsics.usd_vertical_aperture_offset
        )
    camera.CreateClippingRangeAttr(
        pxr.Gf.Vec2f(intrinsics.near_clip, intrinsics.far_clip)
    )


def _to_usd_angle(value: float, is_angular: bool) -> float:
    """Convert a schema coordinate to USD units (degrees for angular DOFs)."""
    return float(np.degrees(value)) if is_angular else float(value)


def _write_contact_surface(prim: Any, prefix: str, surface: ContactSurface) -> None:
    """Author a :class:`ContactSurface` under ``prefix``.

    One writer for both ``Contact.surface`` and ``Physics.contact`` so the two can
    never drift apart. All eight fields are sidecars: UsdPhysicsMaterialAPI covers
    only friction and restitution, in float32.
    """
    c.write_double(prim, f"{prefix}MuStatic", surface.mu_static)
    c.write_double(prim, f"{prefix}MuDynamic", surface.mu_dynamic)
    c.write_double(prim, f"{prefix}Restitution", surface.restitution)
    c.write_double(prim, f"{prefix}Stiffness", surface.stiffness)
    c.write_double(prim, f"{prefix}Damping", surface.damping)
    c.write_json_attr(prim, f"{prefix}SlipCompliance", surface.slip_compliance)
    c.write_double(prim, f"{prefix}SoftCfm", surface.soft_cfm)
    c.write_double(prim, f"{prefix}SoftErp", surface.soft_erp)


def _metadata_to_dict(metadata: Any) -> dict[str, Any]:
    """Serialize :class:`Metadata` for the root prim's customData."""
    return {
        "name": metadata.name,
        "version": metadata.version,
        "author": metadata.author,
        "description": metadata.description,
        "source_format": metadata.source_format,
        "creation_date": metadata.creation_date,
        "units": metadata.units,
        "extensions": metadata.extensions,
    }


__all__ = ["USDExporter"]
