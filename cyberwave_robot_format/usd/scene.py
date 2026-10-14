# Copyright [2025] Cyberwave
# Licensed under the Apache License, Version 2.0

"""Multi-robot scene export: a deterministic USD scene bundle.

:func:`export_scene_usd` places one or more robot instances (each a
``CommonSchema`` plus a spawn pose) in one scene and writes it as a directory of
USD layers that any UsdPhysics runtime can load, together with the identity map
that ties every prim back to its instance, link, joint and sensor. It is the USD
counterpart of :meth:`CommonSchema.merge_in` followed by the MJCF scene export:
an instance id plays the role of ``merge_in``'s ``instance_name``, and nothing
here knows what the caller's instances stand for. Callers that need their own
identity on the stage pass it as opaque ``custom_data``.

Layout of the bundle::

    scene.usda                     root layer (defaultPrim = /World)
    assets/<asset>_<sha12>.usda    one per distinct robot asset, shared by every
                                   instance of it

and of the composed stage::

    /World                                    units/axis metadata, schema version
    /World/PhysicsScene                       UsdPhysics.Scene (gravity)
    /World/<scene>                            ``scene_custom_data``
    /World/<scene>/<instance>                 reference -> asset; ``custom_data``;
                                              PhysicsArticulationRootAPI (from asset)
    /World/<scene>/<instance>/Links/<link>    rigid bodies (flat, UsdPhysics rule)
    /World/<scene>/<instance>/Joints/<joint>  joints + drives (+ generated world anchor)
    /World/<scene>/<instance>/Links/<link>/Sensors/<sensor>   Camera or Xform
    /World/<scene>/<instance>/Links/<link>/Rotors/<actuator>  thrust frames
    /World/<scene>/Lights/{key_light,ambient_light}           generated defaults
    /World/<scene>/Cameras/sim_viewport       free viewport camera
    /World/<scene>/GroundPlane                generated unless an instance has a plane
    /World/<scene>/GroundVisual/{tile,pit}_<n>  its visual, when scenery is buried

``<scene>`` is :attr:`SceneExportOptions.scene_prim_name` and ``<instance>`` is
:attr:`SceneInstance.namespace` (``instance_<hex>`` by default).

The asset layers are produced by :class:`~cyberwave_robot_format.usd.USDExporter`
from a copy of each schema with its world joint removed: placement belongs to
the *instance* (spawn pose), not the asset, which is what lets two instances of a
robot share one layer. A fixed base is re-attached with a generated
``PhysicsFixedJoint`` to the world; a floating base simply has no world joint.

Nothing here imports a simulator. The output uses ``UsdGeom``, ``UsdPhysics``,
``UsdShade`` and ``UsdLux``, ``cyberwave:*`` customData, and the PhysX and Newton
applied API schemas of :mod:`.physx` (``PhysxSceneAPI``, ``PhysxArticulationAPI``,
``PhysxRigidBodyAPI``, ``PhysxJointAPI``, the collision APIs, ``NewtonMimicAPI``),
authored by name without the NVIDIA SDK. The module itself imports fine without
``usd-core`` (only calling the export needs it).
"""

from __future__ import annotations

import copy
import hashlib
import math
import re
import uuid as uuid_module
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from cyberwave_robot_format.schema import (
    ActuatorType,
    CommonSchema,
    GeometryType,
    Joint,
    JointType,
    Physics,
    Pose,
    Quaternion,
    Vector3,
)
from cyberwave_robot_format.usd import common as c
from cyberwave_robot_format.usd import frames as f
from cyberwave_robot_format.usd import ground
from cyberwave_robot_format.usd import physx as px
from cyberwave_robot_format.usd import semantics as sem
from cyberwave_robot_format.usd import tokens as t

ROOT_LAYER_NAME = "scene.usda"
ASSETS_DIR = "assets"
WORLD_PRIM = "World"
PHYSICS_SCENE_NAME = "PhysicsScene"
DEFAULT_SCENE_PRIM = "Scene"
INSTANCE_PRIM_PREFIX = "instance_"
SIM_VIEWPORT_ID = "sim_viewport"
WORLD_ANCHOR_NAME = "world_anchor"
EXPORTER_NAME = "cyberwave-robot-format"

_FIDELITY_BY_KIND = {
    "rgb": "rendered",
    "depth": "rendered",
    "first_person_camera": "rendered",
    "sim_viewport": "rendered",
    "lidar": "geometric_ray",
    "point_cloud": "geometric_ray",
    "range": "geometric_ray",
    "imu": "kinematic",
}
"""Declared fidelity per kind; a runtime that implements less must downgrade it."""

_ASSET_FILE_CHARS = re.compile(r"[^A-Za-z0-9_]")


# --------------------------------------------------------------------------
# Public data types
# --------------------------------------------------------------------------


@dataclass
class SceneInstance:
    """One robot instance to place in the scene.

    The USD counterpart of the arguments of :meth:`CommonSchema.merge_in`: one
    schema, where it spawns, and whether its base is welded.

    Attributes:
        instance_id: Stable, unique identity of the instance in the scene; every
            result entry that belongs to it carries it as ``instance_id``. A UUID
            string becomes its 32-digit hex in prim and default sensor names.
        schema: The robot schema (un-namespaced, as stored on the asset).
        spawn_pose: World pose of the root link (relative to ``attach_to_link``
            when docked), the ``spawn_pose`` of :meth:`CommonSchema.merge_in`.
        fixed_base: ``True`` welds the root link to the world, ``False`` leaves
            it floating. ``None`` infers it from the schema: floating only when
            the schema's own world joint is ``FLOATING``.
        sensor_ids: Schema sensor name -> stable sensor id. Unlisted sensors get
            ``<instance hex>__<sensor name>``, the name ``merge_in`` gives them.
        first_person_camera: Name of the camera sensor that is the instance's
            first-person view, if any.
        namespace: Prim name of the instance; defaults to ``instance_<hex>``.
        attach_to_instance / attach_to_link: Dock this instance onto a link of
            another instance (forces a fixed base).
        custom_data: Extra ``customData`` for the instance prim (key -> plain
            value, written as given), for a caller's own identity keys.
        self_collisions: Whether the instance's links collide with each other.
            ``None`` keeps what the asset authors (PhysX enables articulation
            self-collision by default); ``True``/``False`` overrides it in the
            scene layer with ``physxArticulation:enabledSelfCollisions`` on the
            instance prim, the one carrying ``PhysicsArticulationRootAPI``. An
            asset that sets the attribute itself (``extensions["physx"]``) keeps
            its value. The USD counterpart of the MJCF scene's shared "robot"
            collision group.
    """

    instance_id: str
    schema: CommonSchema
    spawn_pose: Pose = field(default_factory=Pose)
    fixed_base: bool | None = None
    sensor_ids: Mapping[str, str] = field(default_factory=dict)
    first_person_camera: str | None = None
    namespace: str | None = None
    attach_to_instance: str | None = None
    attach_to_link: str | None = None
    custom_data: Mapping[str, Any] = field(default_factory=dict)
    self_collisions: bool | None = None


@dataclass
class ViewportOptions:
    """The free ``sim_viewport`` camera; defaults match cyberwave-sim's MuJoCo view."""

    enabled: bool = True
    lookat: tuple[float, float, float] = (0.0, 0.0, 0.0)
    distance: float = 6.0
    azimuth_deg: float = 90.0
    elevation_deg: float = -20.0
    fovy_deg: float = 45.0
    width: int = 960
    height: int = 540
    fps: float = 15.0


@dataclass
class StaticBoxFixture:
    """A static box (visual + collider) placed directly in the world.

    Procedural scene fixtures (rails, walls, stands) as the MuJoCo scene
    composition welds them to the world: no rigid body, so PhysX treats
    the collider as static. ``position`` / ``orientation`` are the box center's
    world pose; ``size`` the full extents in metres.
    """

    name: str
    position: Vector3
    size: Vector3
    orientation: Quaternion = field(default_factory=Quaternion)
    color: tuple[float, float, float] | None = None
    collision: bool = True
    mu_static: float | None = None
    mu_dynamic: float | None = None
    restitution: float | None = None
    source_id: str | None = None


@dataclass
class SceneExportOptions:
    """Scene-level export settings.

    Attributes:
        scene_prim_name: Name of the prim under ``/World`` that holds the
            instances, lights, cameras, ground and fixtures.
        scene_name: Display name, recorded on that prim as ``cyberwave:name``.
        scene_custom_data: Extra ``customData`` for that prim, as for
            :attr:`SceneInstance.custom_data`.
        physics: World physics; defaults to the first instance's (by id) schema
            physics, else :class:`Physics` defaults.
        lights: Author a generated key (distant) + ambient (dome) light. Cyberwave
            Format has no light source data; MuJoCo uses its headlight.
        key_light_intensity / ambient_light_intensity: Their ``inputs:intensity``.
        ground_plane: ``None`` (default) adds one unless an instance already has a
            plane geometry, mirroring the MuJoCo scene composition.
        viewport: The ``sim_viewport`` camera.
        static_fixtures: Static boxes written under ``Fixtures`` (procedural
            primitives the MuJoCo scene composition welds to the world).
        bake_meshes / mesh_search_paths: Passed to :class:`USDExporter`. Without
            baking, mesh prims carry only their source reference and the bundle
            is reported as incomplete (``mesh_not_embedded``).
    """

    scene_prim_name: str = DEFAULT_SCENE_PRIM
    scene_name: str | None = None
    scene_custom_data: Mapping[str, Any] = field(default_factory=dict)
    physics: Physics | None = None
    lights: bool = True
    key_light_intensity: float = 1000.0
    ambient_light_intensity: float = 500.0
    ground_plane: bool | None = None
    viewport: ViewportOptions = field(default_factory=ViewportOptions)
    static_fixtures: list[StaticBoxFixture] = field(default_factory=list)
    bake_meshes: bool = False
    mesh_search_paths: list[str] = field(default_factory=list)


@dataclass
class UsdExportResult:
    """What :func:`export_scene_usd` wrote, as JSON-ready dictionaries.

    Entries that belong to an instance carry its ``instance_id`` (``None`` for
    scene-level entries such as the viewport camera), and so do findings in
    ``compatibility``; ``identity_map["instances"]`` lists the instance prims.
    Paths in ``files`` and ``root_layer`` are relative to the output directory;
    every ``usd_path`` is an absolute prim path in the composed stage.
    Drive gains in ``articulations`` are SI per radian (the schema's units), not
    the per-degree values UsdPhysics stores for angular drives.
    """

    root_layer: str
    files: list[dict[str, Any]]
    identity_map: dict[str, list[dict[str, Any]]]
    sensors: list[dict[str, Any]]
    articulations: list[dict[str, Any]]
    compatibility: dict[str, list[dict[str, Any]]]
    units: dict[str, Any]
    exporter: dict[str, Any]
    physics_scene: dict[str, Any]

    @property
    def ok(self) -> bool:
        """``True`` when there are no compatibility errors."""
        return not self.compatibility["errors"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_layer": self.root_layer,
            "files": self.files,
            "identity_map": self.identity_map,
            "sensors": self.sensors,
            "articulations": self.articulations,
            "compatibility": self.compatibility,
            "units": self.units,
            "exporter": self.exporter,
            "physics_scene": self.physics_scene,
        }


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def export_scene_usd(
    scene_or_instances: CommonSchema | SceneInstance | Iterable[SceneInstance],
    out_dir: str | Path,
    *,
    options: SceneExportOptions | None = None,
) -> UsdExportResult:
    """Write a USD scene bundle for ``scene_or_instances`` into ``out_dir``.

    Accepts a sequence of :class:`SceneInstance`, a single instance, or a bare
    ``CommonSchema`` (one instance whose id is derived from its name). Output is
    byte-for-byte deterministic for the same input and package version: instances
    are ordered by id, every float is authored from the same arithmetic, and
    no timestamps are written. Existing files in ``out_dir`` other than the ones
    listed in the result are left alone.

    Raises:
        ImportError: ``usd-core`` is not installed.
        ValueError: Two instances share an id or a prim namespace.
    """
    pxr = c.require_pxr()
    options = options or SceneExportOptions()
    specs = _coerce_specs(scene_or_instances)
    out = Path(out_dir)
    (out / ASSETS_DIR).mkdir(parents=True, exist_ok=True)
    return _SceneBuilder(pxr, specs, out, options).build()


# --------------------------------------------------------------------------
# Implementation
# --------------------------------------------------------------------------


class _Findings:
    def __init__(self) -> None:
        self.errors: list[dict[str, Any]] = []
        self.warnings: list[dict[str, Any]] = []
        self.rejected: list[dict[str, Any]] = []

    @staticmethod
    def _finding(
        code: str,
        message: str,
        *,
        instance_id: str | None = None,
        subject: str | None = None,
    ) -> dict[str, Any]:
        finding: dict[str, Any] = {"code": code, "message": message}
        if instance_id is not None:
            finding["instance_id"] = instance_id
        if subject is not None:
            finding["subject"] = subject
        return finding

    def error(self, code: str, message: str, **context: Any) -> None:
        self.errors.append(self._finding(code, message, **context))

    def warning(self, code: str, message: str, **context: Any) -> None:
        self.warnings.append(self._finding(code, message, **context))

    def reject(self, code: str, message: str, **context: Any) -> None:
        self.rejected.append(self._finding(code, message, **context))

    def extend(self, other: _Findings, **context: Any) -> None:
        """Copy in every finding of *other*, each with *context* added."""
        for mine, theirs in (
            (self.errors, other.errors),
            (self.warnings, other.warnings),
            (self.rejected, other.rejected),
        ):
            mine.extend(self._finding(f["code"], f["message"], subject=f.get("subject"), **context) for f in theirs)

    def as_dict(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "errors": self.errors,
            "warnings": self.warnings,
            "rejected_features": self.rejected,
        }


@dataclass
class _Asset:
    """One written asset layer, shared by every instance with identical content."""

    relpath: str
    paths: Any  # exporter.AssetPaths
    schema: CommonSchema
    # Found once per layer, reported once per instance that references it.
    findings: _Findings = field(default_factory=_Findings)


@dataclass
class _Instance:
    spec: SceneInstance
    hex: str
    namespace: str
    asset_schema: CommonSchema
    base: str
    world_joint: Joint | None
    kept_world_joints: list[Joint]
    root_link: str | None
    path: Any
    asset: _Asset | None = None
    world: np.ndarray = field(default_factory=lambda: np.eye(4))
    anchor_path: Any = None


class _SceneBuilder:
    def __init__(
        self,
        pxr: Any,
        specs: list[SceneInstance],
        out: Path,
        options: SceneExportOptions,
    ) -> None:
        self.pxr = pxr
        self.out = out
        self.options = options
        self.findings = _Findings()
        self.assets_by_digest: dict[str, _Asset] = {}
        self.files: dict[str, bytes] = {}
        self.scene_path = pxr.Sdf.Path(f"/{WORLD_PRIM}").AppendChild(c.sanitize_prim_name(options.scene_prim_name))
        self.instances = self._prepare_instances(specs)
        self.identity: dict[str, list[dict[str, Any]]] = {
            "instances": [],
            "links": [],
            "joints": [],
            "sensors": [],
            "cameras": [],
            "materials": [],
        }
        self.sensors: list[dict[str, Any]] = []
        self.articulations: list[dict[str, Any]] = []

    # -- instances -------------------------------------------------------

    def _prepare_instances(self, specs: list[SceneInstance]) -> list[_Instance]:
        instances: list[_Instance] = []
        seen_ids: set[str] = set()
        seen_namespaces: set[str] = set()
        for spec in sorted(specs, key=lambda s: str(s.instance_id)):
            instance_id = str(spec.instance_id)
            if instance_id in seen_ids:
                raise ValueError(f"Duplicate instance_id {instance_id!r}")
            seen_ids.add(instance_id)
            hex_id = _uuid_hex(instance_id)
            namespace = c.sanitize_prim_name(spec.namespace or f"{INSTANCE_PRIM_PREFIX}{hex_id}")
            if namespace in seen_namespaces:
                raise ValueError(f"Duplicate instance prim namespace {namespace!r}")
            seen_namespaces.add(namespace)
            instances.append(self._prepare_instance(spec, hex_id, namespace))
        return instances

    def _prepare_instance(self, spec: SceneInstance, hex_id: str, namespace: str) -> _Instance:
        """Split placement from the asset: strip the world joint, decide the base."""
        schema = copy.deepcopy(spec.schema)
        context = {"instance_id": str(spec.instance_id)}
        world_joints = [j for j in schema.joints if j.parent_link == t.WORLD_LINK]
        stripped: Joint | None = None
        kept: list[Joint] = []
        for joint in world_joints:
            if stripped is None and joint.type in (JointType.FIXED, JointType.FLOATING):
                stripped = joint
            else:
                kept.append(joint)
        if stripped is not None:
            schema.joints = [j for j in schema.joints if j is not stripped]
        if len([j for j in kept if j.type in (JointType.FIXED, JointType.FLOATING)]) > 0:
            self.findings.warning(
                "multiple_world_joints",
                "More than one fixed/floating world joint; only the first becomes "
                "the instance placement, the others are kept as world-attached joints.",
                **context,
            )

        child_links = {j.child_link for j in schema.joints}
        roots = [link.name for link in schema.links if link.name not in child_links and link.name != t.WORLD_LINK]
        if not roots:
            self.findings.error(
                "no_root_link",
                "The schema has no root link (empty model or kinematic loop).",
                **context,
            )
        elif len(roots) > 1:
            self.findings.warning(
                "multiple_root_links",
                f"Schema has {len(roots)} root links; '{roots[0]}' is the "
                "articulation base and the others are free bodies.",
                subject=",".join(roots),
                **context,
            )

        docked = spec.attach_to_instance is not None
        if kept:
            base = "fixed"
        elif docked:
            base = "fixed"
            if spec.fixed_base is False:
                self.findings.warning(
                    "docked_twin_forced_fixed",
                    "A docked instance is always welded to its parent link.",
                    **context,
                )
        elif spec.fixed_base is not None:
            base = "fixed" if spec.fixed_base else "floating"
        else:
            floating = stripped is not None and stripped.type == JointType.FLOATING
            base = "floating" if floating else "fixed"

        return _Instance(
            spec=spec,
            hex=hex_id,
            namespace=namespace,
            asset_schema=schema,
            base=base,
            world_joint=stripped,
            kept_world_joints=kept,
            root_link=roots[0] if roots else None,
            path=self.scene_path.AppendChild(namespace),
        )

    # -- build -----------------------------------------------------------

    def build(self) -> UsdExportResult:
        for instance in self.instances:
            instance.asset = self._write_asset(instance)
        self._resolve_world_transforms()

        pxr = self.pxr
        root_file = self.out / ROOT_LAYER_NAME
        layer = self._new_root_layer(root_file)
        stage = pxr.Usd.Stage.Open(layer)
        self.stage = stage
        pxr.UsdGeom.SetStageUpAxis(stage, pxr.UsdGeom.Tokens.z)
        pxr.UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        pxr.UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
        from cyberwave_robot_format import __version__

        # Wording kept from before the instance API so bundle bytes (and digests) stay put.
        layer.comment = f"Generated by {EXPORTER_NAME} {__version__} (environment scene bundle)"

        world = pxr.UsdGeom.Xform.Define(stage, pxr.Sdf.Path(f"/{WORLD_PRIM}")).GetPrim()
        stage.SetDefaultPrim(world)
        c.write_custom_data_value(world, t.CD_USD_SCHEMA_VERSION, t.USD_SCHEMA_VERSION)
        physics_scene = self._write_physics_scene()

        scene_path = self.scene_path
        scene_prim = pxr.UsdGeom.Xform.Define(stage, scene_path).GetPrim()
        for key, value in sorted(self.options.scene_custom_data.items()):
            c.write_custom_data_value(scene_prim, key, value)
        if self.options.scene_name:
            c.write_string(scene_prim, t.A_NAME, self.options.scene_name)

        for instance in self.instances:
            self._write_instance(instance)
        if self.options.lights:
            self._write_lights(scene_path)
        self._write_viewport(scene_path)
        self._write_static_fixtures(scene_path)
        if self._wants_ground_plane():
            self._write_ground_plane(scene_path)
        self._verify_identity_paths()

        self.files[ROOT_LAYER_NAME] = layer.ExportToString().encode("utf-8")
        files = self._write_files()
        del stage, self.stage

        return UsdExportResult(
            root_layer=ROOT_LAYER_NAME,
            files=files,
            identity_map={k: v for k, v in self.identity.items() if v or k != "materials"},
            sensors=self.sensors,
            articulations=self.articulations,
            compatibility=self.findings.as_dict(),
            units={"meters_per_unit": 1.0, "kilograms_per_unit": 1.0, "up_axis": "Z"},
            exporter={
                "name": EXPORTER_NAME,
                "version": __version__,
                "usd_schema_version": t.USD_SCHEMA_VERSION,
            },
            physics_scene=physics_scene,
        )

    def _new_root_layer(self, root_file: Path) -> Any:
        """A layer anchored at its real path, so relative asset references resolve.

        ``Sdf.Layer.New`` rather than ``CreateNew``: nothing touches the disk until
        :meth:`_write_files` writes the exported bytes, which keeps writing in one
        place and makes the output independent of USD's save path.
        """
        pxr = self.pxr
        identifier = str(root_file.resolve())
        existing = pxr.Sdf.Layer.Find(identifier)
        if existing:
            existing.Clear()
            return existing
        return pxr.Sdf.Layer.New(pxr.Sdf.FileFormat.FindById("usda"), identifier)

    # -- assets ----------------------------------------------------------

    def _write_asset(self, instance: _Instance) -> _Asset:
        from cyberwave_robot_format.usd.exporter import USDExporter

        exporter = USDExporter(
            bake_meshes=self.options.bake_meshes,
            mesh_search_paths=list(self.options.mesh_search_paths),
        )
        stage, paths = exporter.build_stage(instance.asset_schema)
        data = stage.GetRootLayer().ExportToString().encode("utf-8")
        digest = hashlib.sha256(data).hexdigest()
        existing = self.assets_by_digest.get(digest)
        if existing is not None:
            self.findings.extend(existing.findings, instance_id=str(instance.spec.instance_id))
            return existing

        stem = _ASSET_FILE_CHARS.sub("_", instance.asset_schema.metadata.name).strip("_")
        relpath = f"{ASSETS_DIR}/{(stem or 'asset').lower()}_{digest[:12]}.usda"
        self.files[relpath] = data
        # Composition reads the layer from disk; write it now (and drop any stale
        # copy USD still caches from an earlier export into the same directory).
        target = self.out / relpath
        target.write_bytes(data)
        cached = self.pxr.Sdf.Layer.Find(str(target.resolve()))
        if cached:
            cached.Reload(force=True)

        asset = _Asset(relpath=relpath, paths=paths, schema=instance.asset_schema)
        self.assets_by_digest[digest] = asset
        self._check_asset(stage, asset.findings, instance, paths)
        self.findings.extend(asset.findings, instance_id=str(instance.spec.instance_id))
        return asset

    def _check_asset(self, stage: Any, findings: _Findings, instance: _Instance, paths: Any) -> None:
        """Asset-level findings into *findings*, checked once per distinct asset layer.

        They carry no ``instance_id``: :meth:`_write_asset` reports them under every
        instance sharing the layer, since callers group findings per instance.
        """
        pxr = self.pxr
        root = stage.GetDefaultPrim()
        for prim in pxr.Usd.PrimRange(root):
            if not prim.IsA(pxr.UsdGeom.Mesh):
                continue
            if pxr.UsdGeom.Mesh(prim).GetPointsAttr().HasAuthoredValue():
                continue
            is_collision = prim.HasAPI(pxr.UsdPhysics.CollisionAPI)
            subject = str(prim.GetPath().MakeRelativePath(root.GetPath()))
            message = (
                "Mesh geometry is only a reference to its source file; the bundle "
                "has no points for it. Export with bake_meshes=True."
            )
            if is_collision:
                findings.error("mesh_not_embedded", message, subject=subject)
            else:
                findings.warning("mesh_not_embedded", message, subject=subject)

        schema = instance.asset_schema
        textures = sorted(
            {
                visual.material.texture
                for link in schema.links
                for visual in link.visuals
                if visual.material is not None and visual.material.texture
            }
        )
        for texture in textures:
            findings.warning(
                "texture_not_bundled",
                "Texture is referenced by path and not copied into the bundle.",
                subject=texture,
            )

        primary = sem.primary_actuators(schema)
        secondary = sorted({a.joint for a in schema.actuators if a.joint and primary.get(a.joint) is not a})
        if secondary:
            findings.warning(
                "secondary_actuator_not_driven",
                f"{len(secondary)} joint(s) have more than one actuator; the USD drive "
                "follows the first, the others exist only as records.",
                subject=",".join(secondary),
            )
        joints_by_name = {joint.name: joint for joint in schema.joints}
        for joint in schema.joints:
            if joint.mimic is None:
                continue
            problem = px.native_mimic_problem(joint, joints_by_name.get(joint.mimic.joint))
            if problem is not None:
                findings.warning(
                    "mimic_joint_not_native",
                    f"No NewtonMimicAPI could be authored ({problem}); the coupling "
                    "is only in cyberwave:joint:mimic* and the joint is passive in Isaac.",
                    subject=joint.name,
                )
        for subject, problem in paths.physx_problems:
            findings.warning(
                "physx_attribute_skipped",
                f"extensions.physx entry not authored: {problem}.",
                subject=subject,
            )
        for joint in schema.joints:
            if joint.type == JointType.FLOATING:
                # The placement world joint was already stripped; this one is
                # inside the tree, which an articulation cannot represent.
                findings.warning(
                    "joint_type_not_native",
                    "An internal floating joint has no UsdPhysics equivalent; it is "
                    "authored as an unconstrained PhysicsJoint outside the joint vector.",
                    subject=joint.name,
                )
            elif joint.type in (JointType.PLANAR, JointType.SPHERICAL, JointType.UNIVERSAL):
                findings.warning(
                    "multi_dof_joint_not_indexed",
                    f"{joint.type.value} joints are simulated (D6 locks / spherical "
                    "joint) but have no dof_index, drive or reset entry.",
                    subject=joint.name,
                )

        sites = sem.sites_by_name(schema)
        for actuator in schema.actuators:
            if actuator.type == ActuatorType.ADHESION:
                findings.reject(
                    "adhesion_actuator_unsupported",
                    "Adhesion actuators have no USD representation.",
                    subject=actuator.name,
                )
                continue
            try:
                transmission = actuator.get_mjcf_site_transmission()
            except ValueError:
                transmission = None
            if transmission is None:
                continue
            rotor = sem.rotor_spec(actuator, sites)
            if rotor is None:
                findings.reject(
                    "site_motor_not_rotor",
                    "Site motor without a thrust component (e.g. a reaction " "wheel) has no USD representation.",
                    subject=actuator.name,
                )
                continue
            for note in rotor.notes:
                findings.warning(note, f"Rotor {actuator.name!r}: {note}.", subject=actuator.name)

    # -- placement -------------------------------------------------------

    def _resolve_world_transforms(self) -> None:
        by_id = {str(instance.spec.instance_id): instance for instance in self.instances}
        resolved: dict[str, np.ndarray] = {}

        def resolve(instance: _Instance, chain: tuple[str, ...]) -> np.ndarray:
            key = str(instance.spec.instance_id)
            if key in resolved:
                return resolved[key]
            local = f.pose_to_matrix(instance.spec.spawn_pose)
            if instance.world_joint is not None:
                local = local @ f.pose_to_matrix(instance.world_joint.pose)
            parent_id = instance.spec.attach_to_instance
            parent = by_id.get(str(parent_id)) if parent_id is not None else None
            if parent_id is not None and (parent is None or key in chain):
                self.findings.error(
                    "dock_parent_missing" if parent is None else "dock_cycle",
                    f"Cannot dock onto instance {parent_id!r}; placing at the spawn " "pose in the world frame instead.",
                    instance_id=key,
                )
                instance.spec = copy.copy(instance.spec)
                instance.spec.attach_to_instance = None
                parent = None
            if parent is not None:
                link_world = f.accumulated_world_transforms(parent.asset_schema)
                link_name = instance.spec.attach_to_link or parent.root_link or ""
                if link_name not in link_world:
                    self.findings.error(
                        "dock_link_missing",
                        f"Parent instance has no link {link_name!r}; docking to its root.",
                        instance_id=key,
                        subject=link_name,
                    )
                    link_name = parent.root_link or ""
                    instance.spec = copy.copy(instance.spec)
                    instance.spec.attach_to_link = link_name
                self.findings.warning(
                    "twin_docked",
                    "Docked instance is placed at the parent's zero configuration and " "held by a generated fixed joint.",
                    instance_id=key,
                    subject=link_name,
                )
                parent_world = resolve(parent, chain + (key,))
                if key in resolved:
                    # The recursion came back to this instance: it is in a dock
                    # cycle and the inner frame already placed it at its spawn
                    # pose. Composing onto the parent now would undo that.
                    return resolved[key]
                world = parent_world @ link_world.get(link_name, np.eye(4)) @ local
            else:
                world = local
            resolved[key] = world
            return world

        for instance in self.instances:
            instance.world = resolve(instance, ())

    # -- root layer authoring --------------------------------------------

    def _write_physics_scene(self) -> dict[str, Any]:
        pxr = self.pxr
        physics = self.options.physics
        if physics is None:
            physics = (
                next(
                    (inst.spec.schema.physics for inst in self.instances if inst.spec.schema.physics),
                    None,
                )
                or Physics()
            )
        path = pxr.Sdf.Path(f"/{WORLD_PRIM}/{PHYSICS_SCENE_NAME}")
        scene = pxr.UsdPhysics.Scene.Define(self.stage, path)
        gravity = physics.gravity
        magnitude = float(np.linalg.norm([gravity.x, gravity.y, gravity.z]))
        if magnitude > 0.0:
            scene.CreateGravityDirectionAttr(
                pxr.Gf.Vec3f(gravity.x / magnitude, gravity.y / magnitude, gravity.z / magnitude)
            )
        scene.CreateGravityMagnitudeAttr(magnitude)
        prim = scene.GetPrim()
        c.write_vector3(prim, t.A_PHYSICS_GRAVITY, gravity)
        c.write_double(prim, t.A_PHYSICS_TIMESTEP, physics.timestep)
        for problem in px.write_physx(prim, px.SCENE_APIS, px.physx_extension(physics.extensions)):
            self.findings.warning(
                "physx_attribute_skipped",
                f"extensions.physx entry not authored: {problem}.",
                subject="physics",
            )
        return {
            "usd_path": str(path),
            "gravity": [sem.rounded(gravity.x), sem.rounded(gravity.y), sem.rounded(gravity.z)],
            "timestep": sem.rounded(physics.timestep),
        }

    def _write_instance(self, instance: _Instance) -> None:
        pxr = self.pxr
        assert instance.asset is not None
        spec = instance.spec
        instance_id = str(spec.instance_id)
        context = {"instance_id": instance_id}
        path = instance.path
        prim = pxr.UsdGeom.Xform.Define(self.stage, path).GetPrim()
        prim.GetReferences().AddReference(f"./{instance.asset.relpath}")
        _set_transform(prim, instance.world)
        for key, value in sorted(spec.custom_data.items()):
            c.write_custom_data_value(prim, key, value)
        c.write_custom_data_value(prim, t.CD_BASE, instance.base)
        self.identity["instances"].append({"instance_id": instance_id, "usd_path": str(path)})

        paths = instance.asset.paths
        schema = instance.asset_schema

        def composed(relative: str) -> Any:
            return path.AppendPath(relative)

        for link in schema.links:
            if link.name in paths.links:
                self.identity["links"].append(
                    {
                        "instance_id": instance_id,
                        "link": link.name,
                        "usd_path": str(composed(paths.links[link.name])),
                    }
                )
        dof = sem.dof_indices(schema)
        for joint in schema.joints:
            if joint.name in dof and joint.name in paths.joints:
                self.identity["joints"].append(
                    {
                        "instance_id": instance_id,
                        "joint": joint.name,
                        "usd_path": str(composed(paths.joints[joint.name])),
                        "dof_index": dof[joint.name],
                    }
                )
        for material_name, relative in sorted(paths.materials.items()):
            self.identity["materials"].append(
                {
                    "material_id": f"{instance.hex}__{material_name}",
                    "usd_path": str(composed(relative)),
                }
            )

        self._write_world_attachment(instance, composed)
        self._write_instance_sensors(instance, composed, context)
        self._write_articulation(instance, composed)
        self._disable_inert_colliders(instance)
        self._write_self_collisions(instance)

    def _write_self_collisions(self, instance: _Instance) -> None:
        """Author the instance's ``self_collisions`` choice on its articulation root.

        PhysX collides every pair of links of an articulation except the ones a
        joint connects, so auto-fitted link colliders that overlap across a joint
        chain (a quadruped's lower leg hull and its body) fight each other with
        large contact forces from the first step: the robot crumples to the floor.
        """
        wanted = instance.spec.self_collisions
        if wanted is None:
            return
        name = "physxArticulation:enabledSelfCollisions"
        if name in px.physx_extension(instance.asset_schema.extensions):
            return
        prim = self.stage.GetPrimAtPath(instance.path)
        if not prim.HasAPI(self.pxr.UsdPhysics.ArticulationRootAPI):
            return
        px.write_physx(prim, px.ARTICULATION_APIS, {name: bool(wanted)})

    def _disable_inert_colliders(self, instance: _Instance) -> None:
        """Turn off colliders whose group collides with nothing (contype = conaffinity = 0).

        The asset layer filters such a group only against its *own* groups, and every
        asset authors its own ``CollisionGroups`` scope, so PhysX would still collide
        it with every other instance — e.g. a track exported with
        ``physics_override.disable_collision`` pinning the bogie that rides on it. An
        override in the scene layer keeps the asset layer round-trippable.
        """
        inert = _inert_collision_groups(instance.asset_schema)
        if not inert:
            return
        pxr = self.pxr
        for prim in pxr.Usd.PrimRange(self.stage.GetPrimAtPath(instance.path)):
            if not prim.HasAPI(pxr.UsdPhysics.CollisionAPI):
                continue
            group = prim.GetAttribute(t.A_COLLISION_GROUP).Get() or "default"
            if group in inert:
                pxr.UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)

    def _write_world_attachment(self, instance: _Instance, composed: Any) -> None:
        """Weld a fixed base to the world (or its dock link); re-anchor kept joints."""
        pxr = self.pxr
        assert instance.asset is not None
        paths = instance.asset.paths
        # World-attached joints kept in the asset carry localPos0/localRot0 in the
        # asset frame, but UsdPhysics reads an empty body0 frame as *world*, so
        # they need the instance placement folded in.
        for joint in instance.kept_world_joints:
            relative = paths.joints.get(joint.name)
            if relative is None:
                continue
            usd_joint = pxr.UsdPhysics.Joint(self.stage.GetPrimAtPath(composed(relative)))
            pos = usd_joint.GetLocalPos0Attr().Get() or pxr.Gf.Vec3f(0.0)
            rot = usd_joint.GetLocalRot0Attr().Get() or pxr.Gf.Quatf(1.0)
            local = f.pose_to_matrix(
                Pose(
                    position=Vector3(*[float(v) for v in pos]),
                    orientation=Quaternion(
                        x=rot.GetImaginary()[0],
                        y=rot.GetImaginary()[1],
                        z=rot.GetImaginary()[2],
                        w=rot.GetReal(),
                    ),
                )
            )
            _set_joint_frame0(pxr, usd_joint, instance.world @ local)
            instance.anchor_path = composed(relative)

        if instance.base != "fixed" or instance.kept_world_joints or instance.root_link is None:
            return
        root_relative = paths.links.get(instance.root_link)
        if root_relative is None:
            return

        name = instance.world_joint.name if instance.world_joint is not None else WORLD_ANCHOR_NAME
        joints_scope = instance.path.AppendChild(t.SCOPE_JOINTS)
        if not self.stage.GetPrimAtPath(joints_scope):
            pxr.UsdGeom.Scope.Define(self.stage, joints_scope)
        prim_name = c.sanitize_prim_name(name)
        candidate = joints_scope.AppendChild(prim_name)
        suffix = 2
        while self.stage.GetPrimAtPath(candidate):
            candidate = joints_scope.AppendChild(f"{prim_name}_{suffix}")
            suffix += 1

        usd_joint = pxr.UsdPhysics.FixedJoint.Define(self.stage, candidate)
        prim = usd_joint.GetPrim()
        spec = instance.spec
        if spec.attach_to_instance is not None:
            parent = next(inst for inst in self.instances if str(inst.spec.instance_id) == str(spec.attach_to_instance))
            assert parent.asset is not None
            parent_link = spec.attach_to_link or parent.root_link or ""
            parent_relative = parent.asset.paths.links.get(parent_link)
            body0 = [parent.path.AppendPath(parent_relative)] if parent_relative else []
            frame0 = f.pose_to_matrix(spec.spawn_pose)
            if instance.world_joint is not None:
                frame0 = frame0 @ f.pose_to_matrix(instance.world_joint.pose)
            c.write_string(prim, t.A_JOINT_PARENT_LINK, parent_link)
        else:
            body0 = []
            frame0 = instance.world
        usd_joint.CreateBody0Rel().SetTargets(body0)
        usd_joint.CreateBody1Rel().SetTargets([composed(root_relative)])
        _set_joint_frame0(pxr, usd_joint, frame0)
        usd_joint.CreateLocalPos1Attr(pxr.Gf.Vec3f(0.0, 0.0, 0.0))
        usd_joint.CreateLocalRot1Attr(pxr.Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        c.write_token(prim, t.A_JOINT_TYPE, JointType.FIXED.value)
        c.write_custom_data_value(prim, t.CD_JOINT_NAME, name)
        c.write_custom_data_value(prim, t.CD_GENERATED, True)
        instance.anchor_path = candidate

    def _sensor_id(self, instance: _Instance, sensor_name: str) -> str:
        return str(instance.spec.sensor_ids.get(sensor_name) or f"{instance.hex}__{sensor_name}")

    def _write_instance_sensors(self, instance: _Instance, composed: Any, context: dict[str, Any]) -> None:
        assert instance.asset is not None
        paths = instance.asset.paths
        instance_id = str(instance.spec.instance_id)
        first_person = instance.spec.first_person_camera
        found_first_person = False

        for sensor in instance.asset_schema.sensors:
            relative = paths.sensors.get(sensor.name)
            kind = sem.sensor_kind(sensor.type)
            if kind is None:
                self.findings.reject(
                    "sensor_kind_unsupported",
                    f"Sensor type {sensor.type!r} has no v1 sensor kind; it is kept "
                    "as a parametric Xform but no runtime sensor is declared.",
                    subject=sensor.name,
                    **context,
                )
                continue
            if relative is None:
                continue
            usd_path = composed(relative)
            prim = self.stage.GetPrimAtPath(usd_path)
            sensor_id = self._sensor_id(instance, sensor.name)
            if sensor.name == first_person:
                if kind in sem.CAMERA_KINDS:
                    kind = "first_person_camera"
                    found_first_person = True
                else:
                    self.findings.warning(
                        "first_person_camera_not_camera",
                        "The first-person camera names a non-camera sensor.",
                        subject=sensor.name,
                        **context,
                    )
            if sensor.type.strip().lower() == "rgbd":
                self.findings.warning(
                    "rgbd_streamed_as_rgb",
                    "RGB-D sensors are declared as RGB; add a separate depth sensor " "for depth output.",
                    subject=sensor.name,
                    **context,
                )

            params = sem.normalized_sensor_params(sensor, kind)
            fidelity = _FIDELITY_BY_KIND[kind]
            if kind in ("rgb", "first_person_camera") and sem.is_thermal_sensor(sensor):
                # Thermal is a display type on an RGB stream (cyberwave-sim): the
                # kind stays rgb, the marker tells the runtime to render a heat
                # image (thermal_width/height, min/max_temperature_c pass through).
                params["display_palette"] = sem.THERMAL_DISPLAY_PALETTE
                params = dict(sorted(params.items()))
                fidelity = sem.THERMAL_FIDELITY
            if sensor.noise:
                noise = _plain_numbers(sensor.noise)
                if noise:
                    params["noise"] = noise
            c.write_custom_data_value(prim, t.CD_SENSOR_ID, sensor_id)
            c.write_custom_data_value(prim, t.CD_SENSOR_KIND, kind)

            self.identity["sensors"].append(
                {
                    "instance_id": instance_id,
                    "sensor_id": sensor_id,
                    "kind": kind,
                    "usd_path": str(usd_path),
                }
            )
            if kind in sem.CAMERA_KINDS:
                role = "first_person" if kind == "first_person_camera" else "sensor"
                c.write_custom_data_value(prim, t.CD_CAMERA_ROLE, role)
                self.identity["cameras"].append(
                    {
                        "instance_id": instance_id,
                        "camera_id": sensor_id,
                        "sensor_id": sensor_id,
                        "role": role,
                        "usd_path": str(usd_path),
                    }
                )
                for note in sem.camera_intrinsics(sensor.parameters).notes:
                    self.findings.warning(
                        note,
                        f"Camera {sensor.name!r}: {note.replace('_', ' ')}.",
                        subject=sensor.name,
                        **context,
                    )
            self.sensors.append(
                {
                    "sensor_id": sensor_id,
                    "instance_id": instance_id,
                    "kind": kind,
                    "fidelity": fidelity,
                    "params": params,
                    "name": sensor.name,
                    "parent_link": sensor.parent_link,
                    "usd_path": str(usd_path),
                }
            )

        if first_person and not found_first_person:
            self.findings.warning(
                "first_person_camera_missing",
                f"First-person camera {first_person!r} is not a camera sensor of this instance.",
                subject=first_person,
                **context,
            )

    def _write_articulation(self, instance: _Instance, composed: Any) -> None:
        assert instance.asset is not None
        schema = instance.asset_schema
        paths = instance.asset.paths
        dof = sem.dof_indices(schema)
        primary = sem.primary_actuators(schema)
        drives: list[dict[str, Any]] = []
        joint_positions: dict[str, float] = {}
        for joint in schema.joints:
            if joint.name not in dof:
                continue
            drives.append(
                _drive_entry(
                    joint,
                    primary.get(joint.name),
                    dof[joint.name],
                    str(composed(paths.joints[joint.name])) if joint.name in paths.joints else None,
                )
            )
            joint_positions[joint.name] = sem.rounded(joint.home_position or 0.0)

        pose = f.matrix_to_pose(instance.world)
        sites = sem.sites_by_name(schema)
        rotors = []
        for actuator in schema.actuators:
            rotor = sem.rotor_spec(actuator, sites) if not actuator.joint else None
            if rotor is None or actuator.name not in paths.rotors:
                continue
            rotors.append(
                {
                    **_snake_rotor(rotor.to_custom_data()),
                    "usd_path": str(composed(paths.rotors[actuator.name])),
                }
            )

        entry: dict[str, Any] = {
            "instance_id": str(instance.spec.instance_id),
            "root_usd_path": str(instance.path),
            "base": instance.base,
            "root_link": instance.root_link,
            "root_link_usd_path": (
                str(composed(paths.links[instance.root_link])) if instance.root_link in paths.links else None
            ),
            "world_anchor_usd_path": str(instance.anchor_path) if instance.anchor_path else None,
            "drives": drives,
            "reset": {
                "base_pose": {
                    "position": [
                        sem.rounded(pose.position.x),
                        sem.rounded(pose.position.y),
                        sem.rounded(pose.position.z),
                    ],
                    "orientation_wxyz": [
                        sem.rounded(pose.orientation.w),
                        sem.rounded(pose.orientation.x),
                        sem.rounded(pose.orientation.y),
                        sem.rounded(pose.orientation.z),
                    ],
                },
                "joint_positions": joint_positions,
            },
        }
        if instance.world_joint is not None:
            entry["source_world_joint"] = {
                "name": instance.world_joint.name,
                "type": instance.world_joint.type.value,
            }
        if rotors:
            entry["rotors"] = rotors
        if instance.spec.attach_to_instance is not None:
            entry["attached_to"] = {
                "instance_id": str(instance.spec.attach_to_instance),
                "link": instance.spec.attach_to_link,
            }
        self.articulations.append(entry)

    def _write_lights(self, scene_path: Any) -> None:
        pxr = self.pxr
        scope = scene_path.AppendChild("Lights")
        pxr.UsdGeom.Scope.Define(self.stage, scope)

        key = pxr.UsdLux.DistantLight.Define(self.stage, scope.AppendChild("key_light"))
        key.CreateIntensityAttr(float(self.options.key_light_intensity))
        key.CreateAngleAttr(0.53)
        # A distant light shines along its -Z; tilt it 45 degrees off vertical so
        # shapes read in the viewport, like MuJoCo's headlight-plus-ambient look.
        tilt = f.pose_to_matrix(
            Pose(orientation=_axis_angle_quat((0.0, 1.0, 0.0), math.radians(-35.0)))
        ) @ f.pose_to_matrix(Pose(orientation=_axis_angle_quat((1.0, 0.0, 0.0), math.radians(25.0))))
        _set_transform(key.GetPrim(), tilt)
        c.write_custom_data_value(key.GetPrim(), t.CD_LIGHT_ROLE, "key")
        c.write_custom_data_value(key.GetPrim(), t.CD_GENERATED, True)

        dome = pxr.UsdLux.DomeLight.Define(self.stage, scope.AppendChild("ambient_light"))
        dome.CreateIntensityAttr(float(self.options.ambient_light_intensity))
        c.write_custom_data_value(dome.GetPrim(), t.CD_LIGHT_ROLE, "ambient")
        c.write_custom_data_value(dome.GetPrim(), t.CD_GENERATED, True)

    def _write_viewport(self, scene_path: Any) -> None:
        from cyberwave_robot_format.usd.exporter import write_camera_intrinsics

        viewport = self.options.viewport
        if not viewport.enabled:
            return
        pxr = self.pxr
        scope = scene_path.AppendChild("Cameras")
        pxr.UsdGeom.Scope.Define(self.stage, scope)
        path = scope.AppendChild(SIM_VIEWPORT_ID)
        camera = pxr.UsdGeom.Camera.Define(self.stage, path)
        prim = camera.GetPrim()
        _set_transform(prim, viewport_matrix(viewport))
        raw = {
            "width": viewport.width,
            "height": viewport.height,
            "vfov_rad": math.radians(viewport.fovy_deg),
        }
        intrinsics = sem.camera_intrinsics(raw)
        write_camera_intrinsics(camera, intrinsics)
        params: dict[str, Any] = {
            "width": intrinsics.width,
            "height": intrinsics.height,
            "fx": intrinsics.fx,
            "fy": intrinsics.fy,
            "cx": intrinsics.cx,
            "cy": intrinsics.cy,
            "hfov_rad": intrinsics.hfov_rad,
            "vfov_rad": intrinsics.vfov_rad,
            "near_clip": intrinsics.near_clip,
            "far_clip": intrinsics.far_clip,
            "update_rate_hz": sem.rounded(viewport.fps),
            "default_view": {
                "lookat": [sem.rounded(v) for v in viewport.lookat],
                "distance": sem.rounded(viewport.distance),
                "azimuth_deg": sem.rounded(viewport.azimuth_deg),
                "elevation_deg": sem.rounded(viewport.elevation_deg),
            },
        }
        c.write_custom_data_value(prim, t.CD_SENSOR_ID, SIM_VIEWPORT_ID)
        c.write_custom_data_value(prim, t.CD_SENSOR_KIND, "sim_viewport")
        c.write_custom_data_value(prim, t.CD_CAMERA_ROLE, "sim_viewport")
        c.write_custom_data_value(prim, t.CD_SENSOR_PARAMS, params)
        c.write_custom_data_value(prim, t.CD_GENERATED, True)
        self.identity["sensors"].append(
            {
                "instance_id": None,
                "sensor_id": SIM_VIEWPORT_ID,
                "kind": "sim_viewport",
                "usd_path": str(path),
            }
        )
        self.identity["cameras"].append(
            {
                "instance_id": None,
                "camera_id": SIM_VIEWPORT_ID,
                "sensor_id": SIM_VIEWPORT_ID,
                "role": "sim_viewport",
                "usd_path": str(path),
            }
        )
        self.sensors.append(
            {
                "sensor_id": SIM_VIEWPORT_ID,
                "instance_id": None,
                "kind": "sim_viewport",
                "fidelity": _FIDELITY_BY_KIND["sim_viewport"],
                "params": params,
                "usd_path": str(path),
            }
        )

    def _wants_ground_plane(self) -> bool:
        if self.options.ground_plane is not None:
            return self.options.ground_plane
        for instance in self.instances:
            for link in instance.asset_schema.links:
                geometries = [g.geometry for g in link.collisions] + [g.geometry for g in link.visuals]
                if any(g is not None and g.type == GeometryType.PLANE for g in geometries):
                    return False
        return True

    def _write_ground_plane(self, scene_path: Any) -> None:
        pxr = self.pxr
        plane = pxr.UsdGeom.Plane.Define(self.stage, scene_path.AppendChild("GroundPlane"))
        plane.CreateAxisAttr(pxr.UsdGeom.Tokens.z)
        plane.CreateWidthAttr(100.0)
        plane.CreateLengthAttr(100.0)
        plane.CreateDisplayColorAttr(pxr.Vt.Vec3fArray([pxr.Gf.Vec3f(0.3, 0.3, 0.3)]))
        prim = plane.GetPrim()
        pxr.UsdPhysics.CollisionAPI.Apply(prim)
        c.write_custom_data_value(prim, t.CD_GENERATED, True)
        pits = self._buried_scenery_pits()
        if pits:
            # The collider stays exactly as it is; only what is drawn changes. A
            # guide prim is not rendered, as with every asset-layer collider.
            pxr.UsdGeom.Imageable(prim).CreatePurposeAttr(pxr.UsdGeom.Tokens.guide)
            self._write_ground_visual(scene_path, pits)

    def _buried_scenery_pits(self) -> list[ground.GroundRect]:
        """Pits under static, visual-only instances that sink below the ground (see :mod:`.ground`).

        The footprint is the instance's world-aligned visual bound: collider prims are
        ``guide`` purpose, so the bound covers exactly what is rendered.
        """
        pxr = self.pxr
        cache = None
        pits: list[ground.GroundRect] = []
        for instance in self.instances:
            if not _is_static_visual_only(instance):
                continue
            if cache is None:
                cache = pxr.UsdGeom.BBoxCache(
                    pxr.Usd.TimeCode.Default(),
                    [pxr.UsdGeom.Tokens.default_, pxr.UsdGeom.Tokens.render],
                )
            bound = cache.ComputeWorldBound(self.stage.GetPrimAtPath(instance.path)).ComputeAlignedRange()
            if bound.IsEmpty():
                continue
            pit = ground.buried_pit(bound.GetMin(), bound.GetMax())
            if pit is not None:
                pits.append(pit)
        return pits

    def _write_ground_visual(self, scene_path: Any, pits: list[ground.GroundRect]) -> None:
        """Visual-only ground tiles around ``pits`` plus a pit floor under each."""
        pxr = self.pxr
        scope = scene_path.AppendChild("GroundVisual")
        pxr.UsdGeom.Scope.Define(self.stage, scope)
        c.write_custom_data_value(self.stage.GetPrimAtPath(scope), t.CD_GENERATED, True)
        tiles, floors = ground.ground_visual_layout(pits)
        for prefix, rects in (("tile", tiles), ("pit", floors)):
            for index, rect in enumerate(rects):
                plane = pxr.UsdGeom.Plane.Define(self.stage, scope.AppendChild(f"{prefix}_{index}"))
                plane.CreateAxisAttr(pxr.UsdGeom.Tokens.z)
                width, length = rect.size
                plane.CreateWidthAttr(width)
                plane.CreateLengthAttr(length)
                plane.CreateDisplayColorAttr(pxr.Vt.Vec3fArray([pxr.Gf.Vec3f(0.3, 0.3, 0.3)]))
                matrix = np.eye(4)
                matrix[:3, 3] = rect.center
                _set_transform(plane.GetPrim(), matrix)

    def _write_static_fixtures(self, scene_path: Any) -> None:
        fixtures = self.options.static_fixtures
        if not fixtures:
            return
        pxr = self.pxr
        scope = scene_path.AppendChild("Fixtures")
        pxr.UsdGeom.Scope.Define(self.stage, scope)
        materials: dict[tuple[Any, Any, Any], Any] = {}
        used: set[str] = set()
        for index, fixture in enumerate(fixtures):
            name = c.sanitize_prim_name(fixture.name) or f"fixture_{index}"
            while name in used:
                name = f"{name}_{index}"
            used.add(name)
            xform = pxr.UsdGeom.Xform.Define(self.stage, scope.AppendChild(name)).GetPrim()
            _set_transform(
                xform,
                f.pose_to_matrix(Pose(position=fixture.position, orientation=fixture.orientation)),
            )
            c.write_custom_data_value(xform, t.CD_GENERATED, True)
            if fixture.source_id:
                c.write_string(xform, t.A_NAME, fixture.source_id)
            cube = pxr.UsdGeom.Cube.Define(self.stage, xform.GetPath().AppendChild("box"))
            cube.CreateSizeAttr(1.0)
            size = fixture.size
            cube.AddScaleOp().Set(pxr.Gf.Vec3f(float(size.x), float(size.y), float(size.z)))
            if fixture.color is not None:
                cube.CreateDisplayColorAttr(pxr.Vt.Vec3fArray([pxr.Gf.Vec3f(*fixture.color)]))
            if not fixture.collision:
                continue
            prim = cube.GetPrim()
            pxr.UsdPhysics.CollisionAPI.Apply(prim)
            key = (fixture.mu_static, fixture.mu_dynamic, fixture.restitution)
            if key == (None, None, None):
                continue
            if key not in materials:
                path = scope.AppendChild(f"{t.PHYSICS_MATERIAL_PREFIX}{len(materials)}")
                material = pxr.UsdShade.Material.Define(self.stage, path)
                api = pxr.UsdPhysics.MaterialAPI.Apply(material.GetPrim())
                if fixture.mu_static is not None:
                    api.CreateStaticFrictionAttr(float(fixture.mu_static))
                if fixture.mu_dynamic is not None:
                    api.CreateDynamicFrictionAttr(float(fixture.mu_dynamic))
                if fixture.restitution is not None:
                    api.CreateRestitutionAttr(float(fixture.restitution))
                materials[key] = material
            pxr.UsdShade.MaterialBindingAPI.Apply(prim)
            pxr.UsdShade.MaterialBindingAPI(prim).Bind(
                materials[key],
                pxr.UsdShade.Tokens.weakerThanDescendants,
                t.TOK_MATERIAL_PURPOSE_PHYSICS,
            )

    # -- output ----------------------------------------------------------

    def _verify_identity_paths(self) -> None:
        for section, entries in self.identity.items():
            for entry in entries:
                path = entry.get("usd_path")
                if path is None or not self.stage.GetPrimAtPath(path):
                    self.findings.error(
                        "identity_path_unresolved",
                        f"identity_map.{section} entry does not resolve in the " "composed stage.",
                        subject=str(path),
                        instance_id=entry.get("instance_id"),
                    )

    def _write_files(self) -> list[dict[str, Any]]:
        listing: list[dict[str, Any]] = []
        for relpath in sorted(self.files):
            data = self.files[relpath]
            target = self.out / relpath
            if not target.exists() or target.read_bytes() != data:
                target.write_bytes(data)
            listing.append(
                {
                    "path": relpath,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "size": len(data),
                }
            )
        return listing


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _coerce_specs(
    scene_or_instances: CommonSchema | SceneInstance | Iterable[SceneInstance],
) -> list[SceneInstance]:
    if isinstance(scene_or_instances, CommonSchema):
        name = scene_or_instances.metadata.name
        derived = uuid_module.uuid5(uuid_module.NAMESPACE_URL, f"cyberwave-robot-format:{name}")
        return [SceneInstance(instance_id=str(derived), schema=scene_or_instances)]
    if isinstance(scene_or_instances, SceneInstance):
        return [scene_or_instances]
    specs = list(scene_or_instances)
    for spec in specs:
        if not isinstance(spec, SceneInstance):
            raise TypeError(f"Expected SceneInstance, got {type(spec).__name__}")
    return specs


def _uuid_hex(value: str) -> str:
    try:
        return uuid_module.UUID(str(value)).hex
    except ValueError:
        return c.sanitize_prim_name(str(value)).lstrip("_") or "unnamed"


def _inert_collision_groups(schema: CommonSchema) -> set[str]:
    """Collision groups that collide with nothing (contype = conaffinity = 0)."""
    config = schema.collision_config
    if config is None or not config.groups:
        return set()
    return {name for name, group in config.groups.items() if group.contype == 0 and group.conaffinity == 0}


def _is_static_visual_only(instance: _Instance) -> bool:
    """A world-welded instance with no moving joint whose every collider is inert.

    ``physics_override.disable_collision`` exports exactly this: the instance is
    scenery that is seen but never touched.
    """
    schema = instance.asset_schema
    if instance.base != "fixed" or instance.spec.attach_to_instance is not None:
        return False
    if any(joint.type != JointType.FIXED for joint in schema.joints):
        return False
    collisions = [collision for link in schema.links for collision in link.collisions]
    inert = _inert_collision_groups(schema)
    return bool(collisions) and all((collision.group or "default") in inert for collision in collisions)


def _set_transform(prim: Any, matrix: np.ndarray) -> None:
    if np.allclose(matrix, np.eye(4), rtol=0.0, atol=0.0):
        return
    pxr = c.require_pxr()
    pxr.UsdGeom.Xformable(prim).AddTransformOp().Set(c.np_to_gf_matrix(matrix))


def _set_joint_frame0(pxr: Any, usd_joint: Any, matrix: np.ndarray) -> None:
    pose = f.matrix_to_pose(matrix)
    q = pose.orientation
    usd_joint.CreateLocalPos0Attr(pxr.Gf.Vec3f(pose.position.x, pose.position.y, pose.position.z))
    usd_joint.CreateLocalRot0Attr(pxr.Gf.Quatf(q.w, q.x, q.y, q.z))


def _axis_angle_quat(axis: tuple[float, float, float], angle: float) -> Quaternion:
    s = math.sin(angle / 2.0)
    return Quaternion(x=axis[0] * s, y=axis[1] * s, z=axis[2] * s, w=math.cos(angle / 2.0))


def viewport_matrix(viewport: ViewportOptions) -> np.ndarray:
    """Camera-to-world transform of MuJoCo's free camera for these settings.

    MuJoCo places the free camera at ``lookat - distance * forward`` with
    ``forward = (cos el cos az, cos el sin az, sin el)``; a USD camera looks
    down its -Z with +Y up, so the camera axes are ``(right, up, -forward)``.
    """
    az = math.radians(viewport.azimuth_deg)
    el = math.radians(viewport.elevation_deg)
    forward = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    eye = np.asarray(viewport.lookat, dtype=float) - viewport.distance * forward
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(right) < 1e-9:  # looking straight up/down
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    matrix = np.eye(4)
    matrix[:3, 0] = right
    matrix[:3, 1] = up
    matrix[:3, 2] = -forward
    matrix[:3, 3] = eye
    return np.vectorize(sem.rounded)(matrix)


def _plain_numbers(values: Mapping[str, Any]) -> dict[str, Any]:
    plain: dict[str, Any] = {}
    for key, value in sorted(values.items()):
        if isinstance(value, bool):
            continue
        if isinstance(value, int | float) and math.isfinite(float(value)):
            plain[str(key)] = sem.rounded(value)
        elif isinstance(value, list | tuple) and all(
            isinstance(v, int | float) and not isinstance(v, bool) for v in value
        ):
            plain[str(key)] = [sem.rounded(v) for v in value]
    return plain


def _snake_rotor(data: dict[str, Any]) -> dict[str, Any]:
    names = {
        "thrustAxis": "thrust_axis",
        "thrustGain": "thrust_gain",
        "torqueToThrust": "torque_to_thrust",
        "controlRange": "control_range",
        "maxThrust": "max_thrust",
    }
    return {names.get(key, key): value for key, value in data.items()}


def _drive_entry(joint: Joint, actuator: Any, dof_index: int, usd_path: str | None) -> dict[str, Any]:
    """The articulation's per-joint drive declaration, SI per radian/metre."""

    def num(value: float | None) -> float | None:
        if value is None or not math.isfinite(float(value)):
            return None
        return sem.rounded(value)

    limits = joint.limits
    dynamics = joint.dynamics
    entry: dict[str, Any] = {
        "joint": joint.name,
        "usd_path": usd_path,
        "dof_index": dof_index,
        "joint_type": joint.type.value,
    }
    stiffness = damping = max_force = target = None
    if actuator is not None:
        mode = sem.CONTROL_MODE_FOR_ACTUATOR.get(actuator.type, "none")
        entry["actuator"] = actuator.name
        if mode == "position":
            stiffness, damping, target = actuator.kp, actuator.kd, joint.home_position
        elif mode == "velocity":
            damping = actuator.kd
        max_force = actuator.max_torque
        if max_force is None and actuator.force_range is not None:
            max_force = max(abs(actuator.force_range[0]), abs(actuator.force_range[1]))
        max_velocity = actuator.max_velocity
    elif dynamics is not None and (dynamics.damping or dynamics.spring_stiffness):
        mode = "passive"
        stiffness = dynamics.spring_stiffness or None
        damping = dynamics.damping or None
        target = joint.home_position if joint.home_position is not None else dynamics.spring_reference
        max_velocity = None
    else:
        mode = "none"
        max_velocity = None
    if max_force is None and limits is not None:
        max_force = limits.effort
    if max_velocity is None and limits is not None:
        max_velocity = limits.velocity

    entry.update(
        mode=mode,
        stiffness=num(stiffness),
        damping=num(damping),
        max_force=num(max_force),
        max_velocity=num(max_velocity),
        target_position=num(target),
    )
    if limits is not None and joint.type != JointType.CONTINUOUS:
        entry["lower"] = num(limits.lower)
        entry["upper"] = num(limits.upper)
    if dynamics is not None:
        entry["joint_damping"] = num(dynamics.damping)
        entry["joint_friction"] = num(dynamics.friction)
        entry["armature"] = num(dynamics.armature)
    if joint.mimic is not None:
        entry["mimic"] = {
            "joint": joint.mimic.joint,
            "multiplier": num(joint.mimic.multiplier),
            "offset": num(joint.mimic.offset),
        }
    return entry
