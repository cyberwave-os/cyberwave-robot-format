# Copyright [2025] Cyberwave
# Licensed under the Apache License, Version 2.0

"""Parser from OpenUSD (``.usd``/``.usda``/``.usdc``/``.usdz``) to :class:`CommonSchema`.

The parser has two audiences and handles both from one code path:

* **Stages written by :class:`~cyberwave_robot_format.usd.exporter.USDExporter`.**
  Every ``cyberwave:``-namespaced sidecar is treated as authoritative, which makes
  the round trip exact — including the fields UsdPhysics cannot express at all
  (mimic, armature, jerkmax, motor electricals) and the many it can only express
  in ``float32``.
* **Foreign stages** from Isaac Sim, Omniverse, onshape-to-robot or a hand-written
  ``.usda``. With no sidecars present, structure is inferred from the native
  UsdPhysics encoding: prim types give joint kinds, ``physics:body0``/``body1``
  give the kinematic tree, ``physics:localRot0``/``localRot1`` plus
  ``physics:axis`` give the joint frame and axis, and degrees are converted back
  to radians.

Unauthored is always read as ``None`` rather than as USD's fallback, so
UsdPhysics sentinels (``physics:mass = 0`` meaning "derive from density",
``physics:centerOfMass = (-inf, -inf, -inf)``, ``physics:lowerLimit = -inf``
meaning "unlimited") never leak into the schema as data.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

from cyberwave_robot_format.core import BaseParser, ParseContext
from cyberwave_robot_format.schema import (
    Actuator,
    ActuatorType,
    Collision,
    CollisionConfig,
    CollisionExclude,
    CollisionGroup,
    CollisionPair,
    CommonSchema,
    Contact,
    ContactSurface,
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
    Physics,
    PhysicsSolver,
    Pose,
    Quaternion,
    Scene,
    Sensor,
    Vector3,
    Visual,
)
from cyberwave_robot_format.usd import common as c
from cyberwave_robot_format.usd import frames as f
from cyberwave_robot_format.usd import tokens as t

logger = logging.getLogger(__name__)

USD_SUFFIXES = frozenset({".usd", ".usda", ".usdc", ".usdz"})


class USDParser(BaseParser):
    """Parser for OpenUSD stages carrying physics and articulation data.

    Args:
        allow_external_layers: Whether composition may pull in layers from outside
            the parsed file's own directory. Defaults to ``True`` because layering is
            the defining feature of USD — real assets routinely reference sibling
            directories — but see the security note below before pointing this at
            input you did not author.

    Security:
        Opening a USD stage *composes* it: ``subLayers``, ``references``, ``payloads``
        and variants are resolved and their content becomes part of the result. A
        hostile file can therefore name an absolute path and pull that file's prims
        into the schema you get back:

        .. code-block:: text

            #usda 1.0
            (
                subLayers = [@/home/someone/private/robot.usda@]
            )

        Any service that parses an uploaded stage and returns (or stores) the
        resulting schema would be handing back the content of local USD files it
        never meant to expose. Pass ``allow_external_layers=False`` for untrusted
        input: every composed layer is then required to resolve inside the input
        file's directory, and anything escaping it raises ``ValueError``. Symlinks
        are resolved before the check, so they cannot be used to step outside.

        Two things this does *not* need to defend against, verified against
        ``usd-core`` 26.8: the default asset resolver does not fetch ``http(s)://``
        references, so there is no SSRF vector, and ``.usdz`` is read in place rather
        than extracted.
    """

    def __init__(self, allow_external_layers: bool = True) -> None:
        self.allow_external_layers = allow_external_layers

    def can_parse(self, file_path: str | Path) -> bool:
        """Whether ``file_path`` looks like a USD file.

        Suffix first, then a magic-byte sniff so a ``.usd`` of unknown flavour
        (and an extension-less file) is still recognized: ``#usda`` for ASCII,
        ``PXR-USDC`` for crate, ``PK\\x03\\x04`` for a usdz package.
        """
        path = Path(file_path)
        if path.suffix.lower() in USD_SUFFIXES:
            return True
        return self._sniff_suffix(path) is not None

    def parse(self, input_path: str | Path) -> CommonSchema:
        """Parse a USD file into a :class:`CommonSchema`.

        A file whose name does not end in a USD suffix is staged through a
        temporary copy that does, because ``Usd.Stage.Open`` resolves the layer's
        encoding from the extension and hard-errors without one.
        """
        input_path = Path(input_path)
        context = ParseContext(file_path=input_path, base_dir=input_path.parent)
        allowed_roots = {input_path.parent.resolve()}

        if input_path.suffix.lower() in USD_SUFFIXES:
            stage = self._open(input_path)
            self._assert_layers_within(stage, allowed_roots)
            return self.parse_stage(stage, context)

        suffix = self._sniff_suffix(input_path)
        if suffix is None:
            raise ValueError(
                f"{input_path} does not look like a USD file (expected one of "
                f"{sorted(USD_SUFFIXES)}, or USD magic bytes)."
            )
        import shutil
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            staged = Path(tmp_dir) / f"{input_path.stem}{suffix}"
            shutil.copyfile(input_path, staged)
            # The staging directory is a legitimate root too: relative arcs inside the
            # file resolve against the copy, not against the original location.
            stage = self._open(staged)
            self._assert_layers_within(
                stage, allowed_roots | {Path(tmp_dir).resolve()}
            )
            return self.parse_stage(stage, context)

    def _assert_layers_within(self, stage: Any, allowed_roots: set[Path]) -> None:
        """Reject a stage whose composition reached outside ``allowed_roots``.

        Checked against ``stage.GetUsedLayers()`` — what USD *actually* resolved —
        rather than by scanning the arcs the file declares. That way it covers every
        composition mechanism (subLayers, references, payloads, inherits,
        specializes, variants) including ones added to USD later, and it sees through
        relative paths and symlinks because the resolved ``realPath`` is compared.

        Anonymous layers have no ``realPath`` and are skipped: they are the in-memory
        session layer and anything built by ``parse_string``.
        """
        if self.allow_external_layers:
            return

        offenders: list[str] = []
        for layer in stage.GetUsedLayers():
            real_path = getattr(layer, "realPath", "") or ""
            if not real_path:
                continue
            resolved = Path(real_path).resolve()
            if not any(_is_within(resolved, root) for root in allowed_roots):
                offenders.append(str(resolved))

        if offenders:
            roots = ", ".join(sorted(str(root) for root in allowed_roots))
            raise ValueError(
                "USD composition referenced layers outside the permitted "
                f"directory ({roots}): {', '.join(sorted(offenders))}. "
                "Pass allow_external_layers=True if this input is trusted."
            )

    def _open(self, path: Path) -> Any:
        pxr = c.require_pxr()
        try:
            stage = pxr.Usd.Stage.Open(str(path))
        except pxr.Tf.ErrorException as exc:
            raise ValueError(f"Could not open USD stage from {path}: {exc}") from exc
        if stage is None:
            raise ValueError(f"Could not open USD stage from {path}")
        return stage

    @staticmethod
    def _sniff_suffix(path: Path) -> str | None:
        """Return the USD suffix implied by the file's magic bytes, if any."""
        try:
            with open(path, "rb") as handle:
                head = handle.read(8)
        except OSError:
            return None
        if head.startswith(b"#usda"):
            return ".usda"
        if head.startswith(b"PXR-USDC"):
            return ".usdc"
        if head.startswith(b"PK\x03\x04"):
            return ".usdz"
        return None

    def parse_string(self, usda_text: str) -> CommonSchema:
        """Parse ``.usda`` text into a :class:`CommonSchema`.

        The counterpart of ``USDExporter.export_to_string`` — the entry point that
        lets the tests exercise the round trip without touching the filesystem.
        """
        pxr = c.require_pxr()
        layer = pxr.Sdf.Layer.CreateAnonymous(".usda")
        try:
            imported = layer.ImportFromString(usda_text)
        except pxr.Tf.ErrorException as exc:
            # Malformed input raises rather than returning False, so both have to be
            # handled to keep a ValueError as the parser's contract for bad input.
            raise ValueError(f"Could not parse the supplied text as USDA: {exc}") from exc
        if not imported:
            raise ValueError("Could not parse the supplied text as USDA.")
        stage = pxr.Usd.Stage.Open(layer)
        if stage is None:
            raise ValueError("Could not open a USD stage from the supplied USDA text.")
        # Text has no directory to anchor against, so under restriction the only
        # acceptable layers are the anonymous ones this call created.
        self._assert_layers_within(stage, set())
        context = ParseContext(file_path=Path("<string>"), base_dir=Path("."))
        return self.parse_stage(stage, context)

    def parse_stage(
        self, stage: Any, context: ParseContext | None = None
    ) -> CommonSchema:
        """Parse an already-open ``Usd.Stage``."""
        if context is None:
            context = ParseContext(file_path=Path("<stage>"), base_dir=Path("."))
        return _StageReader(stage, context).read()


class _StageReader:
    """Per-parse mutable state, so :class:`USDParser` itself stays reusable."""

    def __init__(self, stage: Any, context: ParseContext) -> None:
        self.pxr = c.require_pxr()
        self.stage = stage
        self.context = context
        self.root_prim: Any = None
        self.link_name_by_path: dict[str, str] = {}
        self.link_prims: list[Any] = []

    # ------------------------------------------------------------------

    def read(self) -> CommonSchema:
        self.root_prim = self._find_root_prim()
        schema = CommonSchema(metadata=self._read_metadata())
        schema.extensions = c.read_extensions(self.root_prim, t.CD_EXTENSIONS)

        schema.links = self._read_links()
        schema.joints = self._read_joints()
        schema.actuators = self._read_actuators()
        schema.sensors = self._read_sensors()
        schema.contacts = self._read_contacts()
        schema.collision_config = self._read_collision_config()
        schema.scene = self._read_scene()
        schema.physics = self._read_physics()
        return schema

    def _find_root_prim(self) -> Any:
        """Locate the prim that stands for the robot.

        Preference order: the ``PhysicsArticulationRootAPI`` prim (what the
        exporter writes, and the standard marker in Isaac/Omniverse assets), then
        the stage's ``defaultPrim``, then the pseudo-root. Falling back to the
        pseudo-root keeps a bare stage of rigid bodies parseable.
        """
        pxr = self.pxr
        roots = [
            prim
            for prim in self.stage.Traverse()
            if prim.HasAPI(pxr.UsdPhysics.ArticulationRootAPI)
        ]
        if roots:
            if len(roots) > 1:
                # CommonSchema can hold a whole scene, but one parse produces one
                # articulation; say so rather than returning a silent subset.
                self.context.add_warning(
                    f"Stage declares {len(roots)} articulation roots "
                    f"({', '.join(str(prim.GetPath()) for prim in roots)}); "
                    f"only {roots[0].GetPath()} is parsed."
                )
            return roots[0]
        default_prim = self.stage.GetDefaultPrim()
        if default_prim and default_prim.IsValid():
            self.context.add_warning(
                "No PhysicsArticulationRootAPI prim found; using the stage's "
                "defaultPrim as the robot root."
            )
            return default_prim
        self.context.add_warning(
            "No articulation root or defaultPrim found; parsing the whole stage."
        )
        return self.stage.GetPseudoRoot()

    def _read_metadata(self) -> Metadata:
        stored = c.read_custom_data_json(self.root_prim, t.CD_METADATA)
        if isinstance(stored, dict) and "name" in stored:
            return Metadata(
                name=str(stored["name"]),
                version=str(stored.get("version", "1.0")),
                author=stored.get("author"),
                description=stored.get("description"),
                source_format=stored.get("source_format"),
                creation_date=stored.get("creation_date"),
                units=str(stored.get("units", "SI")),
                extensions=stored.get("extensions") or {},
            )
        name = (
            c.schema_name_from_prim(self.root_prim)
            if self.root_prim.GetPath() != self.pxr.Sdf.Path.absoluteRootPath
            else "usd_stage"
        )
        return Metadata(name=name, source_format="usd")

    # -- ordering ------------------------------------------------------

    def _ordered(
        self, container: Any, key: str, candidates: list[Any]
    ) -> list[Any]:
        """Order ``candidates`` by the ``cyberwave:order`` list on ``container``.

        ``list.__eq__`` is order-sensitive and USD does not guarantee child
        ordering, so the exporter records the intended order as prim paths
        relative to the prim carrying the customData. Candidates missing from the
        list (a foreign stage, or a prim added after export) keep their traversal
        position at the end rather than being dropped.
        """
        order = c.read_custom_data_json(container, t.CD_ORDER)
        names = (order or {}).get(key) if isinstance(order, dict) else None
        if not names:
            return candidates

        base = container.GetPath()
        remaining = {
            str(prim.GetPath().MakeRelativePath(base)): prim for prim in candidates
        }
        ordered = [remaining.pop(name) for name in names if name in remaining]
        ordered.extend(remaining.values())
        return ordered

    # -- links ---------------------------------------------------------

    def _read_links(self) -> list[Link]:
        pxr = self.pxr
        candidates = [
            prim
            for prim in pxr.Usd.PrimRange(self.root_prim)
            if prim.HasAPI(pxr.UsdPhysics.RigidBodyAPI)
        ]
        self.link_prims = self._ordered(self.root_prim, "links", candidates)

        links: list[Link] = []
        for prim in self.link_prims:
            name = c.schema_name_from_prim(prim)
            self.link_name_by_path[str(prim.GetPath())] = name
            visuals, collisions = self._read_link_geometry(prim)
            links.append(
                Link(
                    name=name,
                    mass=self._read_mass(prim),
                    center_of_mass=self._read_center_of_mass(prim),
                    inertia=self._read_inertia(prim),
                    visuals=visuals,
                    collisions=collisions,
                    extensions=c.merge_usd_passthrough(
                        c.read_extensions(prim, t.CD_EXTENSIONS),
                        c.read_usd_passthrough(prim, t.USD_PASSTHROUGH_LINK),
                    ),
                )
            )
        return links

    def _read_mass(self, prim: Any) -> float:
        sidecar = c.read_double(prim, t.A_LINK_MASS)
        if sidecar is not None:
            return sidecar
        native = c.read_native(self.pxr.UsdPhysics.MassAPI(prim).GetMassAttr())
        return float(native) if native is not None else 0.0

    def _read_center_of_mass(self, prim: Any) -> Vector3:
        sidecar = c.read_vector3(prim, t.A_LINK_CENTER_OF_MASS)
        if sidecar is not None:
            return sidecar
        native = c.read_native(
            self.pxr.UsdPhysics.MassAPI(prim).GetCenterOfMassAttr()
        )
        if native is None or not all(math.isfinite(v) for v in native):
            # (-inf, -inf, -inf) is UsdPhysics' "derive it from the geometry"
            # sentinel, not a centre of mass.
            return Vector3()
        return c.gf_to_vector3(native)

    def _read_inertia(self, prim: Any) -> Inertia:
        sidecar = c.read_float_list(prim, t.A_LINK_INERTIA)
        if sidecar is not None and len(sidecar) == 6:
            return Inertia(
                ixx=sidecar[0],
                iyy=sidecar[1],
                izz=sidecar[2],
                ixy=sidecar[3],
                ixz=sidecar[4],
                iyz=sidecar[5],
            )
        mass_api = self.pxr.UsdPhysics.MassAPI(prim)
        diagonal = c.read_native(mass_api.GetDiagonalInertiaAttr())
        if diagonal is None:
            return Inertia()
        principal = c.read_native(mass_api.GetPrincipalAxesAttr())
        quaternion = (
            c.gf_to_quaternion(principal) if principal is not None else Quaternion()
        )
        if (
            quaternion.x == 0.0
            and quaternion.y == 0.0
            and quaternion.z == 0.0
            and quaternion.w == 0.0
        ):
            # (0, 0, 0, 0) is the schema fallback, not a rotation.
            quaternion = Quaternion()
        return f.inertia_from_principal(
            (float(diagonal[0]), float(diagonal[1]), float(diagonal[2])), quaternion
        )

    # -- visuals / collisions ------------------------------------------

    def _read_link_geometry(self, prim: Any) -> tuple[list[Visual], list[Collision]]:
        visuals_scope = prim.GetChild(t.SCOPE_VISUALS)
        collisions_scope = prim.GetChild(t.SCOPE_COLLISIONS)

        if visuals_scope.IsValid() or collisions_scope.IsValid():
            visual_prims = (
                self._ordered(prim, "visuals", list(visuals_scope.GetChildren()))
                if visuals_scope.IsValid()
                else []
            )
            collision_prims = (
                self._ordered(prim, "collisions", list(collisions_scope.GetChildren()))
                if collisions_scope.IsValid()
                else []
            )
            return (
                [self._read_visual(child) for child in visual_prims],
                [self._read_collision(child) for child in collision_prims],
            )

        return self._read_foreign_link_geometry(prim)

    def _read_foreign_link_geometry(
        self, prim: Any
    ) -> tuple[list[Visual], list[Collision]]:
        """Classify a foreign stage's gprims into visuals and collisions.

        Mirrors ``MJCFParser``'s handling of a geom with no explicit group: a prim
        with ``PhysicsCollisionAPI`` is a collider, a visible prim whose purpose is
        renderable is a visual, and a prim that is both contributes to both lists.
        """
        pxr = self.pxr
        visuals: list[Visual] = []
        collisions: list[Collision] = []
        for child in pxr.Usd.PrimRange(prim):
            if child == prim or not child.IsA(pxr.UsdGeom.Gprim):
                continue
            if child.HasAPI(pxr.UsdPhysics.CollisionAPI):
                collisions.append(self._read_collision(child))
            if self._is_renderable(child):
                visuals.append(self._read_visual(child))
        return visuals, collisions

    def _is_renderable(self, prim: Any) -> bool:
        pxr = self.pxr
        imageable = pxr.UsdGeom.Imageable(prim)
        if not imageable:
            return True
        try:
            if imageable.ComputeVisibility() == pxr.UsdGeom.Tokens.invisible:
                return False
            purpose = imageable.ComputePurpose()
        except Exception:  # pragma: no cover - defensive against pxr API drift
            return True
        return purpose not in (pxr.UsdGeom.Tokens.proxy, pxr.UsdGeom.Tokens.guide)

    def _read_visual(self, prim: Any) -> Visual:
        pose, scale = self._read_pose_and_scale(prim)
        # Absence of the name sidecar on a visual must read back as None, so this
        # deliberately does not fall back to the prim name the way links do.
        return Visual(
            name=c.read_string(prim, t.A_NAME),
            pose=pose,
            geometry=self._read_geometry(prim, scale),
            material=self._read_bound_material(prim),
            extensions=c.read_extensions(prim, t.CD_EXTENSIONS),
        )

    def _read_collision(self, prim: Any) -> Collision:
        pose, scale = self._read_pose_and_scale(prim)
        group = c.read_string(prim, t.A_COLLISION_GROUP)
        material_api = self.pxr.UsdPhysics.MaterialAPI
        mu_static = c.read_double(prim, t.A_COLLISION_MU_STATIC)
        mu_dynamic = c.read_double(prim, t.A_COLLISION_MU_DYNAMIC)
        restitution = c.read_double(prim, t.A_COLLISION_RESTITUTION)
        if mu_static is None and mu_dynamic is None and restitution is None:
            mu_static, mu_dynamic, restitution = self._read_bound_physics_material(
                prim, material_api
            )
        return Collision(
            name=c.read_string(prim, t.A_NAME),
            pose=pose,
            geometry=self._read_geometry(prim, scale),
            group=group if group is not None else "default",
            mu_static=mu_static,
            mu_dynamic=mu_dynamic,
            restitution=restitution,
            stiffness=c.read_double(prim, t.A_COLLISION_STIFFNESS),
            damping=c.read_double(prim, t.A_COLLISION_DAMPING),
            extensions=c.merge_usd_passthrough(
                c.read_extensions(prim, t.CD_EXTENSIONS),
                c.read_usd_passthrough(prim, t.USD_PASSTHROUGH_COLLISION),
            ),
        )

    def _read_bound_physics_material(
        self, prim: Any, material_api: Any
    ) -> tuple[float | None, float | None, float | None]:
        """Recover friction/restitution from a bound physics material (foreign path)."""
        pxr = self.pxr
        try:
            binding = pxr.UsdShade.MaterialBindingAPI(prim)
            material = binding.ComputeBoundMaterial(
                t.TOK_MATERIAL_PURPOSE_PHYSICS
            )[0]
            if not material:
                return None, None, None
            api = material_api(material.GetPrim())
            values = [
                c.read_native(api.GetStaticFrictionAttr()),
                c.read_native(api.GetDynamicFrictionAttr()),
                c.read_native(api.GetRestitutionAttr()),
            ]
            return tuple(  # type: ignore[return-value]
                None if value is None else float(value) for value in values
            )
        except Exception:  # pragma: no cover - defensive
            return None, None, None

    def _read_pose_and_scale(self, prim: Any) -> tuple[Pose, Vector3 | None]:
        """Return the prim's local pose and its ``xformOp:scale``, if any.

        The ``xformOp:transform`` attribute is read directly rather than via the
        composed local transformation, because the composed matrix folds in
        ``xformOp:scale`` — which encodes box dimensions and mesh scale, not pose.
        """
        pxr = self.pxr
        scale_attr = prim.GetAttribute("xformOp:scale")
        scale = (
            c.gf_to_vector3(scale_attr.Get())
            if scale_attr and scale_attr.HasAuthoredValue()
            else None
        )

        transform_attr = prim.GetAttribute("xformOp:transform")
        if transform_attr and transform_attr.HasAuthoredValue():
            return f.matrix_to_pose(c.gf_matrix_to_np(transform_attr.Get())), scale

        xformable = pxr.UsdGeom.Xformable(prim)
        if not xformable or not xformable.GetOrderedXformOps():
            return Pose(), scale

        # Foreign stages use translate/orient/rotateXYZ ops; decompose the local
        # matrix and drop the scale factors so they are not mistaken for pose.
        matrix = c.gf_matrix_to_np(xformable.GetLocalTransformation())
        rotation = matrix[:3, :3].copy()
        norms = [float(v) for v in (rotation**2).sum(axis=0) ** 0.5]
        if all(n > 1e-12 for n in norms):
            if scale is None:
                scale = Vector3(x=norms[0], y=norms[1], z=norms[2])
            for column, norm in enumerate(norms):
                rotation[:, column] /= norm
        pose = Pose(
            position=Vector3(
                x=float(matrix[0, 3]), y=float(matrix[1, 3]), z=float(matrix[2, 3])
            ),
            orientation=f.matrix_to_quat(rotation),
        )
        return pose, scale

    def _read_geometry(self, prim: Any, scale: Vector3 | None) -> Geometry | None:
        type_token = c.read_string(prim, t.A_GEOM_TYPE)
        if type_token is not None:
            # Our own output: the sidecars are complete, so an absent one means the
            # schema field was None rather than that we should guess from USD.
            return Geometry(
                type=_geometry_type_from_token(type_token),
                size=c.read_vector3(prim, t.A_GEOM_SIZE),
                radius=c.read_double(prim, t.A_GEOM_RADIUS),
                length=c.read_double(prim, t.A_GEOM_LENGTH),
                filename=c.read_string(prim, t.A_GEOM_MESH_FILENAME),
                scale=c.read_vector3(prim, t.A_GEOM_SCALE),
                extensions=c.read_extensions(prim, t.CD_GEOMETRY_EXTENSIONS),
            )
        if self._is_cyberwave_geometry_owner(prim):
            # A gprim we wrote that has no type sidecar had `geometry=None`.
            return None
        return self._read_foreign_geometry(prim, scale)

    def _is_cyberwave_geometry_owner(self, prim: Any) -> bool:
        """Whether this visual/collision prim came from :class:`USDExporter`.

        Distinguishes "we wrote this prim and its ``Geometry`` was ``None``" from
        "this is a foreign gprim whose shape must be inferred". The ordering
        customData on the owning link is the marker, since it is written for every
        link that has any visual or collision.
        """
        link = prim.GetParent().GetParent()
        return bool(link) and link.IsValid() and (
            c.read_custom_data_json(link, t.CD_ORDER) is not None
        )

    def _read_foreign_geometry(
        self, prim: Any, scale: Vector3 | None
    ) -> Geometry | None:
        pxr = self.pxr
        type_name = str(prim.GetTypeName())
        geometry_type = t.PRIM_TYPE_TO_GEOMETRY_TYPE.get(type_name)
        if geometry_type is None:
            return None

        # UsdGeom shape attributes are read with `.Get()`, not `read_native`: an
        # unauthored `size`/`radius`/`height`/`width` is not missing data the way an
        # unauthored `physics:mass` is — the schema fallback *is* the dimension, and
        # hard-coding a second copy of it here is how the two drift apart.
        if geometry_type == GeometryType.BOX:
            edge = float(pxr.UsdGeom.Cube(prim).GetSizeAttr().Get())
            factors = scale or Vector3(x=1.0, y=1.0, z=1.0)
            return Geometry(
                type=GeometryType.BOX,
                size=Vector3(
                    x=edge * factors.x, y=edge * factors.y, z=edge * factors.z
                ),
            )

        if geometry_type == GeometryType.SPHERE:
            radius = float(pxr.UsdGeom.Sphere(prim).GetRadiusAttr().Get())
            if scale is not None and not (scale.x == scale.y == scale.z):
                # A non-uniformly scaled sphere is an ellipsoid; `size` holds the
                # semi-axes, matching the exporter's ELLIPSOID encoding.
                return Geometry(
                    type=GeometryType.ELLIPSOID,
                    size=Vector3(
                        x=radius * scale.x, y=radius * scale.y, z=radius * scale.z
                    ),
                )
            factor = scale.x if scale is not None else 1.0
            return Geometry(type=GeometryType.SPHERE, radius=radius * factor)

        if geometry_type in (GeometryType.CYLINDER, GeometryType.CAPSULE):
            shape = (
                pxr.UsdGeom.Cylinder(prim)
                if geometry_type == GeometryType.CYLINDER
                else pxr.UsdGeom.Capsule(prim)
            )
            return Geometry(
                type=geometry_type,
                radius=float(shape.GetRadiusAttr().Get()),
                length=float(shape.GetHeightAttr().Get()),
            )

        if geometry_type == GeometryType.PLANE:
            plane = pxr.UsdGeom.Plane(prim)
            return Geometry(
                type=GeometryType.PLANE,
                size=Vector3(
                    x=float(plane.GetWidthAttr().Get()),
                    y=float(plane.GetLengthAttr().Get()),
                    z=0.0,
                ),
            )

        if geometry_type == GeometryType.MESH:
            # A foreign UsdGeomMesh carries its points inline; there is no source
            # filename to recover, so the geometry is reported as a mesh with no
            # reference rather than fabricating one.
            self.context.add_warning(
                f"Mesh prim {prim.GetPath()} has inline points and no "
                f"{t.A_GEOM_MESH_FILENAME} reference; Geometry.filename is unset."
            )
            return Geometry(type=GeometryType.MESH, scale=scale)

        return Geometry(type=geometry_type)

    def _read_bound_material(self, prim: Any) -> Material | None:
        pxr = self.pxr
        try:
            binding = pxr.UsdShade.MaterialBindingAPI(prim)
            material = binding.ComputeBoundMaterial()[0]
        except Exception:  # pragma: no cover - defensive
            return None
        if not material:
            return None

        material_prim = material.GetPrim()
        name = self._material_name(material_prim, prim)

        color = c.read_float_list(material_prim, t.A_MATERIAL_COLOR)
        specular = c.read_float_list(material_prim, t.A_MATERIAL_SPECULAR)
        emissive = c.read_float_list(material_prim, t.A_MATERIAL_EMISSIVE)
        shininess = c.read_double(material_prim, t.A_MATERIAL_SHININESS)
        texture = c.read_string(material_prim, t.A_MATERIAL_TEXTURE)

        if color is None:
            color, foreign_texture = self._read_preview_surface(material)
            texture = texture or foreign_texture

        extensions = c.read_extensions(material_prim, t.CD_EXTENSIONS)
        if (
            name is None
            and color is None
            and specular is None
            and emissive is None
            and shininess is None
            and texture is None
            and not extensions
        ):
            return None
        return Material(
            name=name,
            color=color,
            texture=texture,
            specular=specular,
            emissive=emissive,
            shininess=shininess,
            extensions=extensions,
        )

    def _material_name(self, material_prim: Any, bound_to: Any) -> str | None:
        """Recover ``Material.name``, which may legitimately be ``None``.

        The exporter authors ``cyberwave:name`` for every *named* material, so its
        absence means the material was unnamed and got authored inline under the
        very prim it belongs to. That inline case is detected structurally — the
        material is a child of the prim binding it — rather than by comparing
        against the reserved prim name, so a material genuinely called
        ``"Material"`` still round-trips. The parent identity is what makes it
        exact: a visual whose ``geometry`` is ``None`` or ``UNKNOWN`` is authored as
        a plain ``Xform`` rather than a gprim, while a foreign stage may keep a
        material called ``"Material"`` anywhere at all.
        """
        explicit = c.read_string(material_prim, t.A_NAME)
        if explicit is not None:
            return explicit
        if (
            material_prim.GetName() == t.INLINE_MATERIAL_NAME
            and material_prim.GetParent() == bound_to
        ):
            return None
        return material_prim.GetName()

    def _read_preview_surface(
        self, material: Any
    ) -> tuple[list[float] | None, str | None]:
        """Best-effort ``diffuseColor``/``opacity``/texture recovery from the shader.

        Only reached for foreign stages: extracts a flat diffuse colour and degrades
        to ``None`` on any surprise rather than failing the parse.
        """
        pxr = self.pxr
        try:
            source = material.ComputeSurfaceSource()[0]
            if not source:
                return None, None
            shader = pxr.UsdShade.Shader(source.GetPrim())
            diffuse = shader.GetInput("diffuseColor") or shader.GetInput("baseColor")
            color: list[float] | None = None
            if diffuse:
                value = diffuse.Get()
                if value is not None:
                    color = [float(value[0]), float(value[1]), float(value[2])]
                    opacity = shader.GetInput("opacity")
                    if opacity and opacity.Get() is not None:
                        color.append(float(opacity.Get()))
            texture: str | None = None
            file_input = None
            for child in material.GetPrim().GetChildren():
                candidate = pxr.UsdShade.Shader(child)
                if candidate and candidate.GetIdAttr().Get() == "UsdUVTexture":
                    file_input = candidate.GetInput("file")
                    break
            if file_input and file_input.Get() is not None:
                texture = str(file_input.Get().path)
            return color, texture
        except Exception:  # pragma: no cover - defensive
            return None, None

    # -- joints --------------------------------------------------------

    def _read_joints(self) -> list[Joint]:
        pxr = self.pxr
        candidates = [
            prim
            for prim in pxr.Usd.PrimRange(self.root_prim)
            if prim.IsA(pxr.UsdPhysics.Joint)
        ]
        joints: list[Joint] = []
        for prim in self._ordered(self.root_prim, "joints", candidates):
            joints.append(self._read_joint(prim))
        return joints

    def _read_joint(self, prim: Any) -> Joint:
        pxr = self.pxr
        joint_type = self._read_joint_type(prim)
        usd_joint = pxr.UsdPhysics.Joint(prim)

        pose, axis = self._read_joint_frame(prim, usd_joint)
        sidecar_position = c.read_vector3(prim, t.A_JOINT_POSE_POSITION)
        sidecar_orientation = c.read_quaternion(prim, t.A_JOINT_POSE_ORIENTATION)
        if sidecar_position is not None or sidecar_orientation is not None:
            pose = Pose(
                position=(
                    pose.position if sidecar_position is None else sidecar_position
                ),
                orientation=(
                    pose.orientation
                    if sidecar_orientation is None
                    else sidecar_orientation
                ),
            )
        sidecar_axis = c.read_vector3(prim, t.A_JOINT_AXIS)
        if sidecar_axis is not None:
            axis = sidecar_axis

        return Joint(
            name=c.schema_name_from_prim(prim),
            type=joint_type,
            parent_link=self._read_joint_body(
                prim, usd_joint.GetBody0Rel(), t.A_JOINT_PARENT_LINK
            ),
            child_link=self._read_joint_body(
                prim, usd_joint.GetBody1Rel(), t.A_JOINT_CHILD_LINK
            ),
            pose=pose,
            axis=axis,
            limits=self._read_joint_limits(prim, joint_type),
            dynamics=self._read_joint_dynamics(prim),
            safety_controller=c.read_json_attr(prim, t.A_JOINT_SAFETY_CONTROLLER),
            calibration=c.read_json_attr(prim, t.A_JOINT_CALIBRATION),
            home_position=self._read_home_position(prim, joint_type),
            mimic=self._read_mimic(prim),
            extensions=self._read_joint_extensions(prim, joint_type),
        )

    def _read_joint_extensions(
        self, prim: Any, joint_type: JointType
    ) -> dict[str, Any]:
        """Joint extensions, plus any UsdPhysics-only properties worth keeping.

        ``physics:breakForce``, ``jointEnabled`` and friends are real UsdPhysics
        properties with no ``Joint`` field, so they are preserved the same way
        ``MJCFParser`` preserves MuJoCo's ``solref``/``condim`` — under
        ``extensions["usd"]`` — rather than being dropped on import.
        """
        extensions = c.merge_usd_passthrough(
            c.read_extensions(prim, t.CD_EXTENSIONS),
            c.read_usd_passthrough(prim, t.USD_PASSTHROUGH_JOINT),
        )
        drive = self._drive_attributes(prim, joint_type)
        if drive is not None:
            target_velocity = c.read_native(drive.GetTargetVelocityAttr())
            if target_velocity is not None and float(target_velocity) != 0.0:
                extensions = c.merge_usd_passthrough(
                    extensions,
                    {t.DRIVE_TARGET_VELOCITY_KEY: float(target_velocity)},
                )
        return extensions

    def _read_joint_type(self, prim: Any) -> JointType:
        token = c.read_string(prim, t.A_JOINT_TYPE)
        if token is not None:
            try:
                return JointType(token)
            except ValueError:
                self.context.add_warning(
                    f"Unknown {t.A_JOINT_TYPE} value {token!r} on {prim.GetPath()}; "
                    "falling back to the prim type."
                )
        type_name = str(prim.GetTypeName())
        joint_type = t.PRIM_TYPE_TO_JOINT_TYPE.get(type_name)
        if joint_type is None:
            self.context.add_warning(
                f"Prim {prim.GetPath()} has unrecognized joint type {type_name!r}; "
                "treating it as fixed."
            )
            return JointType.FIXED
        return joint_type

    def _read_joint_body(self, prim: Any, relationship: Any, sidecar: str) -> str:
        """Resolve ``physics:body0``/``body1`` to a schema link name.

        An empty relationship means the world frame in UsdPhysics, and ``"world"``
        is exactly the schema's reserved pseudo-link — so the two encodings map
        onto each other with no special case. A verbatim-name sidecar takes
        precedence, which is how a dangling link reference survives.
        """
        explicit = c.read_string(prim, sidecar)
        if explicit is not None:
            return explicit
        targets = relationship.GetTargets() if relationship else []
        if not targets:
            return t.WORLD_LINK
        target_path = str(targets[0])
        name = self.link_name_by_path.get(target_path)
        if name is not None:
            return name
        self.context.add_warning(
            f"Joint {prim.GetPath()} targets {target_path}, which is not a rigid "
            "body prim; using its prim name."
        )
        return target_path.rsplit("/", 1)[-1]

    def _read_joint_frame(self, prim: Any, usd_joint: Any) -> tuple[Pose, Vector3]:
        local_pos0 = c.read_native(usd_joint.GetLocalPos0Attr())
        local_rot0 = c.read_native(usd_joint.GetLocalRot0Attr())
        local_pos1 = c.read_native(usd_joint.GetLocalPos1Attr())
        local_rot1 = c.read_native(usd_joint.GetLocalRot1Attr())

        axis_token = t.REF_AXIS_TOKEN
        axis_attr = prim.GetAttribute("physics:axis")
        if axis_attr:
            value = axis_attr.Get()
            if value is not None:
                axis_token = str(value)

        return f.joint_pose_axis_from_frames(
            c.gf_to_vector3(local_pos0) if local_pos0 is not None else Vector3(),
            c.gf_to_quaternion(local_rot0) if local_rot0 is not None else Quaternion(),
            c.gf_to_vector3(local_pos1) if local_pos1 is not None else Vector3(),
            c.gf_to_quaternion(local_rot1) if local_rot1 is not None else Quaternion(),
            axis_token,
        )

    def _read_joint_limits(
        self, prim: Any, joint_type: JointType
    ) -> JointLimits | None:
        if c.read_bool(prim, t.A_JOINT_HAS_LIMITS):
            return JointLimits(
                lower=c.read_double(prim, t.A_JOINT_LIMIT_LOWER),
                upper=c.read_double(prim, t.A_JOINT_LIMIT_UPPER),
                effort=c.read_double(prim, t.A_JOINT_LIMIT_EFFORT),
                velocity=c.read_double(prim, t.A_JOINT_LIMIT_VELOCITY),
                acceleration=c.read_double(prim, t.A_JOINT_LIMIT_ACCELERATION),
                jerkmax=c.read_double(prim, t.A_JOINT_LIMIT_JERKMAX),
            )

        lower_attr = prim.GetAttribute("physics:lowerLimit")
        upper_attr = prim.GetAttribute("physics:upperLimit")
        lower = c.read_native(lower_attr)
        upper = c.read_native(upper_attr)
        if lower is None and upper is None:
            return None
        is_angular = joint_type in t.ANGULAR_JOINT_TYPES
        return JointLimits(
            lower=_from_usd_angle(lower, is_angular),
            upper=_from_usd_angle(upper, is_angular),
        )

    def _is_cyberwave_authored(self, prim: Any) -> bool:
        """Whether this joint prim was written by :class:`USDExporter`.

        ``cyberwave:joint:type`` is authored on every joint the exporter writes, so
        it doubles as the marker that the sidecars are complete. That matters
        because the native attributes are *derived* on our own output — a passive
        spring-damper's ``targetPosition``, for instance — and reading them back as
        schema data would invent values the source schema never had.
        """
        attribute = prim.GetAttribute(t.A_JOINT_TYPE)
        return bool(attribute) and attribute.HasAuthoredValue()

    def _read_joint_dynamics(self, prim: Any) -> JointDynamics | None:
        if c.read_bool(prim, t.A_JOINT_HAS_DYNAMICS):
            return JointDynamics(
                damping=c.read_double(prim, t.A_JOINT_DYN_DAMPING) or 0.0,
                friction=c.read_double(prim, t.A_JOINT_DYN_FRICTION) or 0.0,
                spring_reference=c.read_double(prim, t.A_JOINT_DYN_SPRING_REFERENCE)
                or 0.0,
                spring_stiffness=c.read_double(prim, t.A_JOINT_DYN_SPRING_STIFFNESS)
                or 0.0,
                armature=c.read_double(prim, t.A_JOINT_DYN_ARMATURE),
                backlash=c.read_double(prim, t.A_JOINT_DYN_BACKLASH),
            )
        if self._is_cyberwave_authored(prim):
            return None
        return self._infer_dynamics_from_drive(prim)

    def _infer_dynamics_from_drive(self, prim: Any) -> JointDynamics | None:
        """Read a foreign stage's drive gains as joint dynamics.

        In USD a drive whose stiffness pulls towards a target and whose damping
        resists motion *is* a spring-damper, so this is the inverse of what the
        exporter writes for an un-actuated joint with dynamics.
        """
        joint_type = self._read_joint_type(prim)
        drive = self._drive_attributes(prim, joint_type)
        if drive is None:
            return None
        stiffness = c.read_native(drive.GetStiffnessAttr())
        damping = c.read_native(drive.GetDampingAttr())
        if stiffness is None and damping is None:
            return None
        return JointDynamics(
            damping=float(damping) if damping is not None else 0.0,
            spring_stiffness=float(stiffness) if stiffness is not None else 0.0,
        )

    def _read_home_position(self, prim: Any, joint_type: JointType) -> float | None:
        sidecar = c.read_double(prim, t.A_JOINT_HOME_POSITION)
        if sidecar is not None:
            return sidecar
        if self._is_cyberwave_authored(prim):
            # Our own drive targetPosition is derived from home_position or from a
            # spring rest position; reading it back would turn `None` into 0.0.
            return None
        drive = self._drive_attributes(prim, joint_type)
        if drive is None:
            return None
        target = c.read_native(drive.GetTargetPositionAttr())
        if target is None:
            return None
        return _from_usd_angle(target, joint_type in t.ANGULAR_JOINT_TYPES)

    def _drive_attributes(self, prim: Any, joint_type: JointType) -> Any | None:
        instance = t.DRIVE_INSTANCE_FOR_JOINT_TYPE.get(joint_type)
        if instance is None:
            return None
        api = self.pxr.UsdPhysics.DriveAPI(prim, instance)
        return api if api else None

    def _read_mimic(self, prim: Any) -> MimicJoint | None:
        driver = c.read_string(prim, t.A_JOINT_MIMIC_JOINT)
        if driver is None:
            return None
        return MimicJoint(
            joint=driver,
            multiplier=_or_default(
                c.read_double(prim, t.A_JOINT_MIMIC_MULTIPLIER), MimicJoint.multiplier
            ),
            offset=_or_default(
                c.read_double(prim, t.A_JOINT_MIMIC_OFFSET), MimicJoint.offset
            ),
        )

    # -- actuators / sensors / contacts ---------------------------------

    def _read_actuators(self) -> list[Actuator]:
        candidates = [
            prim
            for prim in self.pxr.Usd.PrimRange(self.root_prim)
            if prim.GetAttribute(t.A_ACTUATOR_JOINT)
            and prim.GetAttribute(t.A_ACTUATOR_JOINT).HasAuthoredValue()
        ]
        actuators: list[Actuator] = []
        for prim in self._ordered(self.root_prim, "actuators", candidates):
            type_token = c.read_string(prim, t.A_ACTUATOR_TYPE)
            try:
                actuator_type = (
                    ActuatorType(type_token)
                    if type_token is not None
                    else ActuatorType.POSITION
                )
            except ValueError:
                self.context.add_warning(
                    f"Unknown actuator type {type_token!r} on {prim.GetPath()}; "
                    "defaulting to 'position'."
                )
                actuator_type = ActuatorType.POSITION
            actuators.append(
                Actuator(
                    name=c.schema_name_from_prim(prim),
                    joint=c.read_string(prim, t.A_ACTUATOR_JOINT) or "",
                    type=actuator_type,
                    torque_constant=c.read_double(prim, t.A_ACTUATOR_TORQUE_CONSTANT),
                    gear_ratio=c.read_double(prim, t.A_ACTUATOR_GEAR_RATIO),
                    max_current=c.read_double(prim, t.A_ACTUATOR_MAX_CURRENT),
                    max_torque=c.read_double(prim, t.A_ACTUATOR_MAX_TORQUE),
                    max_velocity=c.read_double(prim, t.A_ACTUATOR_MAX_VELOCITY),
                    kp=c.read_double(prim, t.A_ACTUATOR_KP),
                    ki=c.read_double(prim, t.A_ACTUATOR_KI),
                    kd=c.read_double(prim, t.A_ACTUATOR_KD),
                    control_range=c.read_float_pair(prim, t.A_ACTUATOR_CONTROL_RANGE),
                    force_range=c.read_float_pair(prim, t.A_ACTUATOR_FORCE_RANGE),
                    resistance=c.read_double(prim, t.A_ACTUATOR_RESISTANCE),
                    inductance=c.read_double(prim, t.A_ACTUATOR_INDUCTANCE),
                    efficiency=c.read_double(prim, t.A_ACTUATOR_EFFICIENCY),
                    extensions=c.read_extensions(prim, t.CD_EXTENSIONS),
                )
            )
        return actuators or self._infer_actuators_from_drives()

    def _infer_actuators_from_drives(self) -> list[Actuator]:
        """Derive actuators from a foreign stage's ``UsdPhysicsDriveAPI`` instances.

        A stage with no ``/Actuators`` records is either genuinely passive or was
        authored elsewhere — and in USD, a motorized joint is expressed *only* as a
        drive. Without this, importing an Isaac Sim robot and re-exporting it to
        MJCF would silently produce an unactuated model. The derived actuator is
        named ``<joint>_drive`` and typed ``position``, which is what a
        stiffness/damping drive with a position target actually is.
        """
        pxr = self.pxr
        actuators: list[Actuator] = []
        for prim in pxr.Usd.PrimRange(self.root_prim):
            if not prim.IsA(pxr.UsdPhysics.Joint) or self._is_cyberwave_authored(prim):
                continue
            joint_type = self._read_joint_type(prim)
            drive = self._drive_attributes(prim, joint_type)
            if drive is None:
                continue
            stiffness = c.read_native(drive.GetStiffnessAttr())
            damping = c.read_native(drive.GetDampingAttr())
            max_force = c.read_native(drive.GetMaxForceAttr())
            if stiffness is None and damping is None and max_force is None:
                continue
            joint_name = c.schema_name_from_prim(prim)
            actuators.append(
                Actuator(
                    name=f"{joint_name}_drive",
                    joint=joint_name,
                    type=ActuatorType.POSITION,
                    kp=float(stiffness) if stiffness is not None else None,
                    kd=float(damping) if damping is not None else None,
                    max_torque=(
                        float(max_force)
                        if max_force is not None and math.isfinite(float(max_force))
                        else None
                    ),
                )
            )
        return actuators

    def _read_sensors(self) -> list[Sensor]:
        candidates = [
            prim
            for prim in self.pxr.Usd.PrimRange(self.root_prim)
            if prim.GetAttribute(t.A_SENSOR_TYPE)
            and prim.GetAttribute(t.A_SENSOR_TYPE).HasAuthoredValue()
        ]
        sensors: list[Sensor] = []
        for prim in self._ordered(self.root_prim, "sensors", candidates):
            pose, _ = self._read_pose_and_scale(prim)
            parameters = c.read_json_attr(prim, t.A_SENSOR_PARAMETERS)
            sensors.append(
                Sensor(
                    name=c.schema_name_from_prim(prim),
                    type=c.read_string(prim, t.A_SENSOR_TYPE) or "",
                    parent_link=self._sensor_parent_link(prim),
                    pose=pose,
                    update_rate=c.read_double(prim, t.A_SENSOR_UPDATE_RATE),
                    noise=c.read_json_attr(prim, t.A_SENSOR_NOISE),
                    parameters=parameters if isinstance(parameters, dict) else {},
                    extensions=c.read_extensions(prim, t.CD_EXTENSIONS),
                )
            )
        return sensors

    def _sensor_parent_link(self, prim: Any) -> str:
        """Recover a sensor's parent link from prim ancestry, or its sidecar."""
        explicit = c.read_string(prim, t.A_SENSOR_PARENT_LINK)
        if explicit is not None:
            return explicit
        ancestor = prim.GetParent()
        while ancestor and ancestor.IsValid():
            name = self.link_name_by_path.get(str(ancestor.GetPath()))
            if name is not None:
                return name
            ancestor = ancestor.GetParent()
        return t.WORLD_LINK

    def _read_contacts(self) -> list[Contact]:
        candidates = [
            prim
            for prim in self.pxr.Usd.PrimRange(self.root_prim)
            if prim.GetAttribute(t.A_CONTACT_LINK)
            and prim.GetAttribute(t.A_CONTACT_LINK).HasAuthoredValue()
        ]
        contacts: list[Contact] = []
        for prim in self._ordered(self.root_prim, "contacts", candidates):
            contacts.append(
                Contact(
                    name=c.schema_name_from_prim(prim),
                    link=c.read_string(prim, t.A_CONTACT_LINK) or "",
                    surface=_read_contact_surface(prim, t.CONTACT_SURFACE_PREFIX),
                    extensions=c.read_extensions(prim, t.CD_EXTENSIONS),
                )
            )
        return contacts

    # -- collision config ----------------------------------------------

    def _read_collision_config(self) -> CollisionConfig | None:
        config_prim = self.root_prim.GetChild(t.SCOPE_COLLISION_CONFIG)
        groups_prim = self.root_prim.GetChild(t.SCOPE_COLLISION_GROUPS)
        if not config_prim.IsValid() and not groups_prim.IsValid():
            return None

        groups: dict[str, CollisionGroup] = {}
        group_passthrough: dict[str, Any] = {}
        if groups_prim.IsValid():
            for child in groups_prim.GetChildren():
                name = c.schema_name_from_prim(child)
                harvested = c.read_usd_passthrough(child, t.USD_PASSTHROUGH_GROUP)
                if harvested:
                    group_passthrough[name] = harvested
                defaults = CollisionGroup(name=name)
                groups[name] = CollisionGroup(
                    name=name,
                    contype=_or_default(
                        c.read_int(child, t.A_GROUP_CONTYPE), defaults.contype
                    ),
                    conaffinity=_or_default(
                        c.read_int(child, t.A_GROUP_CONAFFINITY), defaults.conaffinity
                    ),
                )

        excludes: list[CollisionExclude] = []
        pairs: list[CollisionPair] = []
        extensions: dict[str, Any] = {}
        if config_prim.IsValid():
            extensions = c.read_extensions(config_prim, t.CD_EXTENSIONS)
            stored = c.read_custom_data_json(config_prim, t.CD_EXCLUDES) or []
            for entry in stored:
                if isinstance(entry, list) and len(entry) == 2:
                    excludes.append(
                        CollisionExclude(body1=str(entry[0]), body2=str(entry[1]))
                    )
            pairs_prim = config_prim.GetChild(t.SCOPE_PAIRS)
            if pairs_prim.IsValid():
                for child in sorted(
                    pairs_prim.GetChildren(), key=lambda p: _pair_index(p.GetName())
                ):
                    pairs.append(
                        CollisionPair(
                            geom1=c.read_string(child, t.A_PAIR_GEOM1) or "",
                            geom2=c.read_string(child, t.A_PAIR_GEOM2) or "",
                            friction=c.read_float_list(child, t.A_PAIR_FRICTION),
                            solref=c.read_float_list(child, t.A_PAIR_SOLREF),
                            solimp=c.read_float_list(child, t.A_PAIR_SOLIMP),
                            condim=c.read_int(child, t.A_PAIR_CONDIM),
                            margin=c.read_double(child, t.A_PAIR_MARGIN),
                            gap=c.read_double(child, t.A_PAIR_GAP),
                            extensions=c.read_extensions(child, t.CD_EXTENSIONS),
                        )
                    )

        if group_passthrough:
            # CollisionGroup has no extensions field, so per-group USD-only
            # properties are nested under the config's own extensions.
            extensions = c.merge_usd_passthrough(
                extensions, {"groups": group_passthrough}
            )
        return CollisionConfig(
            groups=groups, excludes=excludes, pairs=pairs, extensions=extensions
        )

    # -- scene / physics -----------------------------------------------

    def _read_scene(self) -> Scene | None:
        prim = self.root_prim.GetChild(t.SCOPE_RENDER)
        if not prim.IsValid():
            return None
        shadows = c.read_bool(prim, t.A_SCENE_SHADOWS)
        grid = c.read_bool(prim, t.A_SCENE_GRID)
        origin_visual = c.read_bool(prim, t.A_SCENE_ORIGIN_VISUAL)
        return Scene(
            ambient=c.read_float_list(prim, t.A_SCENE_AMBIENT),
            background=c.read_float_list(prim, t.A_SCENE_BACKGROUND),
            shadows=True if shadows is None else shadows,
            grid=False if grid is None else grid,
            origin_visual=False if origin_visual is None else origin_visual,
            extensions=c.read_extensions(prim, t.CD_EXTENSIONS),
        )

    def _read_physics(self) -> Physics | None:
        prim = self._find_physics_scene()
        if prim is None:
            return None

        gravity = c.read_vector3(prim, t.A_PHYSICS_GRAVITY)
        if gravity is None:
            gravity = self._read_native_gravity(prim)

        defaults = PhysicsSolver()
        solver = PhysicsSolver(
            type=c.read_string(prim, t.A_PHYSICS_SOLVER_TYPE) or defaults.type,
            iterations=_or_default(
                c.read_int(prim, t.A_PHYSICS_SOLVER_ITERATIONS), defaults.iterations
            ),
            sor=_or_default(c.read_double(prim, t.A_PHYSICS_SOLVER_SOR), defaults.sor),
            min_step_size=_or_default(
                c.read_double(prim, t.A_PHYSICS_SOLVER_MIN_STEP),
                defaults.min_step_size,
            ),
            max_step_size=_or_default(
                c.read_double(prim, t.A_PHYSICS_SOLVER_MAX_STEP),
                defaults.max_step_size,
            ),
            tolerance=c.read_double(prim, t.A_PHYSICS_SOLVER_TOLERANCE),
            extensions=c.read_extensions(prim, t.CD_SOLVER_EXTENSIONS),
        )
        return Physics(
            gravity=gravity,
            timestep=_or_default(
                c.read_double(prim, t.A_PHYSICS_TIMESTEP), Physics().timestep
            ),
            solver=solver,
            contact=_read_contact_surface(prim, t.PHYSICS_CONTACT_PREFIX),
            extensions=c.read_extensions(prim, t.CD_EXTENSIONS),
        )

    def _find_physics_scene(self) -> Any | None:
        """Locate the stage's ``UsdPhysicsScene`` prim, or ``None``.

        The canonical path the exporter writes is tried first, then the stage is
        searched by prim type. The search is what makes a *foreign* stage's world
        physics importable at all: Isaac Sim and Omniverse assets name the scene
        whatever they like (``/physicsScene``, ``/World/PhysicsScene``), so keying
        only off our own path silently dropped their gravity and solver settings.
        """
        pxr = self.pxr
        canonical = self.stage.GetPrimAtPath(pxr.Sdf.Path(t.PHYSICS_SCENE_PATH))
        if canonical and canonical.IsValid() and canonical.IsA(pxr.UsdPhysics.Scene):
            return canonical

        scenes = [
            prim for prim in self.stage.Traverse() if prim.IsA(pxr.UsdPhysics.Scene)
        ]
        if not scenes:
            return None
        if len(scenes) > 1:
            # Physics is world-level in CommonSchema, so only one can be kept.
            self.context.add_warning(
                f"Stage declares {len(scenes)} physics scenes "
                f"({', '.join(str(prim.GetPath()) for prim in scenes)}); "
                f"only {scenes[0].GetPath()} is parsed."
            )
        return scenes[0]

    def _read_native_gravity(self, prim: Any) -> Vector3:
        """Reconstruct gravity from ``gravityDirection`` * ``gravityMagnitude``.

        Both attributes carry sentinels — ``(0, 0, 0)`` means "use the stage up
        axis" and ``-inf`` means "earth gravity" — so an unauthored or sentinel
        value falls back to the schema default rather than to zero.
        """
        usd_scene = self.pxr.UsdPhysics.Scene(prim)
        direction = c.read_native(usd_scene.GetGravityDirectionAttr())
        magnitude = c.read_native(usd_scene.GetGravityMagnitudeAttr())

        default = Physics().gravity
        if magnitude is None or not math.isfinite(float(magnitude)):
            return default
        if direction is None or all(float(v) == 0.0 for v in direction):
            # "Down" is -Z on our stages, which are always authored Z-up.
            return Vector3(x=0.0, y=0.0, z=-float(magnitude))
        return Vector3(
            x=float(direction[0]) * float(magnitude),
            y=float(direction[1]) * float(magnitude),
            z=float(direction[2]) * float(magnitude),
        )


# --------------------------------------------------------------------------
# Module-level helpers
# --------------------------------------------------------------------------


def _is_within(candidate: Path, root: Path) -> bool:
    """Whether ``candidate`` is ``root`` or lives beneath it.

    Both are already resolved, so this compares real locations and a symlink cannot
    be used to point outside the root.
    """
    return candidate == root or root in candidate.parents


def _or_default(value: Any, default: Any) -> Any:
    return default if value is None else value


def _from_usd_angle(value: Any, is_angular: bool) -> float | None:
    """Convert a USD coordinate back to schema units, dropping infinities.

    UsdPhysics uses ``±inf`` for "unlimited", which the schema expresses as
    ``None``.
    """
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return math.radians(number) if is_angular else number


def _geometry_type_from_token(token: str) -> GeometryType:
    try:
        return GeometryType(token)
    except ValueError:
        logger.warning("Unknown %s value %r; treating as unknown.", t.A_GEOM_TYPE, token)
        return GeometryType.UNKNOWN


def _pair_index(name: str) -> int:
    """Sort key recovering the authored order of ``pair_<i>`` prims."""
    try:
        return int(name.rsplit("_", 1)[-1])
    except ValueError:
        return 1 << 30


def _read_contact_surface(prim: Any, prefix: str) -> ContactSurface:
    """Read a :class:`ContactSurface` written under ``prefix``.

    One reader for both ``Contact.surface`` and ``Physics.contact``, paired with
    the single writer in the exporter. The five non-optional fields fall back to
    the dataclass defaults so a partially-authored surface still parses.
    """
    defaults = ContactSurface()
    return ContactSurface(
        mu_static=_or_default(c.read_double(prim, f"{prefix}MuStatic"), defaults.mu_static),
        mu_dynamic=_or_default(
            c.read_double(prim, f"{prefix}MuDynamic"), defaults.mu_dynamic
        ),
        restitution=_or_default(
            c.read_double(prim, f"{prefix}Restitution"), defaults.restitution
        ),
        stiffness=_or_default(
            c.read_double(prim, f"{prefix}Stiffness"), defaults.stiffness
        ),
        damping=_or_default(c.read_double(prim, f"{prefix}Damping"), defaults.damping),
        slip_compliance=c.read_json_attr(prim, f"{prefix}SlipCompliance"),
        soft_cfm=c.read_double(prim, f"{prefix}SoftCfm"),
        soft_erp=c.read_double(prim, f"{prefix}SoftErp"),
    )


__all__ = ["USD_SUFFIXES", "USDParser"]
