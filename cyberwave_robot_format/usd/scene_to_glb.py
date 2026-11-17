# Copyright [2025] Cyberwave
# Licensed under the Apache License, Version 2.0

"""Bake a static USD scene into one self-contained GLB.

:class:`USDParser` reads USD as a *robot* (links, joints, physics) and deliberately
reports a foreign ``UsdGeomMesh`` as "a mesh with no filename" -- it never looks at
the vertices. This module is the other half: it reads USD as a *visual scene* and
produces something the platform can display, a single binary glTF with the
transforms baked into the vertices, the materials translated to glTF PBR and the
textures embedded.

What it does, in order:

1. Opens the stage (composition resolved) and refuses layers outside the input's
   directory unless ``allow_external_layers`` is set -- the same contract as
   :class:`USDParser`.
2. Walks every renderable ``Mesh`` (plus ``Cube``/``Sphere``/``Cylinder``/``Cone``/
   ``Capsule``), fan-triangulates it, bakes the full local-to-world transform,
   the stage's ``metersPerUnit`` and the up-axis conversion into the vertices.
3. Translates each bound material to glTF PBR. Four dialects are understood:
   ``UsdPreviewSurface``, glTF-style MDL (``gltf/pbr.mdl``), OmniPBR-style inputs
   and file-based MDL (the ``.mdl`` text is read for its constants and textures).
   A material that cannot be resolved -- typically an MDL library that is not on
   disk -- becomes a neutral grey and is listed in the report rather than silently
   dropped.
4. Downscales textures and embeds them.

Textures and ``.mdl`` files named by the stage are only read from inside the
input's directory (or, with ``allow_external_layers``, the directories of the
layers USD actually composed). Otherwise an uploaded stage naming
``/etc/some-image.png`` would have that file embedded in the GLB it gets back.

Requires ``pxr`` (``usd-core``) and Pillow, both in the ``usd`` extra. Importing
this module needs neither; only converting does.
"""

from __future__ import annotations

import io
import logging
import math
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from trimesh.visual import TextureVisuals
from trimesh.visual.material import PBRMaterial

from cyberwave_robot_format.collision import (
    fit_collision_boxes,
    hollow_collision_warning,
    hull_volume_ratio,
)
from cyberwave_robot_format.usd import common as c
from cyberwave_robot_format.usd.parser import USD_SUFFIXES, USDParser

logger = logging.getLogger(__name__)

_NEUTRAL_GREY = [0.6, 0.6, 0.6, 1.0]
_DEFAULT_MATERIAL_ARGS: dict[str, Any] = {
    "name": "default",
    "baseColorFactor": [0.55, 0.55, 0.58, 1.0],
    "metallicFactor": 0.6,
    "roughnessFactor": 0.45,
}

# Some DCC exports (FBX -> Omniverse) author only a black ``diffuse_color_constant``
# for metals and rely on the renderer's defaults for the rest. That renders as flat
# black in glTF, so a material of this kind gets an explicit PBR preset by name.
# Only applied to an untextured, pure-black OmniPBR-style material.
_NAME_PRESETS: dict[str, tuple[list[float], float, float]] = {
    "chrome": ([0.86, 0.86, 0.88, 1.0], 1.0, 0.15),
    "steel": ([0.75, 0.75, 0.78, 1.0], 1.0, 0.30),
    "copper": ([0.72, 0.45, 0.20, 1.0], 1.0, 0.30),
    "brass": ([0.78, 0.60, 0.25, 1.0], 1.0, 0.30),
    "plastic_black": ([0.02, 0.02, 0.02, 1.0], 0.0, 0.40),
}

# Boxes, not meshes: a hollow mesh can take up to ``BOX_FIT_MAX_BOXES`` of them. Past
# the cap, the meshes with the most boxes collide as their bounding box, so a prim is
# dropped only when the scene has more collider prims than this. The report says so
# in both cases.
_MAX_COLLIDERS = 256

# Prim purposes that are part of the visible scene (``guide`` and ``proxy`` are not).
_RENDER_PURPOSES = ("default", "render")

# Rotations taking a +Z-built primitive onto USD's ``axis = "X"`` / ``"Y"``.
_AXIS_ROTATIONS = {
    "X": trimesh.transformations.rotation_matrix(math.pi / 2, [0, 1, 0]),
    "Y": trimesh.transformations.rotation_matrix(-math.pi / 2, [1, 0, 0]),
}

_MDL_FLOAT_KEYS = (
    "reflection_roughness_constant",
    "metallic_constant",
    "opacity_constant",
)
_MDL_TEXTURE_KEYS = (
    "diffuse_texture",
    "ORM_texture",
    "normalmap_texture",
    "emissive_mask_texture",
)


@dataclass
class SceneGlbReport:
    """What the conversion did, including everything it could not do faithfully."""

    mesh_count: int = 0
    triangle_count: int = 0
    extents_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    #: material name -> how it was translated (``"gltf-mdl"``, ``"omnipbr"``, ...)
    materials: dict[str, str] = field(default_factory=dict)
    #: materials replaced by a placeholder because their source could not be used
    substituted_materials: list[str] = field(default_factory=list)
    #: texture references that could not be read (missing, or outside the roots)
    missing_textures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    #: Box colliders, in output units (m) and axes:
    #: ``{"name", "size": [x, y, z], "xyz": [x, y, z], "rpy": [r, p, y]}``.
    collision: list[dict[str, Any]] = field(default_factory=list)
    #: Mesh volume / convex-hull volume of the whole scene; low means hollow.
    hull_ratio: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "mesh_count": self.mesh_count,
            "triangle_count": self.triangle_count,
            "extents_m": list(self.extents_m),
            "materials": dict(self.materials),
            "substituted_materials": list(self.substituted_materials),
            "missing_textures": list(self.missing_textures),
            "warnings": list(self.warnings),
            "collision": [dict(box) for box in self.collision],
            "hull_ratio": self.hull_ratio,
        }


@dataclass
class SceneGlbResult:
    glb: bytes
    report: SceneGlbReport


class USDSceneToGLB:
    """Convert a static USD scene into a single GLB.

    Args:
        max_texture_size: Longest texture edge in pixels. Larger images are
            downscaled; smaller ones are left alone.
        z_up: Emit Cyberwave's Z-up convention (default). The platform does not
            re-orient user-uploaded GLBs, so a glTF-standard Y-up file would stand
            on its end. ``False`` emits glTF-standard Y-up.
        allow_external_layers: Whether composition (and texture lookup) may leave
            the input file's directory. Leave ``False`` for untrusted input.
        include_invisible: Include prims authored ``visibility = "invisible"``.
            Defaults to ``True`` because authors routinely hide geometry to get
            it out of a debug camera's way, not to remove it from the scene.
        material_overrides: ``{material name: {"base_color": [r, g, b],
            "roughness": float, "metallic": float, "alpha": float}}``. An override
            replaces the translated material entirely; use it for materials the
            report lists as substituted.
        apply_name_presets: Give untextured black ``chrome``/``steel``/``copper``/
            ``brass``/``plastic_black`` OmniPBR-style materials metal PBR values.
        max_triangles: Refuse a scene with more triangles than this (``ValueError``
            naming the limit) instead of exhausting memory. Peak memory is roughly
            0.75 GB per million triangles, so the 5M default stays near 4 GB; a
            service running several conversions at once should keep it modest.
        generate_colliders: Report box colliders for the scene (see
            :attr:`SceneGlbReport.collision`). A ``Cube`` prim or a prim with the USD
            collision API becomes one exact box; any other mesh whose longest world
            edge is at least ``min_collider_size_m`` becomes the boxes that
            :func:`~cyberwave_robot_format.collision.fit_collision_boxes` fits to
            it: one for a solid mesh, several along the surface of a hollow one, so
            walls modelled as one mesh do not close the room they stand around.
            Small props are skipped. A prim with ``physics:collisionEnabled =
            false`` is never a collider -- that is the opt-out.
        min_collider_size_m: Size threshold for the above, in metres.
    """

    def __init__(
        self,
        *,
        max_texture_size: int = 2048,
        z_up: bool = True,
        allow_external_layers: bool = False,
        include_invisible: bool = True,
        material_overrides: dict[str, dict[str, Any]] | None = None,
        apply_name_presets: bool = True,
        max_triangles: int = 5_000_000,
        generate_colliders: bool = True,
        min_collider_size_m: float = 1.0,
    ) -> None:
        if max_texture_size < 1:
            raise ValueError("max_texture_size must be positive")
        if max_triangles < 1:
            raise ValueError("max_triangles must be positive")
        self.max_triangles = max_triangles
        self.generate_colliders = generate_colliders
        self.min_collider_size_m = min_collider_size_m
        self.max_texture_size = max_texture_size
        self.z_up = z_up
        self.allow_external_layers = allow_external_layers
        self.include_invisible = include_invisible
        self.material_overrides = material_overrides or {}
        self.apply_name_presets = apply_name_presets

    # ------------------------------------------------------------------ entry

    def convert(
        self, input_path: str | Path, *, root_dir: str | Path | None = None
    ) -> SceneGlbResult:
        """Convert the USD file at ``input_path``.

        ``root_dir`` bounds composition and texture lookup (see
        ``allow_external_layers``). It defaults to the file's directory; pass the
        archive root when converting a scene extracted from a ZIP, where textures
        often sit in a sibling folder of the scene file.
        """
        input_path = Path(input_path)
        if input_path.suffix.lower() not in USD_SUFFIXES:
            raise ValueError(
                f"{input_path} is not a USD file (expected one of "
                f"{sorted(USD_SUFFIXES)})."
            )
        root = Path(root_dir) if root_dir is not None else input_path.parent
        stage = USDParser(
            allow_external_layers=self.allow_external_layers
        ).open_confined(input_path, root)
        return self.convert_stage(stage, root_dir=root)

    def convert_stage(self, stage: Any, *, root_dir: str | Path) -> SceneGlbResult:
        """Convert an already-open stage; ``root_dir`` anchors texture lookup."""
        return _Conversion(self, stage, Path(root_dir).resolve()).run()


def convert_usd_scene_to_glb(input_path: str | Path, **options: Any) -> SceneGlbResult:
    """One-call form of :class:`USDSceneToGLB`; ``options`` are its keyword args."""
    return USDSceneToGLB(**options).convert(input_path)


# ----------------------------------------------------------------------------
# Implementation
# ----------------------------------------------------------------------------


@dataclass
class _Collider:
    """One prim's boxes, held until the whole scene is known."""

    path: str
    name: str
    boxes: list[dict[str, Any]]
    bounding_box: dict[str, Any]
    notes: list[str] = field(default_factory=list)


class _Conversion:
    """One conversion run: owns the caches that must not outlive it."""

    def __init__(self, cfg: USDSceneToGLB, stage: Any, root_dir: Path) -> None:
        self.cfg = cfg
        self.stage = stage
        self.pxr = c.require_pxr()
        self.report = SceneGlbReport()
        self.scene = trimesh.Scene()
        self._colliders: list[_Collider] = []
        self._materials: dict[str, PBRMaterial] = {}
        # path -> decoded image, or None for a file that could not be read
        self._images: dict[str, Any] = {}
        self._used_names: set[str] = set()
        self._triangles = 0  # running total across meshes, checked against the budget

        self.root_dir = root_dir
        self.roots = {root_dir}
        if cfg.allow_external_layers:
            for layer in stage.GetUsedLayers():
                real = getattr(layer, "realPath", "") or ""
                if real:
                    self.roots.add(Path(real).resolve().parent)
        self._by_basename: dict[str, list[Path]] | None = None

        self.world = self._stage_matrix()

    # --------------------------------------------------------------- run loop

    def run(self) -> SceneGlbResult:
        pxr = self.pxr
        predicate = pxr.Usd.TraverseInstanceProxies(pxr.Usd.PrimDefaultPredicate)
        xcache = pxr.UsdGeom.XformCache(pxr.Usd.TimeCode.Default())
        for prim in pxr.Usd.PrimRange(self.stage.GetPseudoRoot(), predicate):
            if not self._is_renderable(prim):
                continue
            built = self._build_geometry(prim, xcache)
            if built is None:
                continue
            verts, tris, normals, uv = built
            name = self._add(prim, verts, tris, normals, uv)
            if self.cfg.generate_colliders:
                self._collect_collider(prim, name, verts, tris, xcache)

        if not self.scene.geometry:
            raise ValueError("The USD scene contains no renderable meshes.")

        bounds = self.scene.bounds
        size = bounds[1] - bounds[0]
        self.report.extents_m = (float(size[0]), float(size[1]), float(size[2]))
        self.report.mesh_count = len(self.scene.geometry)
        self.report.triangle_count = sum(
            len(g.faces) for g in self.scene.geometry.values()
        )
        self._place_colliders()
        self._assess_collision()
        glb = self.scene.export(file_type="glb")
        return SceneGlbResult(glb=bytes(glb), report=self.report)

    def _is_renderable(self, prim: Any) -> bool:
        UsdGeom = self.pxr.UsdGeom
        if not prim.IsA(UsdGeom.Gprim):
            return False
        img = UsdGeom.Imageable(prim)
        if img.ComputePurpose() not in _RENDER_PURPOSES:
            return False
        if not self.cfg.include_invisible and (
            img.ComputeVisibility() == UsdGeom.Tokens.invisible
        ):
            return False
        return True

    # -------------------------------------------------------------- transform

    def _stage_matrix(self) -> np.ndarray:
        """Units and up-axis conversion applied on top of every world transform."""
        UsdGeom = self.pxr.UsdGeom
        scale = float(UsdGeom.GetStageMetersPerUnit(self.stage) or 1.0)
        up = UsdGeom.GetStageUpAxis(self.stage)
        m = np.eye(4)
        m[:3, :3] *= scale
        # Y-up stage -> Z-up output (+90deg about X), Z-up stage -> Y-up output
        # (-90deg about X); the matching combinations are the identity.
        if up == UsdGeom.Tokens.y and self.cfg.z_up:
            angle = 90.0
        elif up == UsdGeom.Tokens.z and not self.cfg.z_up:
            angle = -90.0
        else:
            angle = 0.0
        rot = trimesh.transformations.rotation_matrix(math.radians(angle), [1, 0, 0])
        return rot @ m

    def _prim_matrix(self, prim: Any, xcache: Any) -> np.ndarray:
        return self.world @ c.gf_matrix_to_np(xcache.GetLocalToWorldTransform(prim))

    # --------------------------------------------------------------- geometry

    def _build_geometry(self, prim: Any, xcache: Any) -> tuple | None:
        UsdGeom = self.pxr.UsdGeom
        matrix = self._prim_matrix(prim, xcache)
        if prim.IsA(UsdGeom.Mesh):
            return self._mesh_geometry(prim, matrix)
        primitive = self._primitive_mesh(prim)
        if primitive is None:
            self.report.warnings.append(
                f"Skipped {prim.GetPath()}: unsupported {prim.GetTypeName()} prim."
            )
            return None
        # One primitive is small, but a scene can hold a million of them (instanced
        # spheres standing in for points), so they count against the budget too.
        self._charge_triangles(len(primitive.faces))
        verts = np.asarray(primitive.vertices, dtype=np.float64)
        tris = np.asarray(primitive.faces, dtype=np.int64)
        normals = np.asarray(primitive.vertex_normals, dtype=np.float64)
        return self._transform(verts, tris, normals, None, matrix, None)

    def _charge_triangles(self, count: int) -> None:
        """Count ``count`` more triangles; refuse the scene once the budget is spent."""
        self._triangles += count
        if self._triangles > self.cfg.max_triangles:
            raise ValueError(
                f"The scene has more than {self.cfg.max_triangles:,} triangles; "
                "simplify or split it and try again."
            )

    def _mesh_geometry(self, prim: Any, matrix: np.ndarray) -> tuple | None:
        UsdGeom = self.pxr.UsdGeom
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        if not points or not counts or indices is None:
            return None
        pts = np.asarray(points, dtype=np.float64)
        counts = np.asarray(counts, dtype=np.int64)
        indices = np.asarray(indices, dtype=np.int64)
        if (counts < 3).any():
            self.report.warnings.append(
                f"{prim.GetPath()}: dropped faces with fewer than 3 vertices."
            )
        # Counted from the face sizes, before any per-corner array is built: that is
        # where the memory goes, and a scene over budget must fail before it does.
        self._charge_triangles(int(np.maximum(counts - 2, 0).sum()))
        corner_tri = _fan_triangulate(counts)  # (T, 3) indices into the corner list
        if len(corner_tri) == 0:
            return None
        vert_tri = indices[corner_tri]  # (T, 3) indices into points

        # Un-welded per-corner data: every attribute is looked up per triangle corner
        # so face-varying normals and UVs survive; welding happens afterwards.
        flat_c = corner_tri.reshape(-1)
        flat_v = vert_tri.reshape(-1)
        face_of_corner = np.repeat(np.arange(len(counts)), counts)
        verts = pts[flat_v]

        normals = self._read_normals(mesh, flat_c, flat_v, face_of_corner)
        uv = self._read_uv(prim, flat_c, flat_v, face_of_corner)
        tris = np.arange(len(verts), dtype=np.int64).reshape(-1, 3)
        left_handed = mesh.GetOrientationAttr().Get() == UsdGeom.Tokens.leftHanded
        return self._transform(verts, tris, normals, uv, matrix, left_handed)

    def _read_normals(
        self, mesh: Any, flat_c: Any, flat_v: Any, face_of_corner: Any
    ) -> np.ndarray | None:
        values = mesh.GetNormalsAttr().Get()
        if values is None or len(values) == 0:
            return None
        return self._per_corner(
            values, mesh.GetNormalsInterpolation(), flat_c, flat_v, face_of_corner
        )

    def _read_uv(
        self, prim: Any, flat_c: Any, flat_v: Any, face_of_corner: Any
    ) -> np.ndarray | None:
        primvar = self.pxr.UsdGeom.PrimvarsAPI(prim).GetPrimvar("st")
        if not primvar or not primvar.HasValue():
            return None
        return self._per_corner(
            primvar.ComputeFlattened(),
            primvar.GetInterpolation(),
            flat_c,
            flat_v,
            face_of_corner,
        )

    def _per_corner(
        self,
        values: Any,
        interpolation: Any,
        flat_c: Any,
        flat_v: Any,
        face_of_corner: Any,
    ) -> np.ndarray | None:
        """Look an attribute up once per triangle corner, whatever its interpolation."""
        tokens = self.pxr.UsdGeom.Tokens
        arr = np.asarray(values, dtype=np.float64)
        if interpolation == tokens.faceVarying:
            return arr[flat_c]
        if interpolation in (tokens.vertex, tokens.varying):
            return arr[flat_v]
        if interpolation == tokens.uniform:
            return arr[face_of_corner[flat_c]]
        if interpolation == tokens.constant:
            return np.repeat(arr[:1], len(flat_c), axis=0)
        return None

    @staticmethod
    def _transform(
        verts: np.ndarray,
        tris: np.ndarray,
        normals: np.ndarray | None,
        uv: np.ndarray | None,
        matrix: np.ndarray,
        left_handed: bool | None,
    ) -> tuple:
        """Bake ``matrix`` into positions and normals and settle the winding.

        With authored normals the winding is decided by agreement with them -- the
        orientation token and the transform's handedness both change which way a
        triangle faces, and the normals are the one thing that is unambiguous. With
        none, the geometric normal is derived and the orientation token plus the
        determinant decide.
        """
        pos = (np.hstack([verts, np.ones((len(verts), 1))]) @ matrix.T)[:, :3]
        mirrored = np.linalg.det(matrix[:3, :3]) < 0

        if normals is not None:
            nmat = np.linalg.inv(matrix[:3, :3]).T
            nrm = normals @ nmat.T
            nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
            t = pos[tris]
            geom = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
            agree = np.einsum("ij,ij->i", geom, nrm[tris].mean(axis=1))
            if (agree < 0).mean() > 0.5:
                tris = tris[:, ::-1]
        else:
            flip = bool(left_handed) != bool(mirrored)
            if flip:
                tris = tris[:, ::-1]
            t = pos[tris]
            geom = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
            geom /= np.maximum(np.linalg.norm(geom, axis=1, keepdims=True), 1e-12)
            # Corner ``i`` belongs to triangle ``i // 3`` whichever way its row runs.
            nrm = np.repeat(geom, 3, axis=0)
        return pos, tris, nrm, uv

    def _primitive_mesh(self, prim: Any) -> trimesh.Trimesh | None:
        UsdGeom = self.pxr.UsdGeom
        if prim.IsA(UsdGeom.Cube):
            size = float(UsdGeom.Cube(prim).GetSizeAttr().Get())
            return trimesh.creation.box(extents=[size] * 3)
        if prim.IsA(UsdGeom.Sphere):
            r = float(UsdGeom.Sphere(prim).GetRadiusAttr().Get())
            return trimesh.creation.uv_sphere(radius=r)
        # All built along +Z, which is also USD's default axis; a cone is re-centred
        # because trimesh puts its base on z = 0.
        for kind, make in (
            (UsdGeom.Cylinder, trimesh.creation.cylinder),
            (UsdGeom.Capsule, lambda r, h: trimesh.creation.capsule(h, r)),
            (
                UsdGeom.Cone,
                lambda r, h: trimesh.creation.cone(r, h).apply_translation(
                    [0, 0, -h / 2.0]
                ),
            ),
        ):
            if not prim.IsA(kind):
                continue
            api = kind(prim)
            shape = make(
                float(api.GetRadiusAttr().Get()), float(api.GetHeightAttr().Get())
            )
            rotation = _AXIS_ROTATIONS.get(str(api.GetAxisAttr().Get()))
            if rotation is not None:
                shape.apply_transform(rotation)
            return shape
        return None

    # -------------------------------------------------------------- colliders

    def _collect_collider(
        self, prim: Any, name: str, verts: np.ndarray, tris: np.ndarray, xcache: Any
    ) -> None:
        """Hold ``prim``'s boxes if it is a collider (see the class doc)."""
        UsdGeom = self.pxr.UsdGeom
        enabled = prim.GetAttribute("physics:collisionEnabled")
        if enabled and enabled.Get() is False:
            return
        explicit = "PhysicsCollisionAPI" in prim.GetAppliedSchemas()
        path = str(prim.GetPath())
        if prim.IsA(UsdGeom.Cube):
            box = self._cube_box(prim, xcache)
            if not explicit and max(box["size"]) < self.cfg.min_collider_size_m:
                return
            self._colliders.append(_Collider(path, name, [box], box))
            return
        # ``verts`` already carry the full world transform, so the boxes are fitted
        # in the frame of the GLB. Not the mesh's bounding box: around walls that
        # share a mesh, or a wall that stands at an angle, that box also covers the
        # floor between them.
        lo, hi = verts.min(axis=0), verts.max(axis=0)
        if not explicit and (hi - lo).max() < self.cfg.min_collider_size_m:
            return
        bounding_box = {
            "size": (hi - lo).tolist(),
            "xyz": ((hi + lo) / 2).tolist(),
            "rpy": [0.0, 0.0, 0.0],
        }
        # The fitter takes axis 2 as up, so a Y-up output keeps the bounding box.
        notes: list[str] = []
        boxes = (
            fit_collision_boxes(verts, tris, notes=notes)
            if self.cfg.z_up
            else [bounding_box]
        )
        self._colliders.append(_Collider(path, name, boxes, bounding_box, notes))

    def _place_colliders(self) -> None:
        """Write the held boxes to the report, within ``_MAX_COLLIDERS``.

        Over the cap, the meshes with the most boxes (the later one on a tie) collide
        as their bounding box until the rest fit. It runs once the whole scene is
        known, so early meshes cannot use up the cap before a late one, often the
        outer walls, gets its box.
        """
        total = sum(len(c.boxes) for c in self._colliders)
        for collider in sorted(
            reversed(self._colliders), key=lambda c: len(c.boxes), reverse=True
        ):
            if total <= _MAX_COLLIDERS or len(collider.boxes) == 1:
                break
            total -= len(collider.boxes) - 1
            collider.boxes = [collider.bounding_box]
            collider.notes = [
                f"the scene needs more than {_MAX_COLLIDERS} collision boxes, so it "
                "collides as its bounding box"
            ]
        for collider in self._colliders:
            # Without them a fit that fell back looks like one that worked.
            for note in collider.notes:
                self._warn(f"{collider.path}: {note}.")
            for index, box in enumerate(collider.boxes):
                if len(self.report.collision) >= _MAX_COLLIDERS:
                    self._warn(
                        f"More than {_MAX_COLLIDERS} colliders; the rest were left "
                        "out of the collision."
                    )
                    return
                # ``-`` is not valid in a USD identifier, so no prim name can clash.
                name = f"{collider.name}-{index}" if index else collider.name
                self.report.collision.append({"name": name, **_rounded(box)})

    def _cube_box(self, prim: Any, xcache: Any) -> dict[str, Any]:
        """A ``Cube`` as a box with its exact size and pose.

        Goes through the same matrix as the geometry (the stage's units and up-axis on
        top of the prim's world transform), so it lines up with the GLB.
        """
        matrix = self._prim_matrix(prim, xcache)
        edge = float(self.pxr.UsdGeom.Cube(prim).GetSizeAttr().Get())
        linear = matrix[:3, :3]
        scale = np.linalg.norm(linear, axis=0)
        rot = linear / np.maximum(scale, 1e-12)
        if np.linalg.det(rot) < 0:  # a mirrored cube is the same box
            rot[:, 2] *= -1
        size = edge * scale
        if np.allclose(np.abs(rot), np.round(np.abs(rot)), atol=1e-6):
            # A quarter-turn (the Y-up -> Z-up conversion): swap the sizes instead of
            # carrying a rotation, so the box is axis-aligned like the walls it is.
            size = np.abs(rot) @ size
            rot = np.eye(3)
        euler = trimesh.transformations.euler_from_matrix(rot, axes="sxyz")
        return {
            "size": size.tolist(),
            "xyz": matrix[:3, 3].tolist(),
            "rpy": [float(a) for a in euler],
        }

    def _assess_collision(self) -> None:
        """Warn when the scene would collide as a solid block (no boxes, hollow hull)."""
        if self.report.collision:
            return
        vertices, offset, faces = [], 0, []
        for mesh in self.scene.geometry.values():
            vertices.append(np.asarray(mesh.vertices))
            faces.append(np.asarray(mesh.faces) + offset)
            offset += len(mesh.vertices)
        ratio = hull_volume_ratio(np.vstack(vertices), np.vstack(faces))
        self.report.hull_ratio = ratio
        warning = hollow_collision_warning(ratio)
        if warning:
            self._warn(warning)

    # ------------------------------------------------------------------ nodes

    def _add(
        self,
        prim: Any,
        verts: np.ndarray,
        tris: np.ndarray,
        normals: np.ndarray,
        uv: np.ndarray | None,
    ) -> str:
        """Add one mesh node; returns the unique name it was given."""
        # Weld identical (position, normal, uv) rows: face-varying data otherwise
        # triples the vertex count for no visual gain.
        columns = [verts, normals] + ([uv] if uv is not None else [])
        key = np.round(np.hstack(columns), 5)
        _, first, inverse = np.unique(
            key, axis=0, return_index=True, return_inverse=True
        )
        inverse = inverse.reshape(-1)
        faces = inverse[tris]
        mesh = trimesh.Trimesh(
            vertices=verts[first],
            faces=faces,
            vertex_normals=normals[first],
            process=False,
        )
        material = self._bound_material(prim)
        if uv is not None:
            mesh.visual = TextureVisuals(uv=uv[first], material=material)
        else:
            mesh.visual = TextureVisuals(material=material)
        name = self._unique_name(prim)
        self.scene.add_geometry(mesh, node_name=name, geom_name=name)
        return name

    def _unique_name(self, prim: Any) -> str:
        base = str(prim.GetPath()).strip("/").replace("/", ".") or prim.GetName()
        name, n = base, 1
        while name in self._used_names:
            n += 1
            name = f"{base}_{n}"
        self._used_names.add(name)
        return name

    # -------------------------------------------------------------- materials

    def _bound_material(self, prim: Any) -> PBRMaterial:
        UsdShade = self.pxr.UsdShade
        bound = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
        if not bound:
            return PBRMaterial(**_DEFAULT_MATERIAL_ARGS)
        key = str(bound.GetPath())
        if key not in self._materials:
            self._materials[key] = self._translate_material(bound.GetPrim())
        return self._materials[key]

    def _translate_material(self, mat_prim: Any) -> PBRMaterial:
        name = mat_prim.GetName()
        override = self.cfg.material_overrides.get(name)
        if override is not None:
            self.report.materials[name] = "override"
            return _override_material(name, override)

        shader = self._surface_shader(mat_prim)
        if shader is None:
            self.report.materials[name] = "default (no surface shader)"
            return PBRMaterial(**{**_DEFAULT_MATERIAL_ARGS, "name": name})

        shader_id = shader.GetIdAttr().Get()
        dialect: str
        args: dict[str, Any] | None
        try:
            if shader_id == "UsdPreviewSurface":
                dialect, args = "UsdPreviewSurface", self._preview_surface(shader)
            elif shader.GetInput("base_color_texture") or shader.GetInput(
                "base_color_factor"
            ):
                dialect, args = "gltf-mdl", self._gltf_mdl(shader)
            else:
                dialect, args = self._mdl_or_omni(name, shader)
        except Exception as exc:  # a bad material must not sink the whole scene
            logger.warning("Material %s could not be translated: %s", name, exc)
            self._warn(f"Material {name}: {exc}")
            dialect, args = "failed", None

        if args is None:
            self.report.substituted_materials.append(name)
            self.report.materials[name] = f"substituted ({dialect})"
            return PBRMaterial(
                name=name,
                baseColorFactor=list(_NEUTRAL_GREY),
                metallicFactor=0.0,
                roughnessFactor=0.8,
            )
        self.report.materials[name] = dialect
        return PBRMaterial(name=name, **args)

    def _mdl_or_omni(self, name: str, shader: Any) -> tuple[str, dict | None]:
        """Translate a shader backed by an ``.mdl`` file or by OmniPBR-style inputs."""
        source = self._source_asset(shader)
        if source is not None and self._is_local_mdl(source):
            return self._mdl_file(name, source)
        if shader.GetInput("diffuse_color_constant") or shader.GetInput(
            "diffuse_texture"
        ):
            # Omniverse's built-in OmniPBR*.mdl is referenced by name and is not a
            # file on disk; the shader's own inputs carry everything we need.
            return "omnipbr", self._omni_values(
                _usd_inputs(shader), self.root_dir, name
            )
        if source is not None and source.suffix.lower() == ".mdl":
            self._warn(
                f"Material {name}: MDL source {source} is not available "
                "(missing, or outside the input directory)."
            )
            return "mdl (unavailable)", None
        return "unrecognized", None

    def _surface_shader(self, mat_prim: Any) -> Any | None:
        UsdShade = self.pxr.UsdShade
        material = UsdShade.Material(mat_prim)
        # Prefer the render-context outputs (``mdl``), then the universal one.
        for context in ("mdl", ""):
            out = material.GetSurfaceOutput(context)
            if out and out.HasConnectedSource():
                src = out.GetConnectedSources()[0][0].source
                return UsdShade.Shader(src.GetPrim())
        for child in self.pxr.Usd.PrimRange(mat_prim):
            if child.IsA(UsdShade.Shader):
                return UsdShade.Shader(child)
        return None

    def _source_asset(self, shader: Any) -> Path | None:
        attr = shader.GetPrim().GetAttribute("info:mdl:sourceAsset")
        value = attr.Get() if attr else None
        if not value or not value.path:
            return None
        # An MDL source is an explicit path (or a built-in like ``OmniPBR.mdl``);
        # matching it by file name anywhere in the tree would be wrong.
        found = self._resolve_path(value, self.root_dir, by_name=False)
        return found if found is not None else Path(value.path)

    def _is_local_mdl(self, source: Path) -> bool:
        """An ``.mdl`` that exists inside the permitted roots (so it can be read)."""
        if source.suffix.lower() != ".mdl":
            return False
        resolved = self._within_roots(source)
        return resolved is not None and resolved.is_file()

    # ------------------------------------------------------ shader plumbing

    def _connected_texture(
        self, shader: Any, slot: str, file_input: str, texture_id: str | None = None
    ) -> tuple[Any, str] | None:
        """``(image, output name)`` for the texture node feeding ``shader``'s ``slot``."""
        slot_input = shader.GetInput(slot)
        if not slot_input or not slot_input.HasConnectedSource():
            return None
        link = slot_input.GetConnectedSources()[0][0]
        texture = self.pxr.UsdShade.Shader(link.source.GetPrim())
        if texture_id is not None and texture.GetIdAttr().Get() != texture_id:
            return None
        file_attr = texture.GetInput(file_input)
        asset = file_attr.Get() if file_attr else None
        if not asset or not asset.path:
            return None
        image = self._load_image(asset, self.root_dir)
        return (image, str(link.sourceName)) if image is not None else None

    # ---------------------------------------------------- dialect: preview

    def _preview_surface(self, shader: Any) -> dict[str, Any]:
        def tex(slot: str) -> tuple[Any, str] | None:
            return self._connected_texture(shader, slot, "file", "UsdUVTexture")

        args: dict[str, Any] = {"doubleSided": False}
        base = tex("diffuseColor")
        rgb = [float(x) for x in _input(shader, "diffuseColor", (0.8, 0.8, 0.8))]
        opacity = float(_input(shader, "opacity", 1.0))
        args["baseColorFactor"] = [1.0, 1.0, 1.0, opacity] if base else rgb + [opacity]
        if base:
            args["baseColorTexture"] = base[0]
            _mask_if_rgba(args, base[0])
        if opacity < 0.99:
            args["alphaMode"] = "BLEND"

        rough_const = float(_input(shader, "roughness", 0.5))
        metal_const = float(_input(shader, "metallic", 0.0))
        rough, metal = tex("roughness"), tex("metallic")
        if rough or metal:
            args["metallicRoughnessTexture"] = _pack_metal_rough(
                rough, metal, rough_const, metal_const
            )
            args["roughnessFactor"] = args["metallicFactor"] = 1.0
        else:
            args["roughnessFactor"] = rough_const
            args["metallicFactor"] = metal_const

        normal = tex("normal")
        if normal:
            args["normalTexture"] = normal[0]
        occ = tex("occlusion")
        if occ:
            args["occlusionTexture"] = occ[0]
        em = tex("emissiveColor")
        if em:
            args["emissiveTexture"], args["emissiveFactor"] = em[0], [1.0] * 3
        else:
            e = [float(x) for x in _input(shader, "emissiveColor", (0.0, 0.0, 0.0))]
            if any(e):
                args["emissiveFactor"] = e
        return args

    # ------------------------------------------------------ dialect: gltf MDL

    def _gltf_mdl(self, shader: Any) -> dict[str, Any]:
        def tex(slot: str) -> Any:
            found = self._connected_texture(shader, slot, "texture")
            return found[0] if found else None

        def num(name: str, default: float) -> float:
            return float(_input(shader, name, default))

        factor = [float(x) for x in _input(shader, "base_color_factor", (1.0,) * 3)]
        base, mr = tex("base_color_texture"), tex("metallic_roughness_texture")
        args: dict[str, Any] = {
            "baseColorFactor": factor[:3] + [num("base_alpha", 1.0)],
            "baseColorTexture": base,
            "metallicRoughnessTexture": mr,
            "normalTexture": tex("normal_texture"),
            "occlusionTexture": tex("occlusion_texture"),
            "metallicFactor": 1.0 if mr else num("metallic_factor", 0.0),
            "roughnessFactor": 1.0 if mr else num("roughness_factor", 0.8),
        }
        emissive = tex("emissive_texture")
        if emissive is not None:
            args["emissiveTexture"], args["emissiveFactor"] = emissive, [1.0] * 3
        _mask_if_rgba(args, base)
        return {k: v for k, v in args.items() if v is not None}

    # --------------------------------------------- dialect: OmniPBR / MDL text

    def _mdl_file(self, name: str, source: Path) -> tuple[str, dict[str, Any]]:
        """Translate a material backed by an ``.mdl`` that ``_is_local_mdl`` vetted."""
        resolved = source.resolve()
        values = _parse_mdl(resolved.read_text(errors="replace"))
        return "mdl-file", self._omni_values(values, resolved.parent, name)

    def _omni_values(
        self, values: dict[str, Any], anchor: Path, material_name: str
    ) -> dict[str, Any]:
        """Map OmniPBR-style inputs (from USD attributes or MDL text) to PBR args."""

        def load(key: str) -> Any:
            asset = values.get(key)
            if asset is None or not getattr(asset, "path", asset):
                return None
            return self._load_image(asset, anchor)

        color = values.get("diffuse_color_constant")
        constant = [float(x) for x in color][:3] if color is not None else [0.8] * 3
        base = load("diffuse_texture")
        args: dict[str, Any] = {
            "baseColorFactor": [1.0, 1.0, 1.0, 1.0] if base else constant + [1.0],
            "baseColorTexture": base,
            "metallicFactor": float(values.get("metallic_constant", 0.0)),
            "roughnessFactor": float(values.get("reflection_roughness_constant", 0.5)),
            "normalTexture": load("normalmap_texture"),
        }
        orm = load("ORM_texture") if values.get("enable_ORM_texture") else None
        if orm is not None:
            # ORM is occlusion/roughness/metallic in R/G/B -- glTF's own packing.
            args["metallicRoughnessTexture"] = orm
            args["occlusionTexture"] = orm
            args["metallicFactor"] = args["roughnessFactor"] = 1.0
        if values.get("enable_opacity") and "opacity_constant" in values:
            opacity = float(values["opacity_constant"])
            if opacity < 0.99:
                # Fully transparent glass would vanish; keep a faint trace.
                args["baseColorFactor"][3] = max(opacity, 0.12)
                args["alphaMode"] = "BLEND"
        if self.cfg.apply_name_presets and base is None and not any(constant):
            for prefix, (rgba, metallic, rough) in _NAME_PRESETS.items():
                if material_name.lower().startswith(prefix):
                    args.update(
                        baseColorFactor=list(rgba),
                        metallicFactor=metallic,
                        roughnessFactor=rough,
                    )
        return {k: v for k, v in args.items() if v is not None}

    # --------------------------------------------------------------- textures

    def _warn(self, message: str) -> None:
        if message not in self.report.warnings:
            self.report.warnings.append(message)

    def _within_roots(self, path: Path) -> Path | None:
        try:
            resolved = path.resolve()
        except OSError:
            return None
        return resolved if any(resolved.is_relative_to(r) for r in self.roots) else None

    def _basename_index(self) -> dict[str, list[Path]]:
        """File name -> every file of that name under the roots (built on first use)."""
        if self._by_basename is None:
            index: dict[str, list[Path]] = {}
            # A root nested inside another would be walked twice.
            roots = [
                r
                for r in self.roots
                if not any(o != r and r.is_relative_to(o) for o in self.roots)
            ]
            for root in sorted(roots):
                for dirpath, dirnames, filenames in os.walk(root):
                    dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
                    for filename in sorted(filenames):
                        if not filename.startswith("."):
                            index.setdefault(filename, []).append(
                                Path(dirpath) / filename
                            )
            self._by_basename = index
        return self._by_basename

    def _resolve_path(
        self, asset: Any, anchor: Path, *, by_name: bool = True
    ) -> Path | None:
        """Find the file an asset path names, only ever inside the permitted roots.

        Tried against the path USD resolved, then ``anchor`` and the input directory.
        Textures are often written relative to a layer other than the root one, so
        with ``by_name`` a miss falls back to a file of the same name -- and says so
        in the report, because with several same-named files it can be the wrong one.
        """
        raw = getattr(asset, "resolvedPath", "") or getattr(asset, "path", str(asset))
        raw_path = Path(raw)
        for candidate in (anchor / raw_path, self.root_dir / raw_path):
            if candidate.is_file():
                inside = self._within_roots(candidate)
                if inside is not None:
                    return inside
        if not by_name:
            return None
        matches = [
            m
            for m in self._basename_index().get(raw_path.name, [])
            if self._within_roots(m) is not None
        ]
        if not matches:
            return None
        found = matches[0].resolve()
        self._warn(
            f"Texture {raw_path} was not found where it is referenced; used {found} "
            f"(matched by file name"
            + (f", first of {len(matches)} candidates)" if len(matches) > 1 else ")")
        )
        return found

    def _image_source(
        self, asset: Any, anchor: Path
    ) -> tuple[str, Callable[[], Any]] | None:
        """``(cache key, opener)`` for the image ``asset`` names, or ``None``.

        An asset inside a ``.usdz`` package resolves to ``/a/scene.usdz[tex/x.png]``
        and is not a file on disk, so it is read back through USD's resolver; the
        package itself must still sit inside the permitted roots.
        """
        resolved = getattr(asset, "resolvedPath", "") or ""
        if "[" in resolved:
            package = Path(resolved[: resolved.index("[")])
            if self._within_roots(package) is None:
                return None
            return resolved, lambda: io.BytesIO(_read_packaged_asset(resolved))
        path = self._resolve_path(asset, anchor)
        return None if path is None else (str(path), lambda: path)

    def _load_image(self, asset: Any, anchor: Path) -> Any | None:
        """Decode, downscale and cache the image ``asset`` names (``None`` if unusable)."""
        Image = _require_pil()
        label = getattr(asset, "path", str(asset))
        source = self._image_source(asset, anchor)
        if source is None:
            self.report.missing_textures.append(label)
            return None
        key, open_source = source
        if key in self._images:
            return self._images[key]
        try:
            image = Image.open(open_source())
            image.load()
        except Exception as exc:
            self.report.missing_textures.append(label)
            self._warn(f"Texture {label} could not be read: {exc}")
            self._images[key] = None
            return None
        longest = max(image.size)
        if longest > self.cfg.max_texture_size:
            scale = self.cfg.max_texture_size / longest
            image = image.resize(
                (
                    max(1, round(image.width * scale)),
                    max(1, round(image.height * scale)),
                ),
                Image.Resampling.LANCZOS,
                reducing_gap=2.0,
            )
        has_alpha = False
        if image.mode in ("RGBA", "LA", "PA"):
            rgba = image if image.mode != "PA" else image.convert("RGBA")
            has_alpha = rgba.getchannel("A").getextrema()[0] < 250
        image = image.convert("RGBA" if has_alpha else "RGB")
        self._images[key] = image
        return image


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _read_packaged_asset(resolved: str) -> bytes:
    """Bytes of an asset stored inside a USD package (``.usdz``)."""
    from pxr import Ar

    ar_asset = Ar.GetResolver().OpenAsset(Ar.ResolvedPath(resolved))
    if not ar_asset:
        raise FileNotFoundError(resolved)
    return bytes(ar_asset.GetBuffer())


def _require_pil() -> Any:
    """The ``PIL.Image`` module, or an actionable error if the extra is missing."""
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - extra not installed
        raise ImportError(
            "Converting USD scenes needs Pillow. Install it with "
            "`pip install cyberwave-robot-format[usd]`."
        ) from exc
    return Image


def _input(shader: Any, name: str, default: Any) -> Any:
    """A shader input's authored value, or ``default`` if it is absent/unauthored."""
    attr = shader.GetInput(name)
    value = attr.Get() if attr else None
    return default if value is None else value


def _usd_inputs(shader: Any) -> dict[str, Any]:
    """Every authored input of ``shader`` by name."""
    return {
        i.GetBaseName(): v for i in shader.GetInputs() if (v := i.Get()) is not None
    }


def _mask_if_rgba(args: dict[str, Any], base_color_image: Any) -> None:
    """Alpha-test a base-colour texture that actually carries an alpha channel."""
    if base_color_image is not None and base_color_image.mode == "RGBA":
        args["alphaMode"], args["alphaCutoff"] = "MASK", 0.5


def _rounded(box: dict[str, Any]) -> dict[str, list[float]]:
    """Box values at 0.1 mm / 1e-6 rad: the report is JSON, not a mesh."""
    return {
        k: [round(float(x), 6 if k == "rpy" else 4) + 0.0 for x in v]
        for k, v in box.items()
    }


def _fan_triangulate(counts: np.ndarray) -> np.ndarray:
    """Fan-triangulate polygons; returns ``(T, 3)`` indices into the corner list.

    Vectorised: a polygon with ``n`` corners yields ``n - 2`` triangles
    ``(0, k, k + 1)``. Faces with fewer than three corners yield none.
    """
    counts = np.asarray(counts, dtype=np.int64)
    offsets = np.cumsum(counts) - counts
    per_face = np.maximum(counts - 2, 0)
    total = int(per_face.sum())
    if total == 0:
        return np.empty((0, 3), dtype=np.int64)
    face = np.repeat(np.arange(len(counts)), per_face)
    first = np.cumsum(per_face) - per_face
    k = np.arange(total) - np.repeat(first, per_face) + 1
    base = offsets[face]
    return np.stack([base, base + k, base + k + 1], axis=1)


def _override_material(name: str, spec: dict[str, Any]) -> PBRMaterial:
    rgb = [float(x) for x in spec.get("base_color", _NEUTRAL_GREY[:3])][:3]
    alpha = float(spec.get("alpha", 1.0))
    kwargs: dict[str, Any] = {
        "name": name,
        "baseColorFactor": rgb + [alpha],
        "metallicFactor": float(spec.get("metallic", 0.0)),
        "roughnessFactor": float(spec.get("roughness", 0.8)),
    }
    if alpha < 0.99:
        kwargs["alphaMode"] = "BLEND"
    return PBRMaterial(**kwargs)


def _pack_metal_rough(
    rough: tuple[Any, str] | None,
    metal: tuple[Any, str] | None,
    rough_const: float,
    metal_const: float,
) -> Any:
    """Pack separate roughness/metallic channels into glTF's G/B layout."""
    Image = _require_pil()

    sources = [s for s in (rough, metal) if s is not None]
    size = max((s[0].size for s in sources), key=lambda sz: sz[0] * sz[1])

    def channel(source: tuple[Any, str] | None, const: float) -> Any:
        if source is None:
            return Image.new("L", size, int(round(max(0.0, min(1.0, const)) * 255)))
        image, out_name = source
        band_name = out_name.upper() if out_name in ("r", "g", "b", "a") else "R"
        if band_name not in image.getbands():
            image = image.convert("RGBA")  # e.g. an RGB image asked for alpha
        band = image.getchannel(band_name)
        return band if band.size == size else band.resize(size)

    r = Image.new("L", size, 255)
    return Image.merge(
        "RGB", (r, channel(rough, rough_const), channel(metal, metal_const))
    )


def _parse_mdl(text: str) -> dict[str, Any]:
    """Read the constants and texture paths out of an OmniPBR-style ``.mdl`` file.

    Only the handful of inputs :meth:`_Conversion._omni_values` consumes. The MDL is
    never executed -- this is a text scrape of the material's instantiation.
    """
    out: dict[str, Any] = {}
    for key in _MDL_TEXTURE_KEYS:
        m = re.search(rf'{key}\s*:\s*texture_2d\(\s*"([^"]+)"', text)
        if m:
            out[key] = m.group(1)
    m = re.search(r"diffuse_color_constant\s*:\s*color\(([^)]*)\)", text)
    if m:
        parts = [float(re.sub(r"[fF]$", "", p.strip())) for p in m.group(1).split(",")]
        out["diffuse_color_constant"] = (
            (parts * 3)[:3] if len(parts) == 1 else parts[:3]
        )
    for key in _MDL_FLOAT_KEYS:
        m = re.search(rf"{key}\s*:\s*([0-9.eE+-]+)f?", text)
        if m:
            out[key] = float(m.group(1))
    for key in ("enable_ORM_texture", "enable_opacity"):
        m = re.search(rf"{key}\s*:\s*(true|false)", text)
        if m:
            out[key] = m.group(1) == "true"
    return out


__all__ = [
    "SceneGlbReport",
    "SceneGlbResult",
    "USDSceneToGLB",
    "convert_usd_scene_to_glb",
]
