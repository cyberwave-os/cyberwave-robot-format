"""
Mesh processing utilities.

Provides mesh format conversion functionality for DAE to OBJ
Used for MuJoCo compatibility, because MuJoCo does not natively support DAE files.
"""

from __future__ import annotations

import hashlib
import io
import logging
import math
import os
import re
import tempfile
from pathlib import Path

import numpy as np
import trimesh

from cyberwave_robot_format.math_utils import Vector3

logger = logging.getLogger(__name__)

# The edge of the flat swatch a bare colour is carried in. One texel would
# hold the colour; four keeps it a size every consumer downstream already
# handles, and matches the backend's copy of this correction.
_SWATCH_EDGE = 4

# Bumped whenever conversion starts producing something different for inputs
# that have not changed. The converted OBJ and trimesh's sidecar PNG are cached
# under a key built from the source path and the scale alone, so without this a
# host that converted a mesh under an older release keeps serving that old
# artifact forever -- the fix is computed, but never reached.
# v2: a COLLADA diffuse colour is no longer gamma-encoded into the atlas.
# v3: a COLLADA file's declared <unit> is applied.
_CONVERSION_REVISION = "3"

# A COLLADA file's root <asset> comes first; this much of the file holds it.
_COLLADA_ASSET_HEAD_BYTES = 64 * 1024
_COLLADA_ASSET = re.compile(rb"<asset\b.*?</asset>", re.DOTALL)
_COLLADA_UNIT_METER = re.compile(rb"<unit\b[^>]*\bmeter\s*=\s*[\"']([^\"']+)[\"']")


def collada_unit_scale(dae_bytes: bytes) -> float:
    """Metres per unit declared by a COLLADA file's root ``<asset><unit>``.

    trimesh records it as metadata but never scales the geometry, so a mesh
    authored in inches converts 39x oversize while the viewer, which honours
    it, shows it right. Both converters here apply it.

    The backend carries its own copy for its texture route, for the reason
    given at the end of ``_bake_display_referred_diffuse``.
    """
    asset = _COLLADA_ASSET.search(dae_bytes[:_COLLADA_ASSET_HEAD_BYTES])
    unit = _COLLADA_UNIT_METER.search(asset.group(0)) if asset else None
    try:
        scale = float(unit.group(1)) if unit else 1.0
    except ValueError:
        return 1.0
    return scale if math.isfinite(scale) and scale > 0 else 1.0


def _bake_display_referred_diffuse(loaded: object) -> None:
    """Hand COLLADA's diffuse colours to trimesh as pixels, not as a factor.

    When a mesh carries more than one material, trimesh concatenates them into
    a single texture atlas, and building that atlas reads each material's
    ``baseColorFactor`` through ``linear_to_srgb``
    (``trimesh/visual/material.py``, ``get_base_color_texture``) because the
    glTF spec defines that factor as a *linear* multiplier. A COLLADA
    ``<diffuse><color>`` is not linear — trimesh's own COLLADA loader puts the
    file's display-referred value straight into that field — so the converted
    mesh renders visibly lighter than the DAE says it is. Measured on the DEEP
    Robotics M20: RAL 7024 ``#3D4A56`` came out ``#86939D``.

    Writing the colour into a flat ``baseColorTexture`` and clearing the factor
    moves the material onto the packer's image branch, which copies pixels
    through untouched. Correcting the factor instead would be both lossy
    (it is stored as uint8) and wrong for a single-material mesh, which never
    reaches the packer and so is never transformed.

    A single-material mesh is rewritten too, although its colour was never at
    risk of being encoded twice, and that is deliberate rather than
    incidental. Correcting the colour is not the only thing this does: it is
    also what gets the colour onto disk at all. ``trimesh``'s OBJ exporter
    writes a sidecar PNG for an image, but only an ``mtl`` ``Kd`` line for a
    bare factor, and ``converted_mesh_texture`` matches ``map_Kd`` — so a bare
    colour left alone reaches the MJCF as nothing and the part renders in
    MuJoCo's default grey. The backend's copy of this correction *does* skip
    the single-material case, because ``_base_color_png`` there builds a
    swatch out of the factor itself and so has its own way of carrying a bare
    colour. This copy has none, which is why the two differ here.

    Only a material carrying a bare colour is touched. One that references an
    image keeps its factor, which genuinely is a multiplier in that branch.

    The cloud scene exporter in ``cyberwave-backend`` carries the same
    correction (``src/lib/robot_export/mesh_textures.py``) rather than calling
    this one: it depends on ``cyberwave-robot-format>=0.1.3``, so importing a
    symbol added here would break every install that resolves an older release,
    and the floor cannot be raised to a version that is not published yet.
    """
    try:
        from PIL import Image
    except ImportError:
        # Pillow is not a base dependency. Without it trimesh cannot build an
        # atlas at all, so there is no converted colour here to get wrong.
        return

    from trimesh.visual.material import PBRMaterial

    geometries = getattr(loaded, "geometry", None)
    meshes = list(geometries.values()) if geometries else [loaded]
    for mesh in meshes:
        material = getattr(getattr(mesh, "visual", None), "material", None)
        if not isinstance(material, PBRMaterial):
            continue
        factor = material.baseColorFactor
        if material.baseColorTexture is not None or factor is None:
            continue
        channels = [int(channel) for channel in np.asarray(factor).reshape(-1)[:4]]
        if len(channels) < 3:
            continue
        rgba = tuple(channels) + (255,) * (4 - len(channels))
        material.baseColorTexture = Image.new(
            "RGBA", (_SWATCH_EDGE, _SWATCH_EDGE), rgba
        )
        material.baseColorFactor = None


def convert_dae_to_obj(
    dae_path: str,
    scale: Vector3 | None = None,
    output_dir: Path | None = None,
) -> str:
    """Convert a DAE (Collada) mesh file to OBJ format.

    MuJoCo does not natively support DAE files, so this function converts
    them to OBJ format using trimesh. The file's declared unit and the scale
    are baked into the vertices.

    Args:
        dae_path: Path to the input DAE file.
        scale: Optional scale to apply to mesh vertices, on top of the unit.
        output_dir: Directory to write the output OBJ file. Defaults to
            /tmp/mujoco_converted_meshes.

    Returns:
        Path to the converted OBJ file.

    Raises:
        FileNotFoundError: If ``dae_path`` is not a file on this host and no
            earlier conversion of it is cached.
        Exception: If mesh conversion fails.
    """
    if output_dir is None:
        output_dir = Path("/tmp/mujoco_converted_meshes")

    # Generate unique name based on path AND scale AND converter revision
    scale_tuple = (scale.x, scale.y, scale.z) if scale else (1.0, 1.0, 1.0)
    scale_str = f"{scale_tuple[0]}_{scale_tuple[1]}_{scale_tuple[2]}"
    h = hashlib.sha256(
        f"{dae_path}_{scale_str}_{_CONVERSION_REVISION}".encode()
    ).hexdigest()
    # One directory per conversion. trimesh names an OBJ's sidecars after the
    # material rather than the mesh -- `material.mtl`, `material_0.png` -- so a
    # shared output directory has every textured mesh overwriting the previous
    # one's image, and each then renders with whichever happened to be written
    # last. The hashed name only ever protected the .obj itself.
    mesh_dir = output_dir / f"{Path(dae_path).stem}_{h}"
    new_filename = mesh_dir / f"{Path(dae_path).stem}.obj"

    if new_filename.exists():
        return str(new_filename)

    if not Path(dae_path).is_file():
        # Deliberately after the cache probe: a mesh whose source has since
        # moved out of reach still resolves to its converted OBJ. Raised rather
        # than logged so the caller can word the message for its own context.
        raise FileNotFoundError(f"DAE mesh is not a file on this host: {dae_path}")

    logger.info(
        "Converting DAE to OBJ: %s -> %s (scale=%s).",
        dae_path,
        new_filename,
        scale_tuple,
    )

    # Load before creating anything on disk: a path trimesh cannot open (a
    # missing file, an ``asset://`` URI) must not leave an empty per-mesh
    # directory behind on every failed attempt.
    mesh = trimesh.load(dae_path)
    # Before anything merges the materials: the merge is where a DAE's own
    # colours would be gamma-encoded a second time.
    _bake_display_referred_diffuse(mesh)
    mesh_dir.mkdir(parents=True, exist_ok=True)

    # Handle Scene objects (flatten to single mesh)
    if isinstance(mesh, trimesh.Scene):
        if len(mesh.geometry) > 0:
            geom = mesh.dump(concatenate=True)
        else:
            geom = trimesh.Trimesh()
    else:
        geom = mesh

    with open(dae_path, "rb") as handle:
        unit = collada_unit_scale(handle.read(_COLLADA_ASSET_HEAD_BYTES))
    if unit != 1.0:
        geom.apply_scale(unit)

    # Apply the scale to the mesh
    if scale:
        transform = np.eye(4)
        transform[0, 0] = scale.x
        transform[1, 1] = scale.y
        transform[2, 2] = scale.z
        geom.apply_transform(transform)

    geom.export(str(new_filename))
    return str(new_filename)


def converted_mesh_texture(obj_path: str) -> str | None:
    """The texture image `convert_dae_to_obj` wrote beside `obj_path`, if any.

    trimesh writes the mesh's base-colour map out as a sibling PNG when it
    exports an OBJ, along with an `.mtl` referencing it. MuJoCo does not read
    `.mtl`, so the image sits there unused unless the MJCF declares it — which
    is what the exporter uses this to do. Returns an *absolute* path: the MJCF
    is written somewhere else entirely and MuJoCo resolves a relative asset
    against `texturedir`/the model file, not against the mesh. Any consumer that
    ships the MJCF off this host has to repackage the image itself.
    """
    mesh_dir = Path(obj_path).parent
    # Safe to read the whole directory because `convert_dae_to_obj` gives each
    # conversion its own; the sidecar names are trimesh's, not the mesh's.
    for mtl in sorted(mesh_dir.glob("*.mtl")):
        for line in mtl.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip().lower().startswith("map_kd"):
                continue
            resolved = mesh_dir / line.split(None, 1)[1].strip()
            if resolved.exists():
                return str(resolved)
    for candidate in sorted(mesh_dir.glob("*.png")):
        return str(candidate)
    return None


def convert_mesh_bytes_to_obj(
    mesh_bytes: bytes,
    original_filename: str,
    scale: Vector3 | None = None,
) -> bytes:
    """Convert mesh bytes (DAE/GLB/glTF/STL/…) to OBJ format in-memory.

    Cloud-safe converter that works with bytes without requiring
    filesystem access to original mesh files. This function is designed
    for use in cloud environments where mesh files are stored in object
    storage (S3, GCS, etc.) rather than on the local filesystem.

    Any format trimesh can load is accepted; the extension is taken from
    ``original_filename`` so trimesh picks the right loader. This is how
    MuJoCo-undecodable formats (DAE, GLB/glTF) are turned into OBJ for the
    MuJoCo scene export. A DAE's declared unit is applied
    (``collada_unit_scale``).

    Args:
        mesh_bytes: Raw mesh file bytes (DAE, GLB, glTF, STL, etc.)
        original_filename: Original filename (used for extension detection)
        scale: Optional scale to apply to mesh vertices, on top of the unit

    Returns:
        OBJ format bytes

    Raises:
        Exception: If mesh conversion fails

    Example:
        >>> dae_bytes = storage.read('model.dae')
        >>> obj_bytes = convert_mesh_bytes_to_obj(dae_bytes, 'model.dae')
        >>> storage.write('model.obj', obj_bytes)
    """
    # Write to temp file (trimesh requires file path for loading)
    with tempfile.NamedTemporaryFile(suffix=os.path.splitext(original_filename)[1], delete=False) as tmp_in:
        tmp_in.write(mesh_bytes)
        tmp_in_path = tmp_in.name

    try:
        mesh = trimesh.load(tmp_in_path)

        # Handle Scene objects (flatten to single mesh)
        if isinstance(mesh, trimesh.Scene):
            if len(mesh.geometry) > 0:
                geom = mesh.dump(concatenate=True)
            else:
                geom = trimesh.Trimesh()
        else:
            geom = mesh

        if original_filename.lower().endswith(".dae"):
            unit = collada_unit_scale(mesh_bytes)
            if unit != 1.0:
                geom.apply_scale(unit)

        # Apply scale transform if provided
        if scale:
            transform = np.eye(4)
            transform[0, 0] = scale.x
            transform[1, 1] = scale.y
            transform[2, 2] = scale.z
            geom.apply_transform(transform)

        # Export to OBJ bytes
        obj_buffer = io.BytesIO()
        geom.export(obj_buffer, file_type="obj")
        return obj_buffer.getvalue()
    finally:
        os.unlink(tmp_in_path)


def get_mesh_lookup_key(filename: str, scale: Vector3 | None) -> tuple[str, tuple[float, float, float]]:
    """Generate a lookup key for mesh asset caching.

    Args:
        filename: Path to the mesh file.
        scale: Optional scale applied to the mesh.

    Returns:
        Tuple of (filename, scale_tuple) for use as a dictionary key.
    """
    scale_tuple = (scale.x, scale.y, scale.z) if scale else (1.0, 1.0, 1.0)
    return (filename, scale_tuple)


def decompose_collision_mesh(
    mesh: trimesh.Trimesh, *, threshold_m: float = 0.001, max_hulls: int = 32
) -> list[trimesh.Trimesh]:
    """Prepare convex collision pieces offline, in the input mesh's frame.

    Supply vertices already scaled to metres. Keep visual meshes and authored
    link inertia unchanged; replace only the selected collision mesh with these
    pieces, retaining its pose/contact properties. MuJoCo otherwise fills
    concavities such as gripper openings with one convex hull. This is explicit
    asset preparation, never an implicit exporter/runtime geometry change.

    CoACD's hull-count cap can relax its requested concavity threshold. Review
    the resulting collision geometry before treating it as calibrated.
    """
    if not np.isfinite(threshold_m) or threshold_m <= 0:
        raise ValueError("Collision decomposition requires a positive metric threshold")
    if isinstance(max_hulls, bool) or not isinstance(max_hulls, int) or not 1 <= max_hulls <= 128:
        raise ValueError("Collision decomposition requires 1..128 hulls")
    if (
        not isinstance(mesh, trimesh.Trimesh)
        or len(mesh.vertices) < 4
        or len(mesh.faces) < 4
        or not np.isfinite(mesh.vertices).all()
    ):
        raise ValueError("Collision decomposition requires a finite triangle mesh")
    try:
        import coacd
    except ImportError as exc:
        raise RuntimeError("Install cyberwave-robot-format[collision] for convex decomposition") from exc
    parts = coacd.run_coacd(
        coacd.Mesh(np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int32)),
        threshold=threshold_m,
        real_metric=True,
        max_convex_hull=max_hulls,
        seed=0,
    )
    if not parts:
        raise ValueError("Collision decomposition returned no geometry")
    result = [trimesh.Trimesh(vertices=v, faces=f, process=False) for v, f in parts]
    if any(not np.isfinite(part.vertices).all() or part.volume <= 0 for part in result):
        raise ValueError("Collision decomposition returned invalid geometry")
    return result


__all__ = [
    "convert_dae_to_obj",
    "convert_mesh_bytes_to_obj",
    "converted_mesh_texture",
    "get_mesh_lookup_key",
    "decompose_collision_mesh",
]
