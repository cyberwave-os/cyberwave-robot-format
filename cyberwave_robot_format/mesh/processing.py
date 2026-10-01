"""
Mesh processing utilities.

Provides mesh format conversion functionality for DAE to OBJ
Used for MuJoCo compatibility, because MuJoCo does not natively support DAE files.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import trimesh

from cyberwave_robot_format.math_utils import Vector3

logger = logging.getLogger(__name__)


def convert_dae_to_obj(
    dae_path: str,
    scale: Vector3 | None = None,
    output_dir: Path | None = None,
) -> str:
    """Convert a DAE (Collada) mesh file to OBJ format.

    MuJoCo does not natively support DAE files, so this function converts
    them to OBJ format using trimesh. The scale is baked into the vertices.

    Args:
        dae_path: Path to the input DAE file.
        scale: Optional scale to apply to mesh vertices.
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

    # Generate unique name based on path AND scale
    scale_tuple = (scale.x, scale.y, scale.z) if scale else (1.0, 1.0, 1.0)
    scale_str = f"{scale_tuple[0]}_{scale_tuple[1]}_{scale_tuple[2]}"
    h = hashlib.sha256(f"{dae_path}_{scale_str}".encode()).hexdigest()
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
    mesh_dir.mkdir(parents=True, exist_ok=True)

    # Handle Scene objects (flatten to single mesh)
    if isinstance(mesh, trimesh.Scene):
        if len(mesh.geometry) > 0:
            geom = mesh.dump(concatenate=True)
        else:
            geom = trimesh.Trimesh()
    else:
        geom = mesh

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
    MuJoCo scene export.

    Args:
        mesh_bytes: Raw mesh file bytes (DAE, GLB, glTF, STL, etc.)
        original_filename: Original filename (used for extension detection)
        scale: Optional scale to apply to mesh vertices

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
