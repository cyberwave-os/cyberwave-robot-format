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
        Exception: If mesh conversion fails.
    """
    if output_dir is None:
        output_dir = Path("/tmp/mujoco_converted_meshes")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate unique name based on path AND scale
    scale_tuple = (scale.x, scale.y, scale.z) if scale else (1.0, 1.0, 1.0)
    scale_str = f"{scale_tuple[0]}_{scale_tuple[1]}_{scale_tuple[2]}"
    h = hashlib.sha256(f"{dae_path}_{scale_str}".encode()).hexdigest()
    new_filename = output_dir / f"{Path(dae_path).stem}_{h}.obj"

    if new_filename.exists():
        return str(new_filename)

    logger.info(
        "Converting DAE to OBJ: %s -> %s (scale=%s).",
        dae_path,
        new_filename,
        scale_tuple,
    )

    mesh = trimesh.load(dae_path)

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


def convert_mesh_bytes_to_obj(
    mesh_bytes: bytes,
    original_filename: str,
    scale: Vector3 | None = None,
) -> bytes:
    """Convert DAE/STL mesh bytes to OBJ format in-memory.

    Cloud-safe converter that works with bytes without requiring
    filesystem access to original mesh files. This function is designed
    for use in cloud environments where mesh files are stored in object
    storage (S3, GCS, etc.) rather than on the local filesystem.

    Args:
        mesh_bytes: Raw mesh file bytes (DAE, STL, etc.)
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
    with tempfile.NamedTemporaryFile(
        suffix=os.path.splitext(original_filename)[1], delete=False
    ) as tmp_in:
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


def get_mesh_lookup_key(
    filename: str, scale: Vector3 | None
) -> tuple[str, tuple[float, float, float]]:
    """Generate a lookup key for mesh asset caching.

    Args:
        filename: Path to the mesh file.
        scale: Optional scale applied to the mesh.

    Returns:
        Tuple of (filename, scale_tuple) for use as a dictionary key.
    """
    scale_tuple = (scale.x, scale.y, scale.z) if scale else (1.0, 1.0, 1.0)
    return (filename, scale_tuple)


__all__ = ["convert_dae_to_obj", "convert_mesh_bytes_to_obj", "get_mesh_lookup_key"]
