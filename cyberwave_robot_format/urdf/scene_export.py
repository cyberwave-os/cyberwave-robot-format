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

"""URDF ZIP export for composed CommonSchema.

This module provides a single entrypoint for exporting a composed CommonSchema
(potentially containing multiple robots merged via merge_in) to a complete
URDF scene ZIP file with all required mesh assets.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

from cyberwave_robot_format.schema import CommonSchema, GeometryType
from cyberwave_robot_format.urdf.exporter import URDFExporter

logger = logging.getLogger(__name__)


def export_urdf_zip(
    schema: CommonSchema,
    output_path: str | Path | None = None,
) -> bytes:
    """Export a CommonSchema to a complete URDF ZIP file.

    This function exports a composed CommonSchema (which may contain multiple
    robots merged via merge_in) to a ZIP file containing:
    - scene.urdf: The complete URDF scene
    - assets/: Directory with all required mesh files

    The mesh filenames in the URDF are rewritten to reference the assets/ directory
    with deterministic names to avoid collisions.

    Args:
        schema: CommonSchema to export (may be a composed scene with multiple robots)
        output_path: Optional path to write the ZIP file. If None, returns bytes only.

    Returns:
        ZIP file contents as bytes

    Raises:
        ValueError: If schema validation fails
        Exception: If export fails

    Example:
        >>> from cyberwave_robot_format import CommonSchema, Metadata
        >>> from cyberwave_robot_format.urdf import export_urdf_zip
        >>> schema = CommonSchema(metadata=Metadata(name="my_scene"))
        >>> # ... add robots via merge_in ...
        >>> zip_bytes = export_urdf_zip(schema, "output/scene_urdf.zip")
    """
    # Validate schema
    errors = schema.validate()
    if errors:
        raise ValueError(f"Schema validation failed: {', '.join(errors)}")

    # Create temporary directory for building the scene
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        assets_dir = temp_path / "assets"
        assets_dir.mkdir(exist_ok=True)

        # Collect all mesh files and build rewrite map
        mesh_rewrite_map: dict[str, str] = {}  # original_path -> new_relative_path
        mesh_counter: dict[str, int] = {}  # basename -> count for deduplication

        for link in schema.links:
            for geom_container in list(link.visuals) + list(link.collisions):
                if geom_container.geometry and geom_container.geometry.type == GeometryType.MESH:
                    original_filename = geom_container.geometry.filename
                    if not original_filename:
                        continue

                    if original_filename in mesh_rewrite_map:
                        # Already processed this mesh
                        continue

                    # Generate deterministic name
                    original_path = Path(original_filename)
                    basename = original_path.stem
                    extension = original_path.suffix or ".obj"

                    # Handle duplicates
                    count = mesh_counter.get(basename, 0)
                    if count > 0:
                        new_name = f"{basename}_{count}{extension}"
                    else:
                        new_name = f"{basename}{extension}"
                    mesh_counter[basename] = count + 1

                    new_relative_path = f"assets/{new_name}"
                    mesh_rewrite_map[original_filename] = new_relative_path

                    # Copy the mesh file if it exists
                    src_path = original_path if original_path.is_absolute() else None

                    # Try common locations for mesh files
                    if src_path is None or not src_path.exists():
                        # Check /tmp/mujoco_converted_meshes (where MJCFExporter puts converted files)
                        mujoco_mesh_dir = Path("/tmp/mujoco_converted_meshes")
                        if mujoco_mesh_dir.exists():
                            potential_src = mujoco_mesh_dir / original_path.name
                            if potential_src.exists():
                                src_path = potential_src

                    if src_path and src_path.exists():
                        dst_path = assets_dir / new_name
                        shutil.copy(src_path, dst_path)
                        logger.debug(f"Copied mesh: {src_path} -> {dst_path}")
                    else:
                        logger.warning(f"Mesh file not found: {original_filename}")

        # Create a modified schema with rewritten mesh paths
        # We need to modify the schema in-place for export
        for link in schema.links:
            for geom_container in list(link.visuals) + list(link.collisions):
                if geom_container.geometry and geom_container.geometry.type == GeometryType.MESH:
                    original_filename = geom_container.geometry.filename
                    if original_filename and original_filename in mesh_rewrite_map:
                        geom_container.geometry.filename = mesh_rewrite_map[original_filename]

        # Export URDF
        urdf_path = temp_path / "scene.urdf"
        exporter = URDFExporter()
        
        # #region agent log
        import json
        import os
        log_path = '/app/tmp/debug.log'
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        cube_link = next((l for l in schema.links if 'attached_cube' in l.name), None)
        with open(log_path, 'a') as f:
            f.write(json.dumps({"location":"scene_export.py:142","message":"Before URDF export","data":{"cube_link_exists":cube_link is not None,"cube_link_name":cube_link.name if cube_link else None,"cube_visuals_count":len(cube_link.visuals) if cube_link else 0,"cube_collisions_count":len(cube_link.collisions) if cube_link else 0,"cube_visual_geometry_type":cube_link.visuals[0].geometry.type.value if cube_link and cube_link.visuals and cube_link.visuals[0].geometry else None,"total_links":len(schema.links)},"timestamp":__import__('time').time()*1000,"sessionId":"debug-session","runId":"initial","hypothesisId":"B"}) + '\n')
        # #endregion
        
        exporter.export(schema, str(urdf_path))
        
        # #region agent log
        log_path = '/app/tmp/debug.log'
        with open(urdf_path, 'r') as urdf_file:
            urdf_content = urdf_file.read()
            cube_in_urdf = 'attached_cube' in urdf_content
            cube_visual_in_urdf = 'attached_cube' in urdf_content and '<visual>' in urdf_content[urdf_content.find('attached_cube'):urdf_content.find('attached_cube')+500] if 'attached_cube' in urdf_content else False
        with open(log_path, 'a') as f:
            f.write(json.dumps({"location":"scene_export.py:143","message":"After URDF export","data":{"cube_in_urdf":cube_in_urdf,"cube_visual_in_urdf":cube_visual_in_urdf,"urdf_file_size":len(urdf_content)},"timestamp":__import__('time').time()*1000,"sessionId":"debug-session","runId":"initial","hypothesisId":"B"}) + '\n')
        # #endregion

        # Create ZIP file
        zip_path = temp_path / "scene_urdf.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # Add scene URDF
            zf.write(urdf_path, "scene.urdf")

            # Add all assets
            for asset_file in assets_dir.rglob("*"):
                if asset_file.is_file():
                    arcname = f"assets/{asset_file.relative_to(assets_dir)}"
                    zf.write(asset_file, arcname)

        # Read ZIP contents
        with open(zip_path, "rb") as f:
            zip_bytes = f.read()

        # Optionally write to output path
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(zip_bytes)
            logger.info(f"Exported URDF ZIP to {output_path}")

        return zip_bytes


def export_urdf_scene_xml(schema: CommonSchema) -> str:
    """Export a CommonSchema to URDF XML string.

    This function exports a composed CommonSchema (which may contain multiple
    robots merged via merge_in) to a URDF XML string. Note that mesh file
    references in the XML will be the original paths from the schema.

    Args:
        schema: CommonSchema to export (may be a composed scene with multiple robots)

    Returns:
        URDF XML as a string

    Raises:
        ValueError: If schema validation fails
        Exception: If export fails
    """
    # Validate schema
    errors = schema.validate()
    if errors:
        raise ValueError(f"Schema validation failed: {', '.join(errors)}")

    # Create temporary file for export
    with tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False) as f:
        temp_path = f.name

    try:
        exporter = URDFExporter()
        exporter.export(schema, temp_path)

        with open(temp_path, "r", encoding="utf-8") as f:
            xml_content = f.read()

        return xml_content
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.unlink(temp_path)


__all__ = ["export_urdf_zip", "export_urdf_scene_xml"]
