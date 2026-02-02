# Copyright [2025] Tomáš Macháček <tomasmachacekw@gmail.com>

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# distributed under the License.

"""MuJoCo ZIP export for composed CommonSchema.

This module provides a single entrypoint for exporting a composed CommonSchema
(potentially containing multiple robots merged via merge_in) to a complete
MuJoCo scene ZIP file with all required mesh assets.
"""

from __future__ import annotations

import logging
import os
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Callable
from pathlib import Path

from cyberwave_robot_format.mjcf.exporter import MJCFExporter
from cyberwave_robot_format.schema import CommonSchema

logger = logging.getLogger(__name__)

# Type alias for mesh resolver that returns (final_filename, bytes)
# This allows in-memory conversion: resolve "foo.dae" -> ("foo.obj", obj_bytes)
MeshResolver = Callable[[str], tuple[str, bytes] | None]


def export_mujoco_zip_cloud(
    schema: CommonSchema,
    mesh_resolver: MeshResolver,
    output_path: str | Path | None = None,
    strict_missing_meshes: bool = False,
) -> bytes:
    """Export a CommonSchema to a complete MuJoCo ZIP file (cloud-native).
    
    This is a cloud-native implementation requiring a mesh_resolver.
    No filesystem access is performed - all meshes must be provided via the resolver.

    This function exports a composed CommonSchema (which may contain multiple
    robots merged via merge_in) to a ZIP file containing:
    - mujoco_scene.xml: The complete MJCF scene
    - assets/: Directory with all required mesh files

    This is a cloud-native implementation that requires mesh_resolver.
    No filesystem access is performed.

    Args:
        schema: CommonSchema to export (may be a composed scene with multiple robots)
        mesh_resolver: Function that takes a mesh filename and returns:
                      - tuple[str, bytes]: (final_filename, bytes) for in-memory conversion
                      - None: mesh not found
                      Enables cloud-safe in-memory conversion:
                      resolve "foo.dae" -> ("foo.obj", obj_bytes)
        output_path: Optional path to write the ZIP file. If None, returns bytes only.
        strict_missing_meshes: If True, raise FileNotFoundError for any missing meshes.
                              If False (default), log warnings and continue.

    Returns:
        ZIP file contents as bytes

    Raises:
        ValueError: If schema validation fails
        FileNotFoundError: If strict_missing_meshes=True and any meshes are missing
        Exception: If export fails

    Example:
        >>> from cyberwave_robot_format.mjcf import export_mujoco_zip_cloud
        >>> schema = CommonSchema(metadata=Metadata(name="my_scene"))
        >>> # ... add robots via merge_in ...
        >>> 
        >>> # Cloud-safe in-memory conversion:
        >>> def my_resolver(filename: str) -> tuple[str, bytes] | None:
        ...     if filename.endswith('.dae'):
        ...         dae_bytes = download_from_s3(filename)
        ...         obj_bytes = convert_dae_to_obj_in_memory(dae_bytes)
        ...         return (filename.replace('.dae', '.obj'), obj_bytes)
        ...     return None
        >>> 
        >>> zip_bytes = export_mujoco_zip_cloud(schema, my_resolver)
    """
    # Validate inputs
    if not callable(mesh_resolver):
        raise ValueError("mesh_resolver must be a callable function")
    
    errors = schema.validate()
    if errors:
        raise ValueError(f"Schema validation failed: {', '.join(errors)}")

    # Create temporary directory for building the scene
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        assets_dir = temp_path / "assets"
        assets_dir.mkdir(exist_ok=True)

        # Export MJCF to temporary file
        mjcf_path = temp_path / "mujoco_scene.xml"
        exporter = MJCFExporter()
        exporter.export(schema, str(mjcf_path))

        # Parse the generated XML to collect mesh files
        tree = ET.parse(mjcf_path)
        root = tree.getroot()

        # Update compiler to use assets directory
        compiler = root.find("compiler")
        if compiler is not None:
            compiler.set("assetdir", "assets")
        else:
            # Insert compiler as first child
            compiler = ET.Element("compiler", angle="radian", assetdir="assets")
            root.insert(0, compiler)

        # Build mesh rewrite map to avoid collisions in composed scenes
        mesh_rewrite_map: dict[str, str] = {}  # original_mesh_file -> unique_asset_name
        mesh_counter: dict[str, int] = {}  # basename -> count for deduplication
        missing_meshes: list[str] = []

        # Collect and copy mesh files
        asset_elem = root.find("asset")
        if asset_elem is not None:
            for mesh_elem in asset_elem.findall("mesh"):
                mesh_file = mesh_elem.get("file")
                if not mesh_file:
                    continue

                # Skip if already processed (deduplication)
                if mesh_file in mesh_rewrite_map:
                    mesh_elem.set("file", mesh_rewrite_map[mesh_file])
                    logger.debug(f"Reusing existing mesh: {mesh_file} -> {mesh_rewrite_map[mesh_file]}")
                    continue

                mesh_path = Path(mesh_file)
                final_filename = None
                mesh_bytes = None
                
                # Resolve mesh via mesh_resolver (cloud-native)
                if mesh_resolver:
                    logger.debug(f"Trying mesh_resolver for: {mesh_file}")
                    result = mesh_resolver(mesh_file)
                    
                    if result is not None:
                        # Expect tuple: (final_filename, bytes)
                        final_filename, mesh_bytes = result
                        logger.debug(f"  Resolved via mesh_resolver: {mesh_file} -> {final_filename} ({len(mesh_bytes)} bytes)")
                
                # Process the mesh if we got bytes
                if mesh_bytes is not None and final_filename is not None:
                    # Generate unique asset name to avoid collisions
                    final_path = Path(final_filename)
                    basename = final_path.stem
                    extension = final_path.suffix or ".obj"
                    
                    # Handle duplicates
                    count = mesh_counter.get(basename, 0)
                    if count > 0:
                        unique_name = f"{basename}_{count}{extension}"
                    else:
                        unique_name = f"{basename}{extension}"
                    mesh_counter[basename] = count + 1
                    
                    # Write to assets directory
                    dst_path = assets_dir / unique_name
                    dst_path.write_bytes(mesh_bytes)
                    logger.debug(f"Wrote mesh to assets: {dst_path}")
                    
                    final_asset_name = unique_name
                    
                    # Update rewrite map and XML
                    mesh_rewrite_map[mesh_file] = final_asset_name
                    mesh_elem.set("file", final_asset_name)
                    logger.debug(f"Updated XML mesh reference: {mesh_file} -> {final_asset_name}")
                else:
                    # Mesh not found
                    missing_meshes.append(mesh_file)
                    logger.error(f"Mesh file not found: {mesh_file}")
                    logger.error(f"  mesh_resolver returned None for this file")
        
        # Handle missing meshes
        if missing_meshes and strict_missing_meshes:
            raise FileNotFoundError(
                f"Missing {len(missing_meshes)} mesh file(s): {', '.join(missing_meshes)}"
            )

        # Write updated XML
        tree.write(mjcf_path, encoding="utf-8", xml_declaration=True)

        # Create ZIP file
        zip_path = temp_path / "scene.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # Add scene XML
            zf.write(mjcf_path, "mujoco_scene.xml")

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
            logger.info(f"Exported MuJoCo ZIP to {output_path}")

        return zip_bytes


def export_mujoco_scene_xml(schema: CommonSchema) -> str:
    """Export a CommonSchema to MJCF XML string.

    This function exports a composed CommonSchema (which may contain multiple
    robots merged via merge_in) to an MJCF XML string. Note that mesh file
    references in the XML will be absolute paths to the converted files
    in /tmp/mujoco_converted_meshes/.

    Args:
        schema: CommonSchema to export (may be a composed scene with multiple robots)

    Returns:
        MJCF XML as a string

    Raises:
        ValueError: If schema validation fails
        Exception: If export fails
    """
    # Validate schema
    errors = schema.validate()
    if errors:
        raise ValueError(f"Schema validation failed: {', '.join(errors)}")

    # Create temporary file for export
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
        temp_path = f.name

    try:
        exporter = MJCFExporter()
        exporter.export(schema, temp_path)

        with open(temp_path, "r", encoding="utf-8") as f:
            xml_content = f.read()

        return xml_content
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.unlink(temp_path)


__all__ = ["export_mujoco_zip_cloud", "export_mujoco_scene_xml"]
