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
import shutil
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from cyberwave_robot_format.mjcf.exporter import MJCFExporter
from cyberwave_robot_format.schema import CommonSchema

logger = logging.getLogger(__name__)


def export_mujoco_zip(
    schema: CommonSchema,
    output_path: str | Path | None = None,
    mesh_base_dirs: list[str | Path] | None = None,
) -> bytes:
    """Export a CommonSchema to a complete MuJoCo ZIP file.

    This function exports a composed CommonSchema (which may contain multiple
    robots merged via merge_in) to a ZIP file containing:
    - mujoco_scene.xml: The complete MJCF scene
    - assets/: Directory with all required mesh files

    Args:
        schema: CommonSchema to export (may be a composed scene with multiple robots)
        output_path: Optional path to write the ZIP file. If None, returns bytes only.
        mesh_base_dirs: Optional list of directories to search for mesh files.

    Returns:
        ZIP file contents as bytes

    Raises:
        ValueError: If schema validation fails
        Exception: If export fails

    Example:
        >>> from cyberwave_robot_format import CommonSchema, Metadata, export_mujoco_zip
        >>> schema = CommonSchema(metadata=Metadata(name="my_scene"))
        >>> # ... add robots via merge_in ...
        >>> zip_bytes = export_mujoco_zip(schema, "output/scene.zip")
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

        # Collect and copy mesh files
        asset_elem = root.find("asset")
        if asset_elem is not None:
            for mesh_elem in asset_elem.findall("mesh"):
                mesh_file = mesh_elem.get("file")
                if mesh_file:
                    mesh_path = Path(mesh_file)
                    src_path = None
                    
                    # Handle absolute paths
                    if mesh_path.is_absolute() and mesh_path.exists():
                        src_path = mesh_path
                    else:
                        # Try in provided mesh_base_dirs
                        search_dirs = list(mesh_base_dirs or []) + [temp_path]
                        for base_dir in search_dirs:
                            candidate = Path(base_dir) / mesh_file
                            if candidate.exists():
                                src_path = candidate
                                break
                        # If not found by path, try searching for filename
                        if src_path is None:
                            target_name = mesh_path.name
                            for base_dir in search_dirs:
                                base = Path(base_dir)
                                if base.exists():
                                    matches = list(base.rglob(target_name))
                                    if matches:
                                        src_path = matches[0]
                                        break
                    
                    if src_path and src_path.exists():
                        # Copy to assets directory
                        dst_path = assets_dir / src_path.name
                        if not dst_path.exists():
                            shutil.copy(src_path, dst_path)
                        
                        final_name = src_path.name
                        
                        # Convert DAE to OBJ for MuJoCo compatibility
                        if dst_path.suffix.lower() == ".dae":
                            try:
                                from cyberwave_robot_format.mesh import convert_dae_to_obj
                                obj_path = convert_dae_to_obj(str(dst_path), output_dir=assets_dir)
                                final_name = Path(obj_path).name
                                # Remove original DAE
                                dst_path.unlink()
                            except Exception as e:
                                logger.warning(f"Failed to convert DAE to OBJ: {e}")
                        
                        # Update XML to reference final filename
                        mesh_elem.set("file", final_name)
                    else:
                        logger.warning(f"Mesh file not found: {mesh_file}")

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


__all__ = ["export_mujoco_zip", "export_mujoco_scene_xml"]
