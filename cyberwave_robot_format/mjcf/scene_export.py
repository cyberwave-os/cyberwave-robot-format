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
) -> bytes:
    """Export a CommonSchema to a complete MuJoCo ZIP file.

    This function exports a composed CommonSchema (which may contain multiple
    robots merged via merge_in) to a ZIP file containing:
    - mujoco_scene.xml: The complete MJCF scene
    - assets/: Directory with all required mesh files

    Args:
        schema: CommonSchema to export (may be a composed scene with multiple robots)
        output_path: Optional path to write the ZIP file. If None, returns bytes only.

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
                    
                    # Handle absolute paths and relative paths
                    if mesh_path.is_absolute() and mesh_path.exists():
                        src_path = mesh_path
                    else:
                        # Try relative to temp directory
                        src_path = temp_path / mesh_file
                        if not src_path.exists():
                            # Try as-is
                            src_path = mesh_path
                    
                    if src_path.exists():
                        # Copy to assets directory
                        dst_path = assets_dir / src_path.name
                        if not dst_path.exists():
                            shutil.copy(src_path, dst_path)
                        
                        # Update XML to reference basename only
                        mesh_elem.set("file", src_path.name)
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


__all__ = ["export_mujoco_zip"]
