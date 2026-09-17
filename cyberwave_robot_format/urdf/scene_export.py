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
import tempfile
import zipfile
from pathlib import Path

import defusedxml.ElementTree as SafeET

from cyberwave_robot_format.mesh.resolution import CloudAssetBundle, MeshResolver
from cyberwave_robot_format.schema import CommonSchema
from cyberwave_robot_format.urdf.exporter import URDFExporter

logger = logging.getLogger(__name__)


def export_urdf_zip_cloud(
    schema: CommonSchema,
    mesh_resolver: MeshResolver,
    output_path: str | Path | None = None,
    strict_missing_meshes: bool = False,
) -> bytes:
    """Export a CommonSchema to a complete URDF ZIP file (cloud-native).
    
    This is a cloud-native implementation requiring a mesh_resolver.
    No filesystem access is performed - all meshes must be provided via the resolver.

    This function exports a composed CommonSchema (which may contain multiple
    robots merged via merge_in) to a ZIP file containing:
    - scene.urdf: The complete URDF scene
    - assets/: Directory with all required mesh files

    The mesh filenames in the URDF are rewritten to reference the assets/ directory
    with deterministic names to avoid collisions.

    This is a cloud-native implementation that requires mesh_resolver.
    No filesystem access is performed.

    IMPORTANT: This function does NOT mutate the input schema. It rewrites only the exported XML.

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
        >>> from cyberwave_robot_format import CommonSchema, Metadata
        >>> from cyberwave_robot_format.urdf import export_urdf_zip_cloud
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
        >>> zip_bytes = export_urdf_zip_cloud(schema, my_resolver)
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

        urdf_path = temp_path / "scene.urdf"
        URDFExporter().export(schema, str(urdf_path))
        tree = SafeET.parse(urdf_path)
        root = tree.getroot()
        bundle = CloudAssetBundle(assets_dir, mesh_resolver)
        missing_meshes: list[str] = []

        # Meshes and declared textures share the same resolver and namespace.
        # Rewrite the exported document so the caller's schema stays untouched.
        for item in [*root.iter("mesh"), *root.iter("texture")]:
            source = item.get("filename")
            if not source:
                continue
            filename = bundle.add(
                source, default_extension=".png" if item.tag == "texture" else ".obj"
            )
            if filename is not None:
                item.set("filename", f"assets/{filename}")
            elif item.tag == "texture":
                raise FileNotFoundError(f"Missing texture asset: {source}")
            else:
                missing_meshes.append(source)
                logger.error("Mesh file not found: %s", source)

        if missing_meshes and strict_missing_meshes:
            raise FileNotFoundError(
                f"Missing {len(missing_meshes)} mesh file(s): {', '.join(missing_meshes)}"
            )
        tree.write(urdf_path, encoding="utf-8", xml_declaration=True)

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


__all__ = ["export_urdf_zip_cloud", "export_urdf_scene_xml"]
