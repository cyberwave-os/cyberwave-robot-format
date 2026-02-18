"""
Utility functions for format detection, validation, and conversion support.
"""

import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from cyberwave_robot_format.schema import CommonSchema

logger = logging.getLogger(__name__)


def detect_format(file_path):
    """Detect the format of a robot description file based on extension and content.

    Args:
        file_path: Path to the file to analyze (str or Path object)

    Returns:
        Format name string or None if format cannot be detected
    """
    # Convert string to Path object if needed
    if isinstance(file_path, str):
        from pathlib import Path

        file_path = Path(file_path)

    if not file_path.exists():
        return None

    # Check file extension first
    ext = file_path.suffix.lower()
    ext_mapping = {
        ".urdf": "urdf",
        ".sdf": "sdf",
        ".world": "sdf",
        ".mjcf": "mjcf",
        ".xml": None,  # Need content analysis
        ".yaml": "schema",
        ".yml": "schema",
        ".json": "schema",
        ".usd": "usd",
        ".usda": "usd",
        ".obj": "obj",
        ".mtl": "obj",
    }

    format_guess: str | None = None
    if ext in ext_mapping:
        format_guess = ext_mapping[ext]
        if format_guess is not None:
            return format_guess

    # Content-based detection for ambiguous extensions
    try:
        if ext in [".xml", ".mjcf"] or format_guess is None:
            return _detect_xml_format(file_path)
        elif ext in [".yaml", ".yml", ".json"]:
            return _detect_schema_format(file_path)
    except Exception as e:
        logger.debug(f"Error during content detection: {e}")
        return None

    return None


def _detect_xml_format(file_path: Path) -> str | None:
    """Detect XML-based format by analyzing root element and structure."""
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

        # Check root element tag
        if root.tag == "robot":
            return "urdf"
        elif root.tag == "sdf":
            return "sdf"
        elif root.tag == "mujoco":
            return "mjcf"
        elif root.tag == "world":
            return "sdf"  # SDF world file

        # Check for characteristic elements
        if root.find(".//link") is not None and root.find(".//joint") is not None:
            if root.find(".//inertial") is not None:
                return "urdf"  # URDF has inertial elements

        if root.find(".//model") is not None:
            return "sdf"  # SDF uses model elements

        if root.find(".//worldbody") is not None or root.find(".//body") is not None:
            return "mjcf"  # MuJoCo uses worldbody/body elements

    except ET.ParseError:
        logger.debug(f"XML parsing failed for {file_path}")

    return None


def _detect_schema_format(file_path: Path) -> str:
    """Detect schema format for YAML/JSON files.

    Currently assumes all YAML/JSON files use the common schema format.
    """
    # For now, assume all YAML/JSON files are schema format
    # Could be enhanced to detect specific schema structures
    return "schema"


def validate_schema(schema: CommonSchema) -> None:
    """Validate common schema and raise exception if invalid.

    Args:
        schema: CommonSchema instance to validate

    Raises:
        ValueError: If schema validation fails
    """
    issues = schema.validate()
    if issues:
        error_msg = "Schema validation failed:\n" + "\n".join(
            f"  - {issue}" for issue in issues
        )
        raise ValueError(error_msg)


def conversion_matrix() -> dict[str, dict[str, bool]]:
    """Return conversion compatibility matrix between formats.

    Returns:
        Nested dict indicating conversion support: matrix[source][target] = supported
    """
    # Define which conversions are supported/recommended
    # True = full support, False = limited/lossy support
    formats = ["urdf", "sdf", "mjcf", "usd", "schema"]

    matrix = {}
    for source in formats:
        matrix[source] = {}
        for target in formats:
            if source == target:
                matrix[source][target] = True
            elif source == "schema":
                # Schema can export to all formats
                matrix[source][target] = True
            elif target == "schema":
                # All formats can be parsed to schema
                matrix[source][target] = True
            else:
                # Cross-format conversions via schema
                matrix[source][target] = True

    # Mark some conversions as limited due to semantic differences
    limited_conversions = [
        ("mjcf", "urdf"),  # MuJoCo features don't map well to URDF
        ("usd", "urdf"),  # USD is much more general
        ("sdf", "urdf"),  # SDF has features URDF lacks
    ]

    for source, target in limited_conversions:
        if source in matrix and target in matrix[source]:
            # Still supported but may have limitations
            matrix[source][target] = True

    return matrix


def get_format_info(format_name: str) -> dict[str, Any]:
    """Get information about a specific format.

    Args:
        format_name: Name of the format

    Returns:
        Dictionary with format information
    """
    format_info = {
        "urdf": {
            "name": "Unified Robot Description Format",
            "extensions": [".urdf"],
            "type": "xml",
            "description": "ROS standard robot description format",
            "features": ["kinematics", "dynamics", "collision", "visual"],
            "limitations": ["limited_sensors", "no_actuators", "basic_materials"],
        },
        "sdf": {
            "name": "Simulation Description Format",
            "extensions": [".sdf", ".world"],
            "type": "xml",
            "description": "Gazebo simulation format with advanced features",
            "features": ["kinematics", "dynamics", "sensors", "plugins", "worlds"],
            "limitations": ["gazebo_specific"],
        },
        "mjcf": {
            "name": "MuJoCo Model Format",
            "extensions": [".mjcf", ".xml"],
            "type": "xml",
            "description": "MuJoCo physics simulator format",
            "features": ["advanced_dynamics", "constraints", "actuators", "sensors"],
            "limitations": ["mujoco_specific", "different_conventions"],
        },
        "usd": {
            "name": "Universal Scene Description",
            "extensions": [".usd", ".usda"],
            "type": "binary/text",
            "description": "Pixar USD format for 3D content",
            "features": ["advanced_graphics", "animation", "composition"],
            "limitations": ["graphics_focused", "limited_physics"],
        },
        "schema": {
            "name": "FIGAROH Common Schema",
            "extensions": [".yaml", ".yml", ".json"],
            "type": "structured_data",
            "description": "Unified intermediate representation",
            "features": ["format_agnostic", "extensible", "validation"],
            "limitations": ["intermediate_format"],
        },
    }

    return format_info.get(format_name.lower(), {})


def sanitize_name(name: str) -> str:
    """Sanitize name for cross-format compatibility.

    Args:
        name: Original name string

    Returns:
        Sanitized name safe for all formats
    """
    # Remove/replace characters that are problematic in XML or other formats
    # Keep alphanumeric, underscore, hyphen
    sanitized = re.sub(r"[^\w\-]", "_", name)

    # Ensure it starts with letter or underscore
    if sanitized and sanitized[0].isdigit():
        sanitized = "_" + sanitized

    # Handle empty results
    if not sanitized:
        sanitized = "unnamed"

    return sanitized


def merge_extensions(
    base_ext: dict[str, Any], new_ext: dict[str, Any]
) -> dict[str, Any]:
    """Merge extension dictionaries, handling conflicts appropriately.

    Args:
        base_ext: Base extensions dictionary
        new_ext: New extensions to merge

    Returns:
        Merged extensions dictionary
    """
    merged = base_ext.copy()

    for key, value in new_ext.items():
        if key in merged:
            if isinstance(merged[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                merged[key] = merge_extensions(merged[key], value)
            else:
                # New value overwrites existing
                merged[key] = value
        else:
            merged[key] = value

    return merged


def get_supported_conversions() -> list[tuple]:
    """Get list of all supported conversion pairs.

    Returns:
        List of (source_format, target_format) tuples
    """
    matrix = conversion_matrix()
    conversions = []

    for source, targets in matrix.items():
        for target, supported in targets.items():
            if supported and source != target:
                conversions.append((source, target))

    return conversions


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted size string
    """
    size: float = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def compare_schemas(schema1: CommonSchema, schema2: CommonSchema) -> dict[str, Any]:
    """Compare two schemas and return differences.

    Args:
        schema1: First schema to compare
        schema2: Second schema to compare

    Returns:
        Dictionary describing differences between schemas
    """
    differences = {
        "metadata": {},
        "links": {},
        "joints": {},
        "actuators": {},
        "sensors": {},
        "summary": {},
    }

    # Compare counts
    differences["summary"] = {
        "links": (len(schema1.links), len(schema2.links)),
        "joints": (len(schema1.joints), len(schema2.joints)),
        "actuators": (len(schema1.actuators), len(schema2.actuators)),
        "sensors": (len(schema1.sensors), len(schema2.sensors)),
    }

    # Compare names
    schema1_link_names = {link.name for link in schema1.links}
    schema2_link_names = {link.name for link in schema2.links}

    differences["links"] = {
        "only_in_first": schema1_link_names - schema2_link_names,
        "only_in_second": schema2_link_names - schema1_link_names,
        "common": schema1_link_names & schema2_link_names,
    }

    schema1_joint_names = {joint.name for joint in schema1.joints}
    schema2_joint_names = {joint.name for joint in schema2.joints}

    differences["joints"] = {
        "only_in_first": schema1_joint_names - schema2_joint_names,
        "only_in_second": schema2_joint_names - schema1_joint_names,
        "common": schema1_joint_names & schema2_joint_names,
    }

    return differences
