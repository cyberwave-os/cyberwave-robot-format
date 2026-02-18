"""Universal schema JSON export utilities.

This module provides functions for exporting CommonSchema to JSON format.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from cyberwave_robot_format.schema import CommonSchema

logger = logging.getLogger(__name__)


def export_universal_schema_json(
    schema: CommonSchema,
    output_path: str | Path | None = None,
) -> bytes:
    """Export a CommonSchema to JSON format.

    Args:
        schema: CommonSchema to export (may be a composed scene with multiple robots)
        output_path: Path to write the JSON file. If None, returns bytes only.

    Returns:
        JSON file contents as bytes

    Raises:
        ValueError: If schema validation fails

    Example:
        >>> from cyberwave_robot_format import CommonSchema, Metadata, export_universal_schema_json
        >>> schema = CommonSchema(metadata=Metadata(name="my_scene"))
        >>> # ... add robots via merge_in ...
        >>> json_bytes = export_universal_schema_json(schema, "output/scene.json")
    """
    # Validate schema
    errors = schema.validate()
    if errors:
        raise ValueError(f"Schema validation failed: {', '.join(errors)}")

    # Serialize to dict
    schema_dict = schema.to_dict()

    # Convert to JSON with pretty printing and sorted keys for determinism
    json_str = json.dumps(schema_dict, sort_keys=True, indent=2, ensure_ascii=False)
    json_bytes = json_str.encode("utf-8")

    # Optionally write to output path
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(json_bytes)
        logger.info(f"Exported universal schema JSON to {output_path}")

    return json_bytes


__all__ = ["export_universal_schema_json"]
