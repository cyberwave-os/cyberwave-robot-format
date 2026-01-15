"""Tests for universal schema JSON export."""

import json
import tempfile
from pathlib import Path

import pytest

from cyberwave_robot_format import (
    CommonSchema,
    Link,
    Metadata,
    export_universal_schema_json,
)


def test_export_universal_schema_json_basic():
    """Test basic JSON export functionality."""
    # Create a simple schema
    schema = CommonSchema(
        metadata=Metadata(name="test_schema", version="1.0"),
        links=[Link(name="base_link", mass=1.0)],
    )
    
    # Export to bytes
    json_bytes = export_universal_schema_json(schema)
    
    # Verify it's valid JSON
    json_data = json.loads(json_bytes.decode('utf-8'))
    
    assert "metadata" in json_data
    assert json_data["metadata"]["name"] == "test_schema"
    assert "links" in json_data
    assert len(json_data["links"]) == 1
    assert json_data["links"][0]["name"] == "base_link"


def test_export_universal_schema_json_with_file():
    """Test JSON export with file output."""
    schema = CommonSchema(
        metadata=Metadata(name="test_schema"),
        links=[Link(name="link1", mass=2.0)],
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_schema.json"
        
        # Export to file
        json_bytes = export_universal_schema_json(schema, output_path)
        
        # Verify file was created
        assert output_path.exists()
        
        # Verify file contents match returned bytes
        with open(output_path, 'rb') as f:
            file_contents = f.read()
        assert file_contents == json_bytes


def test_export_universal_schema_json_roundtrip():
    """Test that exported JSON can be re-imported."""
    # Create a schema with various components
    schema = CommonSchema(
        metadata=Metadata(name="roundtrip_test", version="2.0"),
        links=[
            Link(name="link1", mass=1.0),
            Link(name="link2", mass=2.0),
        ],
    )
    
    # Export to JSON
    json_bytes = export_universal_schema_json(schema)
    
    # Parse JSON
    json_data = json.loads(json_bytes.decode('utf-8'))
    
    # Re-import via from_dict
    imported_schema = CommonSchema.from_dict(json_data)
    
    # Verify metadata
    assert imported_schema.metadata.name == "roundtrip_test"
    assert imported_schema.metadata.version == "2.0"
    
    # Verify links
    assert len(imported_schema.links) == 2
    assert imported_schema.links[0].name == "link1"
    assert imported_schema.links[1].name == "link2"


def test_export_universal_schema_json_validation_error():
    """Test that invalid schemas raise validation errors."""
    # Create an invalid schema (empty metadata name)
    schema = CommonSchema(
        metadata=Metadata(name=""),  # Invalid: empty name
        links=[],
    )
    
    # Should raise ValueError due to validation failure
    with pytest.raises(ValueError, match="Schema validation failed"):
        export_universal_schema_json(schema)


def test_export_universal_schema_json_formatting():
    """Test that JSON is properly formatted."""
    schema = CommonSchema(
        metadata=Metadata(name="format_test"),
        links=[Link(name="link1", mass=1.0)],
    )
    
    json_bytes = export_universal_schema_json(schema)
    json_str = json_bytes.decode('utf-8')
    
    # Verify it's indented (pretty-printed)
    assert '\n' in json_str
    assert '  ' in json_str  # 2-space indentation
    
    # Verify it's valid JSON
    json_data = json.loads(json_str)
    assert json_data is not None
