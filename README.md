# Cyberwave Robot Format

Universal robot description schema and format converters for Cyberwave.

## Overview

This package provides:
- **Universal Schema**: A canonical representation for robotic assets (`CommonSchema`)
- **Format Importers**: Parse URDF, MJCF into the universal schema
- **Format Exporters**: Export universal schema to URDF, MJCF
- **Validation**: Schema validation and consistency checks

## Structure

```
cyberwave_robot_format/
├── schema.py           # Core schema definitions (CommonSchema, Link, Joint, etc.)
├── core.py             # Base classes for parsers/exporters
├── urdf/               # URDF parser and exporter
├── mjcf/               # MJCF (MuJoCo) parser and exporter
├── mesh/               # Mesh processing utilities
├── math_utils.py       # Math utilities (Vector3, Quaternion, etc.)
└── utils.py            # General utilities
```

## Usage

### Parse URDF

```python
from cyberwave_robot_format import CommonSchema
from cyberwave_robot_format.urdf import URDFParser

# Parse a URDF file
parser = URDFParser()
schema = parser.parse("path/to/robot.urdf")

# Validate the schema
issues = schema.validate()
if issues:
    print("Validation issues:", issues)

# Access robot components
for link in schema.links:
    print(f"Link: {link.name}, mass: {link.mass}")

for joint in schema.joints:
    print(f"Joint: {joint.name}, type: {joint.type}")
```

### Parse MJCF (MuJoCo)

```python
from cyberwave_robot_format.mjcf import MJCFParser

# Parse a MuJoCo XML file
parser = MJCFParser()
schema = parser.parse("path/to/robot.xml")

# Access actuators
for actuator in schema.actuators:
    print(f"Actuator: {actuator.name}, joint: {actuator.joint}")
```

### Export to MJCF

```python
from cyberwave_robot_format.mjcf import MJCFExporter

# Export schema to MuJoCo format
exporter = MJCFExporter()
exporter.export(schema, "output/robot.xml")
```

### Cloud-Native Scene Export

Export complete scenes with meshes to ZIP files, supporting cloud storage and in-memory conversion:

```python
from cyberwave_robot_format.mjcf import export_mujoco_zip_cloud
from cyberwave_robot_format.urdf import export_urdf_zip_cloud

# Cloud-safe resolver with in-memory DAE→OBJ conversion
def s3_resolver(filename: str) -> tuple[str, bytes] | None:
    """Download from S3 and convert in memory."""
    mesh_bytes = s3.get_object(Bucket='meshes', Key=filename)['Body'].read()
    
    if filename.endswith('.dae'):
        obj_bytes = convert_dae_to_obj_in_memory(mesh_bytes)
        return (filename.replace('.dae', '.obj'), obj_bytes)
    
    return (Path(filename).name, mesh_bytes)

# Export with cloud resolver (mesh_resolver is required)
mujoco_zip = export_mujoco_zip_cloud(
    schema,
    s3_resolver,
    strict_missing_meshes=True  # Fail fast on missing meshes
)

urdf_zip = export_urdf_zip_cloud(schema, s3_resolver)
```

## Development

Install in editable mode:
```bash
pip install -e .
```

Run tests:
```bash
pytest
```
