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

## Development

Install in editable mode:
```bash
pip install -e .
```

Run tests:
```bash
pytest
```
