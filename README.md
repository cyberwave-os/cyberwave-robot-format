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

#### Tolerated URDF quirks

Real-world URDFs are often slightly out of spec. The parser recovers from these
rather than failing, and records each one under
`schema.extensions["parse_context"]` — in `warnings`, or in `errors` where the
recovery had to discard something the file asked for:

| In the file | What you get |
| ----------- | ------------ |
| `<robot>` with no `name`, or a blank/whitespace one | The name falls back to the URDF file stem |
| `<joint>` with no `type` | The joint is kept as `fixed`, so its child link stays attached to the tree |
| `<joint>` with no `type`, but with `<axis>`, `<limit>`, `<mimic>` or `<safety_controller>` | Also kept as `fixed`, but recorded in `errors` — the file described motion that `fixed` discards, and the spec cannot say which moving type was meant. `<dynamics>` is not treated as motion evidence, because real files carry it on genuinely fixed joints |
| `type="Revolute"`, `type=" revolute "` | Normalized to `revolute` |

An unrecognized joint type (say `type="screw"`) is still an error and the joint
is dropped — recovery covers omissions and formatting, not unsupported
kinematics. Check `parse_context["errors"]` alongside `warnings` when a
conversion looks wrong.

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

**Continuous joints** (a `continuous` joint type, e.g. a wheel or a spinner)
export as an honest `limited="false"` free hinge with no positional `range` —
even if the schema happens to carry limits. Because MuJoCo position actuators
still need a finite `ctrlrange`, the actuator on a continuous joint is given a
finite band centered on the home pose (`home ± π`) instead, so the joint stays
truly unlimited while the servo remains usable. Since v0.1.4.

### Infer URDF mimic joints (gripper coupling)

When a URDF has coupled finger / gripper joints but no `<mimic>` tags, infer pairs from
kinematics and write a new `{stem}-mimic-joint.urdf` (the original file is never modified).

```python
from pathlib import Path
from cyberwave_robot_format.urdf import (
    infer_mimic_joints,
    infer_and_patch_if_needed,
    write_mimic_patched_urdf,
)

result = infer_mimic_joints("robot.urdf")
for m in result.inferred_mimics:
    print(m.driver_joint, "→", m.slave_joint, "mult", m.multiplier, "conf", m.confidence)

# Write patched URDF when confidence ≥ 0.85 (default)
patched = infer_and_patch_if_needed(Path("robot.urdf"))
if patched.output_path:
    print("Wrote", patched.output_path)
```

**Multiplier defaults**

| Context | Default |
|---------|---------|
| URDF `<mimic>` if `multiplier` omitted | `1` |
| `offset` omitted | `0` |
| Inference for opposing prismatic jaws | Often `-1` when complementary limits validate |

Inference checks opposing axes, complementary joint limits, and samples driver positions so
`slave = multiplier × driver + offset` stays within slave limits. Pass an optional `mjcf_path`
to seed coeffs from MuJoCo equality constraints.

Used by Cyberwave backend `seed_controllers --infer-mimic-from-urdf` and
`src/lib/urdf_mimic_utils.py`. See `cyberwave-backend/docs/mimic-joints.md` for the full
platform workflow (autogen, teleop, MQTT).

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

## Acknowledgments

This project incorporates portions of code from
[https://github.com/thanhndv212/robot_format_converter](Robot Format Converter) (Apache 2.0 licensed).

Original repository:
https://github.com/thanhndv212/robot_format_converter

We thank the original authors for their initial work.

```bibtex
@software{robot_format_converter,
author = {Nguyen, Thanh},
title = {Robot Format Converter: Universal Robot Description Format Converter},
year = {2025},
url = {https://github.com/thanhndv212/robot_format_converter},
version = {1.0.0}
}
```
