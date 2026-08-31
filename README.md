# Cyberwave Robot Format

Universal robot description schema and format converters for Cyberwave.

## Overview

This package provides:

- **Universal Schema**: A canonical representation for robotic assets (`CommonSchema`)
- **Format Importers**: Parse URDF, MJCF, USD into the universal schema
- **Format Exporters**: Export universal schema to URDF, MJCF, USD
- **Validation**: Schema validation and consistency checks

## Structure

```
cyberwave_robot_format/
├── schema.py           # Core schema definitions (CommonSchema, Link, Joint, etc.)
├── core.py             # Base classes for parsers/exporters
├── urdf/               # URDF parser and exporter
├── mjcf/               # MJCF (MuJoCo) parser and exporter
├── usd/                # USD (OpenUSD) parser and exporter
├── mesh/               # Mesh processing utilities
├── math_utils.py       # Math utilities (Vector3, Quaternion, etc.)
└── utils.py            # General utilities
```

## Schema changes in 0.1.6

Two fixes to `CommonSchema` itself. Both are backwards compatible to *read* — an
older JSON document still parses — but they change what the package *writes*, so any
hash computed over `export_universal_schema_json` output will differ from 0.1.5.

- **`Material` gained an `extensions` field**, matching every other component
  dataclass. `MJCFParser` already wrote `extensions["reflectance"]` there, so before
  this any MuJoCo model declaring a material with `reflectance` failed to parse with
  an `AttributeError`. Serialized materials now carry an `"extensions": {}` key.
- **`Vector3` and `Quaternion` coerce their components to `float`.** The fields were
  always declared `float`, but `Vector3(0, 0, 1)` — written in `URDFParser` and in
  `Joint.axis`'s own default — left `int` components behind, so the same model
  serialized an axis as `"z": 1` fresh from the parser and `"z": 1.0` after any
  `from_dict`. `export_universal_schema_json` is now idempotent under a round trip.
  As a side effect, `numpy` scalars are normalized too, which `json.dumps` cannot
  serialize at all.

## Installation

```bash
pip install cyberwave-robot-format

# USD support needs the OpenUSD Python bindings, which are an optional extra:
pip install "cyberwave-robot-format[usd]"
```

The rest of the package works without them — importing `cyberwave_robot_format`
never imports `pxr`, so only calling the USD parser or exporter requires it.
`usd-core` publishes no `linux-aarch64` wheel; on that platform install
conda-forge's `openusd` instead.

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

**Materials** are emitted two ways, mirroring how `MJCFParser` reads them. A
material with a `name` is defined once under `<asset>` and referenced by every
geom using it; a nameless one — which is what a geom carrying a bare `rgba`
attribute parses into — cannot be shared, so its colour is written back onto the
geom's own `rgba`. Either way the colour survives a MJCF → schema → MJCF round
trip. A colour carrying no alpha (which is what a `UsdPreviewSurface` without an
`opacity` input yields) is exported as opaque, since MuJoCo will not compile a
model whose `rgba` is short of four values. Note that a geom `rgba` carries only
the colour: an *unnamed* material's texture and specular cannot be expressed,
as MuJoCo has no inline material.

### Parse and export USD (OpenUSD)

Requires the `usd` extra (see [Installation](#installation)). Reads and writes
`.usda` (text), `.usdc` (binary), `.usd` and `.usdz`.

```python
from cyberwave_robot_format import USDExporter, USDParser

# Schema -> USD. The suffix picks the encoding; .usda is the human-readable one.
USDExporter().export(schema, "output/robot.usda")

# USD -> schema
schema = USDParser().parse("output/robot.usda")

# Or work with the text directly, no files involved
usda_text = USDExporter().export_to_string(schema)
schema = USDParser().parse_string(usda_text)
```

The export is a complete `UsdPhysics` articulation — rigid bodies with mass and
inertia, joints with limits and drives, visual and collision geometry, materials,
and collision filtering — so Isaac Sim, Omniverse and `usdview` can consume it
directly. Alongside each native attribute the exporter also writes a
`cyberwave:`-namespaced double-precision copy for anything USD stores lossily or
cannot express at all (mimic joints, armature, jerk limits, motor electricals), and
the parser prefers those. The round trip is therefore lossless for every field the
schema can hold, while the stage stays valid USD for everyone else.

Reading *foreign* USD works too: with no `cyberwave:` attributes present, links come
from `PhysicsRigidBodyAPI`, the kinematic tree from `physics:body0`/`body1`, joint
axes from `physics:axis` plus the joint frame rotations, and a
`UsdPhysicsDriveAPI` becomes an actuator with its gains — so an Isaac Sim robot
imports as an actuated model rather than a passive one.

Because both halves of the schema survive, a USD detour is transparent to the other
converters — `URDF -> schema -> MJCF` and `URDF -> schema -> USD -> schema -> MJCF`
produce byte-identical MJCF:

```python
# The format-specific payloads MJCFParser stashes in `extensions` (MuJoCo's
# contype/conaffinity/solref/margin, its <size> and <default> blocks, <contact><pair>
# attributes) are carried through USD untouched, so nothing is lost on the way.
schema = MJCFParser().parse("robot.xml")
MJCFExporter().export(USDParser().parse_string(USDExporter().export_to_string(schema)),
                      "same_robot.xml")
```

Conversely, UsdPhysics properties with no schema field of their own —
`physics:breakForce`, `jointEnabled`, `kinematicEnabled`, `velocity`, a spherical
joint's cone limits, a drive's `targetVelocity` — are preserved under
`extensions["usd"]` on import and written back as native `physics:` attributes on
export, so a foreign stage survives a schema hop too.

#### Parsing USD you did not author

Opening a USD file **composes** it: `subLayers`, `references`, `payloads` and
variants are resolved, and their content becomes part of the schema you get back.
A hostile file can name an absolute path and pull that file's prims into the
result:

```
#usda 1.0
(
    subLayers = [@/home/someone/private/robot.usda@]
)
```

A service that parses an upload and returns or stores the resulting schema would
be handing back the content of local USD files it never meant to expose. For
untrusted input, restrict composition to the file's own directory:

```python
# Raises ValueError if composition reaches outside the parsed file's directory.
schema = USDParser(allow_external_layers=False).parse(uploaded_path)
```

Layers beside or beneath the input still resolve normally, so ordinary multi-file
assets keep working; symlinks are resolved before the check, so they cannot step
outside. The default is `True` because layering is the defining feature of USD and
real assets reference sibling directories — the restriction is opt-in precisely
because it is the caller who knows whether the input is trusted.

Two things you do not need to defend against (verified against `usd-core` 26.8):
the default asset resolver does not fetch `http(s)://` references, so there is no
SSRF vector, and `.usdz` is read in place rather than extracted. Note also that
`USDExporter(bake_meshes=True)` reads whatever path `Geometry.filename` holds — do
not enable it for schemas from an untrusted source.

Things worth knowing:

- **Angles.** USD stores angular joint limits and drive targets in degrees; the
  schema uses radians. The conversion is automatic, and the `cyberwave:` sidecars
  are always in schema units.
- **Units.** The stage is always authored `metersPerUnit = 1`, `kilogramsPerUnit = 1`
  and Z-up, because that is what every number in the schema means. A schema
  declaring `metadata.units` as anything but `"SI"` is written unconverted, with a
  warning.
- **Link layout.** `UsdPhysics` ignores a rigid body nested under another one, so
  link prims are flat siblings under `/<Robot>/Links`, each carrying its accumulated
  world transform at the zero configuration.
- **Joint axes.** `physics:axis` only accepts `"X"`/`"Y"`/`"Z"`, so an arbitrary
  schema axis is baked into `physics:localRot0`/`localRot1` and the token is always
  `"X"`.
- **Meshes.** `Geometry.filename` is preserved verbatim as a string; USD cannot
  reference an `.stl`/`.dae`/`.obj` as a layer. Pass
  `USDExporter(bake_meshes=True)` to additionally resolve and inline the mesh
  points. A `.usdz` whose meshes are unresolvable `package://` URIs is still
  written, with a warning that those files are not bundled.
- **World physics.** `Physics` maps to a `UsdPhysicsScene` authored *beside* the
  robot prim, so referencing the asset into a larger stage does not drag a second
  gravity definition along. On import the scene is located by prim type rather than
  by path, so a foreign stage that names it anything else (`/physicsScene`,
  `/World/PhysicsScene`) still contributes its gravity and solver settings.
- **One articulation per parse.** A stage declaring several
  `PhysicsArticulationRootAPI` prims (a work cell, a robot plus an AMR) parses as
  the first one, with a warning naming the roots that were skipped.
- **Materials.** A named material is defined once under `/<Robot>/Materials` and
  shared by every visual using it. If two *different* materials share a name, the
  first owns the shared prim and the others are authored inline per visual, with a
  warning — so both appearances survive the round trip.

Since v0.1.6.

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
    strict_missing_meshes=True,  # Fail fast on missing meshes
    compresslevel=1,             # Default: favours write speed over size
)

urdf_zip = export_urdf_zip_cloud(schema, s3_resolver)  # always deflate level 6
```

MuJoCo scene ZIPs are written at deflate level 1 by default, which is 2-5x
faster than zlib's default for 7-24% more bytes — the right trade when a scene is
built per request and streamed to a simulator. Pass a higher `compresslevel` to
`export_mujoco_zip_cloud` when you are archiving the ZIP and the size matters
more than the write. `export_urdf_zip_cloud` takes no such argument and always
writes at zlib's default.

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
