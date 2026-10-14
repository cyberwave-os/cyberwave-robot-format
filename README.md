# Cyberwave Robot Format

Universal robot description schema and format converters for Cyberwave.

## Overview

This package provides:

- **Universal Schema**: A canonical representation for robotic assets (`CommonSchema`)
- **Format Importers**: Parse URDF, MJCF, USD into the universal schema
- **Format Exporters**: Export universal schema to URDF, MJCF, USD
- **Validation**: Schema validation and consistency checks

MJCF export omits all-zero visual boxes used on frame-only links, retaining the
link, joints and collision shapes. Invalid collision boxes and partially zero
or negative visual dimensions still fail MuJoCo compilation; they are not removed.

## Structure

```
cyberwave_robot_format/
├── schema.py           # Core schema definitions (CommonSchema, Link, Joint, etc.)
├── core.py             # Base classes for parsers/exporters
├── urdf/               # URDF parser and exporter
├── mjcf/               # MJCF (MuJoCo) parser and exporter
├── usd/                # USD (OpenUSD) parser and exporter, scene -> GLB converter
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
`usd-core` publishes no `linux-aarch64` wheel (e.g. Docker on Apple Silicon), so
on that platform the `[usd]` extra installs NVIDIA's `usd-exchange`, which ships
the same `pxr` modules.

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

Common-schema JSON reads also accept the older catalog actuator name `motor`.
It becomes `dc_motor`, with `limits.effort` and `limits.velocity` mapped to
`max_torque` and `max_velocity` when those canonical fields are absent.
Explicit canonical fields, including zero or null, take precedence. Legacy
limits must be finite, non-negative numbers. Descriptive gearbox ratios in
`extensions` remain metadata; they do not change the output torque convention.
This compatibility read does not rewrite saved assets or enable hardware control.

```python
from cyberwave_robot_format.mjcf import MJCFParser

# Parse a MuJoCo XML file
parser = MJCFParser()
schema = parser.parse("path/to/robot.xml")

# Access actuators
for actuator in schema.actuators:
    print(f"Actuator: {actuator.name}, joint: {actuator.joint}, site: {actuator.get_mjcf_site_transmission()}")
```

Named MJCF sites and stateless site motors survive the common-schema JSON
round trip and scene composition. `Link.extensions.mjcf_sites` retains the
resolved frame attributes; `Actuator.extensions.mjcf_site_transmission` retains
`site`, optional `refsite`, the full signed gear vector and limit flags.
These actuators have an empty `joint`, not a fictitious joint, and no scalar
`gear_ratio`. Site-bound IMU/frame sensors keep the same frame and source order.
Scene composition namespaces all these references together.

This is model preservation, **not** a declaration of hardware actuator access.
A flight/locomotion skill may command only a higher-level controller while its
simulation model contains motors internally. Runtime adapters must separately
validate the policy's actual output interface. URDF cannot represent these native
transmissions; retain the common-schema artifact instead of round-tripping through
URDF. Current site import supports quaternion/vector orientations and inline
colors; canonicalize Euler/axis-angle/material-based sites first. Unsupported
site servos, malformed extensions and missing transmission frames fail explicitly.

Body adhesion actuators use `type: "adhesion"`, an empty `joint`, a nonnegative
`control_range`, and `extensions.mjcf_adhesion: {"body": "suction_cup", "gain": "20"}`.
The gain is in newtons per control unit; this example is not a hardware rating.
Optional force limits use `force_range`; the native extension also accepts
`group` and `forcelimited`. Import resolves nested adhesion defaults, storage
retains the binding, and scene composition prefixes the body and actuator together.
Missing bodies, malformed limits and unsupported MJCF attributes fail explicitly.
Adhesion applies contact-normal force; it does not model a vacuum seal, imply
successful grasping, or configure a controller command. Keep the common schema
when moving between formats: URDF does not represent this native actuator.
See [MuJoCo adhesion](https://mujoco.readthedocs.io/en/stable/XMLreference.html#actuator-adhesion).

MJCF imports retain nested joint/geometry defaults and body `childclass`
overrides, including inherited limits, collision settings and visual shapes.
Floating roots declared with `<freejoint/>` stay floating without inheriting
joint damping. Mesh names may be inferred from filenames in the compiler's mesh
directory. Inertial-frame and geometry quaternions are retained.
Fresh MJCF imports also retain scalar simulation options, including contact
cone and impedance ratio, and use MuJoCo defaults for omitted options.
This does not rewrite previously imported assets or their saved settings.
Before training on a converted model, compare its compiled dynamics and solver
settings with the intended model; successful parsing alone does not guarantee
simulation equivalence.

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

- **Exact local poses.** Rotated visuals, collisions and sensors carry their
  original position and quaternion in `cyberwave:pose:*`, alongside the native
  matrix. This prevents small numeric changes on repeated export/import. As with
  joint sidecars, these values take precedence when importing; external editors
  changing the matrix should update or remove the sidecars as well. Older and
  foreign stages without them retain native transform decomposition.
- **Embedded JSON.** Metadata is limited to 256 nested arrays/objects, independently
  of Python's recursion behavior. Import logs and ignores only the offending
  attribute or `customData` value; export rejects excessive depth rather than
  emitting data its reader would discard. This does not limit scene hierarchy or
  mesh size, and brackets inside strings do not count as nesting.
- **Angles.** USD stores angular joint limits and drive targets in degrees; the
  schema uses radians. Angular drive gains follow: `UsdPhysicsDriveAPI` stiffness
  is per degree and damping per degree/s, so an actuator's `kp`/`kd` and a joint's
  `spring_stiffness`/`damping` are written as `value * pi / 180` and read back
  (including from foreign stages) as `value * 180 / pi`. Linear joints are SI
  both ways. The conversion is automatic, and the `cyberwave:` sidecars are always
  in schema units.
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

#### Scene bundles

`export_scene_usd` places one or more robot instances in a single scene and writes a
directory of USD layers: `scene.usda` (default prim `/World`) plus one
`assets/<asset>_<sha12>.usda` per distinct robot, shared by every instance of it.
It is the USD counterpart of `CommonSchema.merge_in` followed by the MJCF scene
export. The output is byte-for-byte deterministic for the same input and package
version.

```python
from cyberwave_robot_format import Pose
from cyberwave_robot_format.usd import (
    SceneExportOptions,
    SceneInstance,
    export_scene_usd,
)

result = export_scene_usd(
    [
        SceneInstance(instance_id="arm-1", schema=arm, fixed_base=True),
        SceneInstance(instance_id="amr-1", schema=amr, spawn_pose=Pose()),
    ],
    "output/scene",
    options=SceneExportOptions(bake_meshes=True),
)
assert result.ok, result.compatibility["errors"]
```

- **`SceneInstance`** is one robot: `instance_id`, `schema`, `spawn_pose`,
  `fixed_base` (`None` infers it from the schema's world joint), optional
  `sensor_ids`, `first_person_camera`, `namespace`, docking via
  `attach_to_instance`/`attach_to_link`, and `custom_data` for the caller's own
  identity keys. A bare `CommonSchema` is accepted as a single instance.
- **`SceneExportOptions`** holds the scene-level settings: the scene prim name and
  `scene_custom_data`, `physics` (defaults to the first instance's), generated
  lights, `ground_plane` (added unless an instance already has a plane),
  `viewport` (a **`ViewportOptions`** for the free `sim_viewport` camera),
  `static_fixtures` (**`StaticBoxFixture`** boxes welded to the world) and
  `bake_meshes`/`mesh_search_paths`. Without baking, the bundle is reported
  incomplete (`mesh_not_embedded`).
- **`UsdExportResult`** lists the written `files`, the `identity_map` from
  instances, links, joints and sensors to prim paths, `sensors`,
  `articulations` (drive gains in schema units, per radian), `compatibility`
  findings (`result.ok` is `True` when there are no errors), `units`,
  `exporter` and `physics_scene`. `to_dict()` gives the JSON form.
- **Identity on the stage.** `/World` carries `cyberwave:usdSchemaVersion`, the
  version of the portable prim layout and identity keys (`USD_SCHEMA_VERSION` in
  `usd/tokens.py`, currently `1.1`). Instance prims carry `cyberwave:base`
  (`"fixed"` or `"floating"`). Links, joints, drives and sensors carry
  `cyberwave:linkName`, `jointName`, `dofIndex`, `controlMode`, `actuatorName`,
  `sensorName`, `sensorId`, `sensorKind`, `sensorParams` and `cameraRole`.
  Synthesized prims (world anchor, lights, viewport, ground) carry
  `cyberwave:generated`. These are plain customData values, readable with
  `GetCustomDataByKey`.
- **PhysX passthrough.** `extensions["physx"]`, keyed by full attribute name
  (`{"physxRigidBody:linearDamping": 0.1}`), is authored as the matching PhysX
  applied API on the schema root (articulation), links, collisions, joints and
  `Physics` (scene). Velocity limits, armature and mimic joints are also derived
  from the schema in PhysX/Newton form. An unknown or mistyped entry is skipped
  and reported as `physx_attribute_skipped`, never a failed export. Scene
  attributes the Isaac runtime owns (broadphase, GPU dynamics, step rate) are not
  accepted.

### Convert a USD scene to a GLB

`USDParser` reads USD as a *robot* and never looks at mesh vertices. For a static
**scene** — a room, a set of props, an Omniverse export — `USDSceneToGLB` bakes the
stage into one self-contained binary glTF: transforms baked into the vertices,
materials translated to glTF PBR, textures embedded.

Needs both `usd-core` and Pillow, which the `usd` extra installs.

```python
from cyberwave_robot_format.usd import USDSceneToGLB

result = USDSceneToGLB(max_texture_size=2048).convert("room.usda")
open("room.glb", "wb").write(result.glb)

print(result.report.as_dict())  # what was translated, substituted or skipped
```

- **Z-up by default.** Cyberwave is Z-up and does not re-orient a GLB you upload
  yourself (it only does so for its own generated meshes), so a standard Y-up glTF
  would stand on its end. Pass `z_up=False` for glTF-standard Y-up. Units come from
  the stage's `metersPerUnit`; the output is always metres.
- **Materials.** `UsdPreviewSurface`, glTF-style MDL (`gltf/pbr.mdl`), OmniPBR-style
  inputs, and file-based `.mdl` (its constants and texture paths are read as text —
  MDL is never executed). A material that cannot be resolved, typically an MDL
  library that is not on disk, becomes neutral grey and is listed in
  `report.substituted_materials`; `material_overrides={"Carpet": {"base_color":
  [0.3, 0.33, 0.2], "roughness": 0.95}}` replaces it. Untextured black
  `chrome`/`steel`/`copper`/`brass`/`plastic_black` materials — which FBX exports
  leave to the renderer's defaults — get metal PBR values by name
  (`apply_name_presets=False` turns that off).
- **Hidden prims are included.** Authors often hide geometry to get it out of a
  debug camera's way. Pass `include_invisible=False` to honour `visibility`.
  Guide and proxy geometry is never included.
- **Node names** follow the prim path (`World.left.gauge.needle`), so a part such as
  a needle stays addressable in the GLB.
- **`.usdz` packages** work as given: textures stored inside the package are read
  back through USD's resolver.
- **Triangle budget.** `max_triangles` (default 5,000,000) refuses a larger scene with
  a `ValueError` naming the limit, before any per-corner array is built. Peak memory
  is roughly 0.75 GB per million triangles, and a 48 MB `.usdc` can hold eight million
  of them, so a service converting uploads should keep this modest.
- **Not carried over:** lights, cameras and animation.

Unlike `USDParser`, `allow_external_layers` defaults to **`False`**: the typical
caller is an upload endpoint. Composition and every texture and `.mdl` the stage
names must resolve inside the input's directory, or inside `root_dir` if you pass
one (`convert("extracted/models/scene.usd", root_dir="extracted")`; use it for a
scene unpacked from a ZIP, where the scene file often sits in a subfolder beside a
sibling `textures/` folder), otherwise a stage naming
`/etc/some-image.png` would have that file embedded in the GLB it gets back. Texture
references that are missing or refused are listed in `report.missing_textures`. A
texture that is not where it is referenced but has a same-named file elsewhere in
the input directory is used anyway, with a `report.warnings` entry (noting when
there were several candidates) so a wrong match is visible.

### Export positional loop closures to MuJoCo

For a closed-chain robot, declare link-frame pairs in
`schema.extensions["loop_closures"]`, for example
`[{"frame_a": "rod_tip", "frame_b": "platform_tip"}]`.
MuJoCo connects the two frame origins even if they differ in the zero pose.
Only full `xyz` position constraints are currently supported by this exporter;
other axis selections fail explicitly. Composition prefixes both frame names.
Mark unpowered joints with `joint.extensions["passive"] = True` to exclude them
from fallback motors. Author active actuators and physical properties separately;
successful export does not establish stable dynamics or collision-free motion.

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

## Preparing concave collision meshes

MuJoCo uses a convex hull for each mesh collider. A concave gripper, bracket or
container can therefore have an invisible solid region despite a correct visual
mesh. Prepare collision pieces explicitly before publishing an asset:

```python
# pip install 'cyberwave-robot-format[collision]'
import trimesh
from cyberwave_robot_format.mesh import decompose_collision_mesh

mesh = trimesh.load("collision.obj", force="mesh")
# Apply the authored mesh scale first: input coordinates must be metres.
parts = decompose_collision_mesh(mesh, threshold_m=0.001, max_hulls=32)
for index, part in enumerate(parts):
    part.export(f"collision-{index}.obj")
```

Use one existing `Collision` entry per piece, retaining the source collider's
pose, group and contact properties. The output is already scaled; do not apply
the original scale twice. Preserve the visual mesh and authored link inertia.
Store source hashes and preparation settings with the asset artifacts. Review
clearances, hull count and contacts in simulation: the hull-count limit can relax
the requested approximation tolerance. This utility does not run automatically
on export or motion and does not authorize hardware control.
