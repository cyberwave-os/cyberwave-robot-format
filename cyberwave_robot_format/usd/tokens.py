# Copyright [2025] Cyberwave
# Licensed under the Apache License, Version 2.0

"""Constant tables shared by :mod:`usd.exporter` and :mod:`usd.parser`.

This module is deliberately free of any ``pxr`` import so that the attribute
spellings and type mappings live in exactly one place that can be imported (and
tested) without ``usd-core`` installed. Every ``cyberwave:``-namespaced
attribute name used on either side of the round trip is defined here — if the
exporter and the parser ever disagree about a spelling, it is because one of
them bypassed this module.

Token values (``"X"``, ``"guide"``, ``"force"``, ``"convexHull"``, ...) are
written as plain strings rather than referencing ``UsdGeom.Tokens`` /
``UsdPhysics.Tokens`` for the same reason. They are verified against the
installed schema by ``tests/test_usd_tokens.py``.
"""

from __future__ import annotations

from cyberwave_robot_format.schema import ActuatorType, GeometryType, JointType

# --------------------------------------------------------------------------
# Namespaces and prim/scope names
# --------------------------------------------------------------------------

CW = "cyberwave"
"""Root namespace for every Cyberwave-specific attribute and customData key."""

SCOPE_LINKS = "Links"
SCOPE_JOINTS = "Joints"
SCOPE_VISUALS = "Visuals"
SCOPE_COLLISIONS = "Collisions"
SCOPE_SENSORS = "Sensors"
SCOPE_MATERIALS = "Materials"
SCOPE_PHYSICS_MATERIALS = "PhysicsMaterials"
SCOPE_ACTUATORS = "Actuators"
SCOPE_CONTACTS = "Contacts"
SCOPE_COLLISION_GROUPS = "CollisionGroups"
SCOPE_COLLISION_CONFIG = "CollisionConfig"
SCOPE_PAIRS = "Pairs"
SCOPE_RENDER = "Render"
SCOPE_ROTORS = "Rotors"

PHYSICS_SCENE_PATH = "/PhysicsScene"
"""``UsdPhysicsScene`` lives *beside* the robot prim, not under it.

``Physics`` is world-level in ``CommonSchema``; keeping it out of the
``defaultPrim`` subtree means referencing the exported asset into a larger stage
does not drag a second gravity definition along with it.
"""

INLINE_MATERIAL_NAME = "Material"
"""Prim name used for an unnamed ``Material`` authored inline under its gprim."""

SHADER_NAME = "Shader"
TEXTURE_SHADER_NAME = "Texture"
UV_READER_NAME = "stReader"

PHYSICS_MATERIAL_PREFIX = "physmat_"
PAIR_PREFIX = "pair_"
UNNAMED_VISUAL_PREFIX = "visual_"
UNNAMED_COLLISION_PREFIX = "collision_"

# --------------------------------------------------------------------------
# customData keys (dotted with ':' — USD treats that as a dictionary path, so
# "cyberwave:extensions" nests as {"cyberwave": {"extensions": ...}})
# --------------------------------------------------------------------------

CD_EXTENSIONS = f"{CW}:extensions"
CD_GEOMETRY_EXTENSIONS = f"{CW}:geometryExtensions"
"""``Geometry.extensions``, kept separate from ``CD_EXTENSIONS``.

A ``Visual``/``Collision`` and its ``Geometry`` share a single gprim, and both
dataclasses have their own ``extensions`` dict — so they need distinct customData
keys or the second write silently clobbers the first.
"""
CD_METADATA = f"{CW}:metadata"
CD_ORDER = f"{CW}:order"
CD_SOLVER_EXTENSIONS = f"{CW}:solverExtensions"
CD_EXCLUDES = f"{CW}:excludes"

# Stable identity keys: the portable USD schema a runtime (Isaac Sim, usdview, a
# test) reads to map prims back to schema names and ids without parsing prim
# names. Unlike the JSON payloads above these are plain USD values (string, int,
# dictionary of scalars), so a consumer reads them with GetCustomDataByKey alone.
# The parser never reads them; the sidecars above stay authoritative for the
# CommonSchema round trip. A caller's own identity keys (whatever its scene
# instances stand for) are not defined here: they arrive as opaque
# ``custom_data`` on the scene and instance prims (see :mod:`.scene`).
CD_BASE = f"{CW}:base"
"""``"fixed"`` or ``"floating"`` on a scene instance prim."""
CD_LINK_NAME = f"{CW}:linkName"
CD_JOINT_NAME = f"{CW}:jointName"
CD_DOF_INDEX = f"{CW}:dofIndex"
"""Index in the robot's canonical joint vector (schema order of 1-DOF joints)."""
CD_CONTROL_MODE = f"{CW}:controlMode"
"""``"position"``, ``"velocity"`` or ``"effort"`` for a driven joint."""
CD_ACTUATOR_NAME = f"{CW}:actuatorName"
CD_SENSOR_NAME = f"{CW}:sensorName"
CD_SENSOR_ID = f"{CW}:sensorId"
CD_SENSOR_KIND = f"{CW}:sensorKind"
CD_SENSOR_PARAMS = f"{CW}:sensorParams"
"""Normalized scalar parameters (SI, radians) as a USD dictionary."""
CD_CAMERA_ROLE = f"{CW}:cameraRole"
CD_ROTOR = f"{CW}:rotor"
"""Rotor/thruster parameters as a USD dictionary (see ``semantics.RotorSpec``)."""
CD_LIGHT_ROLE = f"{CW}:lightRole"
CD_GENERATED = f"{CW}:generated"
"""``True`` on prims the scene exporter synthesized (world anchor, ...)."""
CD_USD_SCHEMA_VERSION = f"{CW}:usdSchemaVersion"
"""On ``/World`` of a scene stage: the portable schema version below."""

USD_SCHEMA_VERSION = "1.1"
"""Version of the portable Cyberwave USD schema (prim layout + identity keys).

Bump the minor version for additive keys, the major version for anything a
runtime reading an older bundle would misinterpret.
"""

# --------------------------------------------------------------------------
# Attribute names
# --------------------------------------------------------------------------

A_NAME = f"{CW}:name"
"""Original schema name, authored only when sanitization changed it."""

# Local visual/collision/sensor pose. Native matrices remain available to other
# USD consumers; these fields preserve the source quaternion exactly on import.
A_POSE_POSITION = f"{CW}:pose:position"
A_POSE_ORIENTATION = f"{CW}:pose:orientation"

# Link
#
# UsdPhysicsMassAPI is entirely float32 (``physics:mass`` is ``float``,
# ``physics:centerOfMass`` is ``point3f``, ``physics:diagonalInertia`` is
# ``float3``, ``physics:principalAxes`` is ``quatf``), so every inertial value
# also gets a double sidecar. Without them a round trip would quietly perturb
# masses in the 8th significant digit and re-export would not be byte-stable.
A_LINK_MASS = f"{CW}:link:mass"
A_LINK_CENTER_OF_MASS = f"{CW}:link:centerOfMass"
A_LINK_INERTIA = f"{CW}:link:inertia"
"""Full symmetric tensor as ``[ixx, iyy, izz, ixy, ixz, iyz]``, in the inertial frame."""
A_LINK_INERTIAL_ORIENTATION = f"{CW}:link:inertialOrientation"
"""``Link.inertial_orientation`` (``x, y, z, w``); authored only when not identity.
The native ``MassAPI`` values are already in the link frame."""
A_LINK_MASS_FALLBACK = f"{CW}:link:massFallback"
"""``True`` when the native ``MassAPI`` values are the exporter's stand-in for a
massless link (no mass or inertia, and either no collision geometry or another
link in the schema carrying mass); the schema values are the (zero) sidecars."""

# Stand-in inertial for such a link. UsdPhysics has no "massless" body: with no
# authored mass and no collider to derive one from, PhysX substitutes its own
# default ("negative mass, small sphere approximated inertia"), which moved the
# Kinova wrist camera frames off the arm. One gram with the inertia of a
# 1-cm-radius-of-gyration body is negligible next to any real link yet well
# conditioned in a reduced-coordinate articulation -- the counterpart of the
# MJCF exporter's ``_FALLBACK_MASS`` for movable bodies MuJoCo would reject.
MASSLESS_LINK_FALLBACK_MASS_KG = 1e-3
MASSLESS_LINK_FALLBACK_INERTIA_KG_M2 = 1e-7

# Joint
A_JOINT_TYPE = f"{CW}:joint:type"
A_JOINT_AXIS = f"{CW}:joint:axis"
A_JOINT_POSE_POSITION = f"{CW}:joint:posePosition"
A_JOINT_POSE_ORIENTATION = f"{CW}:joint:poseOrientation"
A_JOINT_PARENT_LINK = f"{CW}:joint:parentLink"
A_JOINT_CHILD_LINK = f"{CW}:joint:childLink"
A_JOINT_HAS_LIMITS = f"{CW}:joint:hasLimits"
A_JOINT_LIMIT_LOWER = f"{CW}:joint:limitLower"
A_JOINT_LIMIT_UPPER = f"{CW}:joint:limitUpper"
A_JOINT_LIMIT_EFFORT = f"{CW}:joint:limitEffort"
A_JOINT_LIMIT_VELOCITY = f"{CW}:joint:limitVelocity"
A_JOINT_LIMIT_ACCELERATION = f"{CW}:joint:limitAcceleration"
A_JOINT_LIMIT_JERKMAX = f"{CW}:joint:limitJerkMax"
A_JOINT_HAS_DYNAMICS = f"{CW}:joint:hasDynamics"
A_JOINT_DYN_DAMPING = f"{CW}:joint:dynamicsDamping"
A_JOINT_DYN_FRICTION = f"{CW}:joint:dynamicsFriction"
A_JOINT_DYN_SPRING_REFERENCE = f"{CW}:joint:dynamicsSpringReference"
A_JOINT_DYN_SPRING_STIFFNESS = f"{CW}:joint:dynamicsSpringStiffness"
A_JOINT_DYN_ARMATURE = f"{CW}:joint:dynamicsArmature"
A_JOINT_DYN_BACKLASH = f"{CW}:joint:dynamicsBacklash"
A_JOINT_MIMIC_JOINT = f"{CW}:joint:mimicJoint"
A_JOINT_MIMIC_MULTIPLIER = f"{CW}:joint:mimicMultiplier"
A_JOINT_MIMIC_OFFSET = f"{CW}:joint:mimicOffset"
A_JOINT_HOME_POSITION = f"{CW}:joint:homePosition"
A_JOINT_SAFETY_CONTROLLER = f"{CW}:joint:safetyController"
A_JOINT_CALIBRATION = f"{CW}:joint:calibration"

# Geometry
A_GEOM_TYPE = f"{CW}:geometry:type"
A_GEOM_SIZE = f"{CW}:geometry:size"
A_GEOM_RADIUS = f"{CW}:geometry:radius"
A_GEOM_LENGTH = f"{CW}:geometry:length"
A_GEOM_MESH_FILENAME = f"{CW}:geometry:meshFilename"
"""Mesh reference, stored as a ``string`` rather than an ``asset``.

An ``asset``-typed attribute would be nicer to look at in usdview, but USD's
``usdz`` localization *rewrites* asset paths when it packages a stage — turning
``package://demo/meshes/arm.STL`` into ``0/arm.STL`` — which silently corrupts the
reference the rest of the pipeline (MJCF/URDF export, mesh resolution) depends on.
A ``string`` is left alone. The same reasoning applies to
:data:`A_MATERIAL_TEXTURE`; the *native* ``UsdUVTexture.inputs:file`` stays an
``asset``, because there localization rewriting the path is exactly right.
"""
A_GEOM_SCALE = f"{CW}:geometry:scale"

# Collision
A_COLLISION_GROUP = f"{CW}:collision:group"
A_COLLISION_MU_STATIC = f"{CW}:collision:muStatic"
A_COLLISION_MU_DYNAMIC = f"{CW}:collision:muDynamic"
A_COLLISION_RESTITUTION = f"{CW}:collision:restitution"
A_COLLISION_STIFFNESS = f"{CW}:collision:stiffness"
A_COLLISION_DAMPING = f"{CW}:collision:damping"

# Material
A_MATERIAL_COLOR = f"{CW}:material:color"
A_MATERIAL_SPECULAR = f"{CW}:material:specular"
A_MATERIAL_EMISSIVE = f"{CW}:material:emissive"
A_MATERIAL_SHININESS = f"{CW}:material:shininess"
A_MATERIAL_TEXTURE = f"{CW}:material:texture"

# Actuator
A_ACTUATOR_JOINT = f"{CW}:actuator:joint"
A_ACTUATOR_TYPE = f"{CW}:actuator:type"
A_ACTUATOR_TORQUE_CONSTANT = f"{CW}:actuator:torqueConstant"
A_ACTUATOR_GEAR_RATIO = f"{CW}:actuator:gearRatio"
A_ACTUATOR_MAX_CURRENT = f"{CW}:actuator:maxCurrent"
A_ACTUATOR_MAX_TORQUE = f"{CW}:actuator:maxTorque"
A_ACTUATOR_MAX_VELOCITY = f"{CW}:actuator:maxVelocity"
A_ACTUATOR_KP = f"{CW}:actuator:kp"
A_ACTUATOR_KI = f"{CW}:actuator:ki"
A_ACTUATOR_KD = f"{CW}:actuator:kd"
A_ACTUATOR_CONTROL_RANGE = f"{CW}:actuator:controlRange"
A_ACTUATOR_FORCE_RANGE = f"{CW}:actuator:forceRange"
A_ACTUATOR_RESISTANCE = f"{CW}:actuator:resistance"
A_ACTUATOR_INDUCTANCE = f"{CW}:actuator:inductance"
A_ACTUATOR_EFFICIENCY = f"{CW}:actuator:efficiency"

# Sensor
A_SENSOR_TYPE = f"{CW}:sensor:type"
A_SENSOR_UPDATE_RATE = f"{CW}:sensor:updateRate"
A_SENSOR_NOISE = f"{CW}:sensor:noise"
A_SENSOR_PARAMETERS = f"{CW}:sensor:parameters"
A_SENSOR_PARENT_LINK = f"{CW}:sensor:parentLink"

# Contact
A_CONTACT_LINK = f"{CW}:contact:link"

# ContactSurface — shared suffixes, used with a "cyberwave:contact:" or
# "cyberwave:physics:contact" prefix so one writer/reader pair serves both
# ``Contact.surface`` and ``Physics.contact``.
SURFACE_SUFFIXES = (
    "MuStatic",
    "MuDynamic",
    "Restitution",
    "Stiffness",
    "Damping",
    "SlipCompliance",
    "SoftCfm",
    "SoftErp",
)
CONTACT_SURFACE_PREFIX = f"{CW}:contact:surface"
PHYSICS_CONTACT_PREFIX = f"{CW}:physics:contact"

# Physics
A_PHYSICS_GRAVITY = f"{CW}:physics:gravity"
A_PHYSICS_TIMESTEP = f"{CW}:physics:timestep"
A_PHYSICS_SOLVER_TYPE = f"{CW}:physics:solverType"
A_PHYSICS_SOLVER_ITERATIONS = f"{CW}:physics:solverIterations"
A_PHYSICS_SOLVER_SOR = f"{CW}:physics:solverSor"
A_PHYSICS_SOLVER_MIN_STEP = f"{CW}:physics:solverMinStepSize"
A_PHYSICS_SOLVER_MAX_STEP = f"{CW}:physics:solverMaxStepSize"
A_PHYSICS_SOLVER_TOLERANCE = f"{CW}:physics:solverTolerance"

# Scene (render)
A_SCENE_AMBIENT = f"{CW}:scene:ambient"
A_SCENE_BACKGROUND = f"{CW}:scene:background"
A_SCENE_SHADOWS = f"{CW}:scene:shadows"
A_SCENE_GRID = f"{CW}:scene:grid"
A_SCENE_ORIGIN_VISUAL = f"{CW}:scene:originVisual"

# CollisionGroup
A_GROUP_CONTYPE = f"{CW}:collisionGroup:contype"
A_GROUP_CONAFFINITY = f"{CW}:collisionGroup:conaffinity"

# CollisionPair
A_PAIR_GEOM1 = f"{CW}:pair:geom1"
A_PAIR_GEOM2 = f"{CW}:pair:geom2"
A_PAIR_FRICTION = f"{CW}:pair:friction"
A_PAIR_SOLREF = f"{CW}:pair:solref"
A_PAIR_SOLIMP = f"{CW}:pair:solimp"
A_PAIR_CONDIM = f"{CW}:pair:condim"
A_PAIR_MARGIN = f"{CW}:pair:margin"
A_PAIR_GAP = f"{CW}:pair:gap"

# --------------------------------------------------------------------------
# USD token values
# --------------------------------------------------------------------------

TOK_X, TOK_Y, TOK_Z = "X", "Y", "Z"
AXIS_TOKENS = (TOK_X, TOK_Y, TOK_Z)

REF_AXIS_TOKEN = TOK_X
"""Reference axis token authored on every ``physics:axis``.

``"X"`` is the schema default for ``UsdPhysicsRevoluteJoint``,
``UsdPhysicsPrismaticJoint`` and ``UsdPhysicsSphericalJoint``, so authoring it
never surprises a consumer that reads an unauthored value, and using *one* token
for every joint keeps a single code path and deterministic output. The arbitrary
schema axis is baked into ``physics:localRot0``/``localRot1`` instead — see
:func:`cyberwave_robot_format.usd.frames.axis_to_ref_quat`.
"""

TOK_PURPOSE_RENDER = "render"
TOK_PURPOSE_GUIDE = "guide"
TOK_PURPOSE_PROXY = "proxy"
TOK_INVISIBLE = "invisible"

TOK_DRIVE_FORCE = "force"
TOK_DRIVE_ACCELERATION = "acceleration"

TOK_APPROX_NONE = "none"
TOK_APPROX_CONVEX_HULL = "convexHull"

TOK_MATERIAL_PURPOSE_PHYSICS = "physics"

DRIVE_ANGULAR = "angular"
DRIVE_LINEAR = "linear"

DOF_TRANS_X, DOF_TRANS_Y, DOF_TRANS_Z = "transX", "transY", "transZ"
DOF_ROT_X, DOF_ROT_Y, DOF_ROT_Z = "rotX", "rotY", "rotZ"

# --------------------------------------------------------------------------
# JointType <-> USD prim type
# --------------------------------------------------------------------------

PRIM_REVOLUTE = "PhysicsRevoluteJoint"
PRIM_PRISMATIC = "PhysicsPrismaticJoint"
PRIM_SPHERICAL = "PhysicsSphericalJoint"
PRIM_FIXED = "PhysicsFixedJoint"
PRIM_JOINT = "PhysicsJoint"

JOINT_TYPE_TO_PRIM_TYPE: dict[JointType, str] = {
    JointType.REVOLUTE: PRIM_REVOLUTE,
    JointType.CONTINUOUS: PRIM_REVOLUTE,
    JointType.PRISMATIC: PRIM_PRISMATIC,
    JointType.FIXED: PRIM_FIXED,
    JointType.FLOATING: PRIM_JOINT,
    JointType.PLANAR: PRIM_JOINT,
    JointType.SPHERICAL: PRIM_SPHERICAL,
    JointType.UNIVERSAL: PRIM_JOINT,
}

JOINT_TYPE_TO_USDPHYSICS_CLASS: dict[JointType, str] = {
    JointType.REVOLUTE: "RevoluteJoint",
    JointType.CONTINUOUS: "RevoluteJoint",
    JointType.PRISMATIC: "PrismaticJoint",
    JointType.FIXED: "FixedJoint",
    JointType.FLOATING: "Joint",
    JointType.PLANAR: "Joint",
    JointType.SPHERICAL: "SphericalJoint",
    JointType.UNIVERSAL: "Joint",
}
"""Attribute name on the ``pxr.UsdPhysics`` module for each joint type."""

JOINT_TYPES_WITH_AXIS_ATTR = frozenset(
    {
        JointType.REVOLUTE,
        JointType.CONTINUOUS,
        JointType.PRISMATIC,
        JointType.SPHERICAL,
    }
)
"""Joint types whose USD prim type actually declares ``physics:axis``.

``PhysicsFixedJoint`` and the base ``PhysicsJoint`` (used for ``FLOATING``,
``PLANAR`` and ``UNIVERSAL``) do not, so their axis survives only via
``cyberwave:joint:axis``.
"""

PRIM_TYPE_TO_JOINT_TYPE: dict[str, JointType] = {
    PRIM_REVOLUTE: JointType.REVOLUTE,
    PRIM_PRISMATIC: JointType.PRISMATIC,
    PRIM_SPHERICAL: JointType.SPHERICAL,
    PRIM_FIXED: JointType.FIXED,
    PRIM_JOINT: JointType.FLOATING,
}
"""Foreign-file fallback only.

``cyberwave:joint:type`` is the authority when authored — it is what
distinguishes ``CONTINUOUS`` from a limit-less ``REVOLUTE``, and ``PLANAR`` /
``UNIVERSAL`` from ``FLOATING`` (all three are a bare ``PhysicsJoint``).
"""

ANGULAR_JOINT_TYPES = frozenset(
    {
        JointType.REVOLUTE,
        JointType.CONTINUOUS,
        JointType.SPHERICAL,
        JointType.UNIVERSAL,
    }
)
"""Joint types whose position coordinate is an angle.

USD authors angular limits and angular drive targets in **degrees** while
``CommonSchema`` uses radians, so these are the types that need converting on
the native attributes. The ``cyberwave:`` sidecars always stay in schema units.
"""

LINEAR_JOINT_TYPES = frozenset({JointType.PRISMATIC, JointType.PLANAR})

DRIVE_INSTANCE_FOR_JOINT_TYPE: dict[JointType, str] = {
    JointType.REVOLUTE: DRIVE_ANGULAR,
    JointType.CONTINUOUS: DRIVE_ANGULAR,
    JointType.SPHERICAL: DRIVE_ANGULAR,
    JointType.UNIVERSAL: DRIVE_ANGULAR,
    JointType.PRISMATIC: DRIVE_LINEAR,
    JointType.PLANAR: DRIVE_LINEAR,
}
"""``UsdPhysicsDriveAPI`` instance name per joint type.

``FIXED`` and ``FLOATING`` are absent: a fixed joint has no DOF to drive, and a
floating joint's six DOFs cannot be expressed by one drive instance. Actuators on
those joints survive via the ``/Actuators`` record prims only.
"""

LOCKED_DOFS_FOR_JOINT_TYPE: dict[JointType, tuple[str, ...]] = {
    # A planar joint moves in the plane whose normal is the joint-frame X axis
    # (the schema/URDF convention: `axis` is the plane normal), so translation
    # along X and rotation about Y/Z are locked.
    JointType.PLANAR: (DOF_TRANS_X, DOF_ROT_Y, DOF_ROT_Z),
    # A universal (Hooke) joint has two rotational DOFs about joint-frame X and
    # Y; everything else is locked.
    JointType.UNIVERSAL: (DOF_TRANS_X, DOF_TRANS_Y, DOF_TRANS_Z, DOF_ROT_Z),
}
"""DOFs locked with a ``UsdPhysicsLimitAPI`` instance where ``low > high``.

``FLOATING`` deliberately has no entry: a generic ``PhysicsJoint`` with no
``LimitAPI`` instances already *is* a 6-DOF free joint in UsdPhysics.
"""

LOCK_LOW = 1.0
LOCK_HIGH = -1.0
"""``low > high`` is UsdPhysics' encoding for "this DOF is locked"."""

# --------------------------------------------------------------------------
# GeometryType <-> USD prim type
# --------------------------------------------------------------------------

PRIM_CUBE = "Cube"
PRIM_SPHERE = "Sphere"
PRIM_CYLINDER = "Cylinder"
PRIM_CAPSULE = "Capsule"
PRIM_PLANE = "Plane"
PRIM_MESH = "Mesh"
PRIM_XFORM = "Xform"

GEOMETRY_TYPE_TO_PRIM_TYPE: dict[GeometryType, str] = {
    GeometryType.BOX: PRIM_CUBE,
    GeometryType.SPHERE: PRIM_SPHERE,
    # UsdGeomSphere + a non-uniform xformOp:scale is USD's ellipsoid; the
    # cyberwave:geometry:type tag is what tells the two apart on import.
    GeometryType.ELLIPSOID: PRIM_SPHERE,
    GeometryType.CYLINDER: PRIM_CYLINDER,
    GeometryType.CAPSULE: PRIM_CAPSULE,
    GeometryType.PLANE: PRIM_PLANE,
    GeometryType.MESH: PRIM_MESH,
    GeometryType.UNKNOWN: PRIM_XFORM,
}

PRIM_TYPE_TO_GEOMETRY_TYPE: dict[str, GeometryType] = {
    PRIM_CUBE: GeometryType.BOX,
    PRIM_SPHERE: GeometryType.SPHERE,
    PRIM_CYLINDER: GeometryType.CYLINDER,
    PRIM_CAPSULE: GeometryType.CAPSULE,
    PRIM_PLANE: GeometryType.PLANE,
    PRIM_MESH: GeometryType.MESH,
    PRIM_XFORM: GeometryType.UNKNOWN,
}
"""Foreign-file fallback; ``cyberwave:geometry:type`` wins when authored."""

DEFAULT_PLANE_EXTENT = 10.0
"""Width/length authored for a ``PLANE`` with no ``size``.

Mirrors ``URDFExporter._add_geometry``'s 10x10 m default so the two exporters
agree on what an unsized plane means.
"""

# --------------------------------------------------------------------------
# ActuatorType
# --------------------------------------------------------------------------

DRIVE_TYPE_FOR_ACTUATOR: dict[ActuatorType, str] = {
    ActuatorType.DC_MOTOR: TOK_DRIVE_FORCE,
    ActuatorType.SERVO: TOK_DRIVE_FORCE,
    ActuatorType.VELOCITY: TOK_DRIVE_FORCE,
    ActuatorType.POSITION: TOK_DRIVE_FORCE,
    ActuatorType.TORQUE: TOK_DRIVE_FORCE,
    ActuatorType.MUSCLE: TOK_DRIVE_FORCE,
}
"""``drive:<inst>:physics:type``.

Every schema actuator maps to ``"force"``: ``"acceleration"`` means the solver
should ignore the body's mass, which no ``ActuatorType`` requests. The
distinction between the six actuator kinds is preserved by
``cyberwave:actuator:type``, not by the drive type.
"""

WORLD_LINK = "world"
"""Reserved pseudo-link name, matching ``CommonSchema.validate``/``merge_in``."""

# --------------------------------------------------------------------------
# USD-native passthrough
# --------------------------------------------------------------------------

USD_EXTENSION_KEY = "usd"
"""Sub-key under an object's ``extensions`` holding UsdPhysics-only properties.

``CommonSchema.extensions`` exists so format-specific features survive a
conversion, and ``MJCFParser`` already uses it that way (``collision.extensions``
carries MuJoCo's ``contype``, ``solref``, ``margin``, ...). These tables are the
USD equivalent: real UsdPhysics properties with no schema field of their own, which
would otherwise be silently dropped when a foreign stage is imported.

Every attribute listed here is one the exporter never authors on its own, which is
what keeps a Cyberwave-written stage from *gaining* an ``extensions["usd"]`` entry
on re-import. Attributes the exporter does derive — ``physics:excludeFromArticulation``
(from ``FLOATING``) and ``physics:approximation`` (from a mesh collision) — are
deliberately absent.
"""

PHYSX_EXTENSION_KEY = "physx"
"""Sub-key under an object's ``extensions`` holding NVIDIA PhysX schema attributes.

Keyed by the full attribute name (``"physxRigidBody:linearDamping"``); see
:mod:`cyberwave_robot_format.usd.physx` for the supported set and where each
applies.
"""

USD_PASSTHROUGH_LINK: tuple[tuple[str, str, str], ...] = (
    ("rigidBodyEnabled", "physics:rigidBodyEnabled", "bool"),
    ("kinematicEnabled", "physics:kinematicEnabled", "bool"),
    ("startsAsleep", "physics:startsAsleep", "bool"),
    ("velocity", "physics:velocity", "vector3f"),
    ("angularVelocity", "physics:angularVelocity", "vector3f"),
    ("density", "physics:density", "float"),
)
"""``(extensions key, USD attribute, value kind)`` for ``RigidBodyAPI``/``MassAPI``."""

USD_PASSTHROUGH_JOINT: tuple[tuple[str, str, str], ...] = (
    ("jointEnabled", "physics:jointEnabled", "bool"),
    ("collisionEnabled", "physics:collisionEnabled", "bool"),
    ("breakForce", "physics:breakForce", "float"),
    ("breakTorque", "physics:breakTorque", "float"),
    # A spherical joint's swing limits are two cone half-angles, which is not what
    # JointLimits.lower/upper mean — mapping them onto those fields would invent a
    # semantic. They are preserved verbatim instead (degrees, as USD stores them).
    ("coneAngle0Limit", "physics:coneAngle0Limit", "float"),
    ("coneAngle1Limit", "physics:coneAngle1Limit", "float"),
)

USD_PASSTHROUGH_COLLISION: tuple[tuple[str, str, str], ...] = (
    ("collisionEnabled", "physics:collisionEnabled", "bool"),
)

USD_PASSTHROUGH_GROUP: tuple[tuple[str, str, str], ...] = (
    ("invertFilteredGroups", "physics:invertFilteredGroups", "bool"),
    ("mergeGroup", "physics:mergeGroup", "string"),
)

DRIVE_TARGET_VELOCITY_KEY = "driveTargetVelocity"
"""``drive:<instance>:physics:targetVelocity``, kept under ``extensions["usd"]``.

``Actuator`` has a ``max_velocity`` but no *target* velocity, so a velocity-driven
Isaac Sim joint would otherwise lose its setpoint.
"""
