# Copyright [2025] Cyberwave
# Licensed under the Apache License, Version 2.0

"""NVIDIA PhysX / Newton schema attributes, authored without the NVIDIA SDK.

Isaac Sim reads a good part of its physics from ``PhysxSchema``
(``omni.usd.schema.physx``) and the Newton schemas rather than from core
UsdPhysics: joint velocity limits, joint armature, articulation solver iterations,
contact offsets, mimic joints. Those schemas ship only inside Kit, so ``usd-core``
has no typed ``Apply()`` for them. It does not need one. An applied API schema is a
token in the prim's ``apiSchemas`` list plus attributes with schema-declared names,
and both are plain USD: ``prim.AddAppliedSchema("PhysxJointAPI")`` and a non-custom
``physxJoint:maxJointVelocity`` attribute are exactly what Kit itself writes.

Two sources feed it:

* **Derived** values. The schema already knows ``limits.velocity``,
  ``dynamics.armature`` and ``mimic``, and Isaac only sees them when they are
  authored in the PhysX/Newton vocabulary (:func:`derived_joint_physx`,
  :func:`write_newton_mimic`).
* **Passthrough** values under ``extensions["physx"]``, keyed by the full
  attribute name (``{"physxRigidBody:linearDamping": 0.1}``), on the schema root
  (articulation), links, collisions, joints and ``Physics`` (scene). An explicit
  passthrough value wins over a derived one. An entry that is unknown, belongs to
  another prim kind or has a value of the wrong type is skipped and reported, never
  a failed export: it comes from user-editable JSON.

The table is generated from the ``generatedSchema.usda`` of
``omni.usd.schema.physx`` 110.3 (Isaac Sim 6). It leaves out the multiple-apply
schemas (their instance names depend on the joint frame), the material schemas
(materials are shared between bodies) and the scene attributes the Isaac runtime
owns (broadphase, GPU dynamics, iteration-count clamps, step rate, update type), so
an asset cannot silently override a runtime profile.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from cyberwave_robot_format.schema import Joint, JointType, MimicJoint
from cyberwave_robot_format.usd import tokens as t
from cyberwave_robot_format.usd.common import require_pxr


@dataclass(frozen=True)
class PhysxAttribute:
    name: str
    api: str
    kind: str
    allowed: tuple[str, ...] = ()
    uniform: bool = False


def _attr(
    name: str,
    api: str,
    kind: str,
    *,
    allowed: tuple[str, ...] = (),
    uniform: bool = False,
) -> PhysxAttribute:
    return PhysxAttribute(name, api, kind, allowed, uniform)


_TABLE: tuple[PhysxAttribute, ...] = (
    _attr("physxArticulation:articulationEnabled", "PhysxArticulationAPI", "bool"),
    _attr("physxArticulation:enabledSelfCollisions", "PhysxArticulationAPI", "bool"),
    _attr("physxArticulation:sleepThreshold", "PhysxArticulationAPI", "float"),
    _attr(
        "physxArticulation:solverPositionIterationCount", "PhysxArticulationAPI", "int"
    ),
    _attr(
        "physxArticulation:solverVelocityIterationCount", "PhysxArticulationAPI", "int"
    ),
    _attr("physxArticulation:stabilizationThreshold", "PhysxArticulationAPI", "float"),
    _attr("physxRigidBody:angularDamping", "PhysxRigidBodyAPI", "float"),
    _attr("physxRigidBody:cfmScale", "PhysxRigidBodyAPI", "float"),
    _attr("physxRigidBody:contactSlopCoefficient", "PhysxRigidBodyAPI", "float"),
    _attr("physxRigidBody:disableGravity", "PhysxRigidBodyAPI", "bool"),
    _attr("physxRigidBody:enableCCD", "PhysxRigidBodyAPI", "bool"),
    _attr("physxRigidBody:enableGyroscopicForces", "PhysxRigidBodyAPI", "bool"),
    _attr("physxRigidBody:enableSpeculativeCCD", "PhysxRigidBodyAPI", "bool"),
    _attr("physxRigidBody:linearDamping", "PhysxRigidBodyAPI", "float"),
    _attr("physxRigidBody:lockedPosAxis", "PhysxRigidBodyAPI", "int"),
    _attr("physxRigidBody:lockedRotAxis", "PhysxRigidBodyAPI", "int"),
    _attr("physxRigidBody:maxAngularVelocity", "PhysxRigidBodyAPI", "float"),
    _attr("physxRigidBody:maxContactImpulse", "PhysxRigidBodyAPI", "float"),
    _attr("physxRigidBody:maxDepenetrationVelocity", "PhysxRigidBodyAPI", "float"),
    _attr("physxRigidBody:maxLinearVelocity", "PhysxRigidBodyAPI", "float"),
    _attr("physxRigidBody:retainAccelerations", "PhysxRigidBodyAPI", "bool"),
    _attr("physxRigidBody:sleepThreshold", "PhysxRigidBodyAPI", "float"),
    _attr("physxRigidBody:solveContact", "PhysxRigidBodyAPI", "bool"),
    _attr("physxRigidBody:solverPositionIterationCount", "PhysxRigidBodyAPI", "int"),
    _attr("physxRigidBody:solverVelocityIterationCount", "PhysxRigidBodyAPI", "int"),
    _attr("physxRigidBody:stabilizationThreshold", "PhysxRigidBodyAPI", "float"),
    _attr("physxContactReport:threshold", "PhysxContactReportAPI", "float"),
    _attr("physxCollision:contactOffset", "PhysxCollisionAPI", "float"),
    _attr("physxCollision:minTorsionalPatchRadius", "PhysxCollisionAPI", "float"),
    _attr("physxCollision:restOffset", "PhysxCollisionAPI", "float"),
    _attr("physxCollision:torsionalPatchRadius", "PhysxCollisionAPI", "float"),
    _attr(
        "physxConvexHullCollision:hullVertexLimit", "PhysxConvexHullCollisionAPI", "int"
    ),
    _attr(
        "physxConvexHullCollision:minThickness", "PhysxConvexHullCollisionAPI", "float"
    ),
    _attr(
        "physxConvexDecompositionCollision:errorPercentage",
        "PhysxConvexDecompositionCollisionAPI",
        "float",
    ),
    _attr(
        "physxConvexDecompositionCollision:hullVertexLimit",
        "PhysxConvexDecompositionCollisionAPI",
        "int",
    ),
    _attr(
        "physxConvexDecompositionCollision:maxConvexHulls",
        "PhysxConvexDecompositionCollisionAPI",
        "int",
    ),
    _attr(
        "physxConvexDecompositionCollision:minThickness",
        "PhysxConvexDecompositionCollisionAPI",
        "float",
    ),
    _attr(
        "physxConvexDecompositionCollision:shrinkWrap",
        "PhysxConvexDecompositionCollisionAPI",
        "bool",
    ),
    _attr(
        "physxConvexDecompositionCollision:voxelResolution",
        "PhysxConvexDecompositionCollisionAPI",
        "int",
    ),
    _attr(
        "physxSDFMeshCollision:sdfBitsPerSubgridPixel",
        "PhysxSDFMeshCollisionAPI",
        "token",
        allowed=("BitsPerPixel8", "BitsPerPixel16", "BitsPerPixel32"),
        uniform=True,
    ),
    _attr(
        "physxSDFMeshCollision:sdfEnableRemeshing",
        "PhysxSDFMeshCollisionAPI",
        "bool",
        uniform=True,
    ),
    _attr(
        "physxSDFMeshCollision:sdfMargin",
        "PhysxSDFMeshCollisionAPI",
        "float",
        uniform=True,
    ),
    _attr(
        "physxSDFMeshCollision:sdfNarrowBandThickness",
        "PhysxSDFMeshCollisionAPI",
        "float",
        uniform=True,
    ),
    _attr(
        "physxSDFMeshCollision:sdfResolution",
        "PhysxSDFMeshCollisionAPI",
        "int",
        uniform=True,
    ),
    _attr(
        "physxSDFMeshCollision:sdfSubgridResolution",
        "PhysxSDFMeshCollisionAPI",
        "int",
        uniform=True,
    ),
    _attr(
        "physxSDFMeshCollision:sdfTriangleCountReductionFactor",
        "PhysxSDFMeshCollisionAPI",
        "float",
        uniform=True,
    ),
    _attr("physxJoint:armature", "PhysxJointAPI", "float"),
    _attr("physxJoint:jointFriction", "PhysxJointAPI", "float"),
    _attr("physxJoint:maxJointVelocity", "PhysxJointAPI", "float"),
    _attr("physxScene:bounceThreshold", "PhysxSceneAPI", "float"),
    _attr(
        "physxScene:collisionSystem",
        "PhysxSceneAPI",
        "token",
        allowed=("PCM", "SAT"),
        uniform=True,
    ),
    _attr("physxScene:disableSleeping", "PhysxSceneAPI", "bool", uniform=True),
    _attr("physxScene:enableCCD", "PhysxSceneAPI", "bool"),
    _attr("physxScene:enableEnhancedDeterminism", "PhysxSceneAPI", "bool"),
    _attr("physxScene:enableExternalForcesEveryIteration", "PhysxSceneAPI", "bool"),
    _attr("physxScene:enableStabilization", "PhysxSceneAPI", "bool"),
    _attr("physxScene:frictionCorrelationDistance", "PhysxSceneAPI", "float"),
    _attr("physxScene:frictionOffsetThreshold", "PhysxSceneAPI", "float"),
    _attr(
        "physxScene:frictionType",
        "PhysxSceneAPI",
        "token",
        allowed=("patch",),
        uniform=True,
    ),
    _attr("physxScene:gpuCollisionStackSize", "PhysxSceneAPI", "uint"),
    _attr("physxScene:gpuFoundLostAggregatePairsCapacity", "PhysxSceneAPI", "uint"),
    _attr("physxScene:gpuFoundLostPairsCapacity", "PhysxSceneAPI", "uint"),
    _attr("physxScene:gpuHeapCapacity", "PhysxSceneAPI", "uint"),
    _attr("physxScene:gpuMaxDeformableSurfaceContacts", "PhysxSceneAPI", "uint"),
    _attr("physxScene:gpuMaxDeformableVolumeContacts", "PhysxSceneAPI", "uint"),
    _attr("physxScene:gpuMaxNumPartitions", "PhysxSceneAPI", "uint"),
    _attr("physxScene:gpuMaxParticleContacts", "PhysxSceneAPI", "uint"),
    _attr("physxScene:gpuMaxRigidContactCount", "PhysxSceneAPI", "uint"),
    _attr("physxScene:gpuMaxRigidPatchCount", "PhysxSceneAPI", "uint"),
    _attr("physxScene:gpuTempBufferCapacity", "PhysxSceneAPI", "uint64"),
    _attr("physxScene:gpuTotalAggregatePairsCapacity", "PhysxSceneAPI", "uint"),
    _attr("physxScene:invertCollisionGroupFilter", "PhysxSceneAPI", "bool"),
    _attr("physxScene:maxBiasCoefficient", "PhysxSceneAPI", "float"),
    _attr("physxScene:reportKinematicKinematicPairs", "PhysxSceneAPI", "bool"),
    _attr("physxScene:reportKinematicStaticPairs", "PhysxSceneAPI", "bool"),
    _attr("physxScene:solveArticulationContactLast", "PhysxSceneAPI", "bool"),
    _attr(
        "physxScene:solverType",
        "PhysxSceneAPI",
        "token",
        allowed=("PGS", "TGS"),
        uniform=True,
    ),
)

PHYSX_ATTRIBUTES: dict[str, PhysxAttribute] = {a.name: a for a in _TABLE}

ARTICULATION_APIS = frozenset({"PhysxArticulationAPI"})
LINK_APIS = frozenset({"PhysxRigidBodyAPI", "PhysxContactReportAPI"})
COLLISION_APIS = frozenset(
    {
        "PhysxCollisionAPI",
        "PhysxConvexHullCollisionAPI",
        "PhysxConvexDecompositionCollisionAPI",
        "PhysxSDFMeshCollisionAPI",
    }
)
JOINT_APIS = frozenset({"PhysxJointAPI"})
SCENE_APIS = frozenset({"PhysxSceneAPI"})

MAX_JOINT_VELOCITY = "physxJoint:maxJointVelocity"
JOINT_ARMATURE = "physxJoint:armature"
DERIVED_JOINT_KEYS = frozenset({MAX_JOINT_VELOCITY, JOINT_ARMATURE})
"""Joint attributes the exporter derives from schema fields.

On a Cyberwave-written joint they are a float32 copy of a sidecar the parser
already reads, so harvesting them would *add* an ``extensions["physx"]`` entry on
every round trip."""

NEWTON_MIMIC_API = "NewtonMimicAPI"
NEWTON_MIMIC_JOINT = "newton:mimicJoint"
NEWTON_MIMIC_COEF0 = "newton:mimicCoef0"
NEWTON_MIMIC_COEF1 = "newton:mimicCoef1"
NEWTON_MIMIC_ENABLED = "newton:mimicEnabled"
# Deprecated by NVIDIA in favour of NewtonMimicAPI; read for older assets only.
_PHYSX_MIMIC_PREFIX = "physxMimicJoint:"
_PHYSX_MIMIC_REFERENCE = ":referenceJoint"

_SINGLE_DOF_JOINTS = (JointType.REVOLUTE, JointType.CONTINUOUS, JointType.PRISMATIC)
_INTEGER_RANGES = {
    "int": (-(2**31), 2**31 - 1),
    "uint": (0, 2**32 - 1),
    "uint64": (0, 2**64 - 1),
}
_KIND_DESCRIPTIONS = {
    "bool": "a boolean",
    "int": "a 32-bit integer",
    "uint": "a non-negative 32-bit integer",
    "uint64": "a non-negative 64-bit integer",
    "float": "a finite number",
    "token": "a string",
}


def physx_extension(extensions: Mapping[str, Any] | None) -> dict[str, Any]:
    """The ``extensions["physx"]`` dict of an object, or ``{}``."""
    if not extensions:
        return {}
    stored = extensions.get(t.PHYSX_EXTENSION_KEY)
    return dict(stored) if isinstance(stored, Mapping) else {}


def coerce_value(attribute: PhysxAttribute, value: Any) -> Any:
    """``value`` as the Python type USD stores for ``attribute``.

    Raises ``ValueError`` for anything else. Booleans are not accepted as numbers
    (JSON ``true`` for a solver iteration count is a mistake, not a ``1``).
    """
    kind = attribute.kind
    if kind == "bool":
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
    elif kind in _INTEGER_RANGES and not isinstance(value, (bool, np.bool_)):
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        if isinstance(value, (int, np.integer)):
            low, high = _INTEGER_RANGES[kind]
            if low <= int(value) <= high:
                return int(value)
    elif kind == "float" and not isinstance(value, (bool, np.bool_)):
        if isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(
            float(value)
        ):
            return float(value)
    elif kind == "token" and isinstance(value, str):
        if not attribute.allowed or value in attribute.allowed:
            return value
    expected = (
        "one of " + ", ".join(attribute.allowed)
        if attribute.allowed
        else _KIND_DESCRIPTIONS[kind]
    )
    raise ValueError(f"{attribute.name} expects {expected}, got {value!r}")


def _value_type(kind: str) -> Any:
    names = require_pxr().Sdf.ValueTypeNames
    return {
        "bool": names.Bool,
        "int": names.Int,
        "uint": names.UInt,
        "uint64": names.UInt64,
        "float": names.Float,
        "token": names.Token,
    }[kind]


def write_physx(
    prim: Any, apis: frozenset[str], values: Mapping[str, Any]
) -> list[str]:
    """Author ``values`` on ``prim`` and apply the API schema each belongs to.

    ``apis`` is the set of schemas that make sense on this kind of prim; an
    attribute of any other schema is refused rather than written somewhere Isaac
    would ignore it. Returns one message per skipped entry.
    """
    pxr = require_pxr()
    problems: list[str] = []
    for name in sorted(values):
        attribute = PHYSX_ATTRIBUTES.get(name)
        if attribute is None:
            problems.append(f"{name!r} is not a supported PhysX attribute")
            continue
        if attribute.api not in apis:
            problems.append(f"{name!r} belongs to {attribute.api}, not to this prim")
            continue
        try:
            value = coerce_value(attribute, values[name])
        except ValueError as error:
            problems.append(str(error))
            continue
        # Idempotent: a second add of the same token is a no-op.
        prim.AddAppliedSchema(attribute.api)
        prim.CreateAttribute(
            name,
            _value_type(attribute.kind),
            custom=False,
            variability=(
                pxr.Sdf.VariabilityUniform
                if attribute.uniform
                else pxr.Sdf.VariabilityVarying
            ),
        ).Set(value)
    return problems


def read_physx(
    prim: Any, apis: frozenset[str], *, skip: frozenset[str] = frozenset()
) -> dict[str, Any]:
    """Harvest the authored attributes of ``apis`` from ``prim``.

    Only authored values are returned, so a schema fallback never turns into data.
    The API token itself is not required: ``usd-core`` cannot see unregistered
    schemas in ``GetAppliedSchemas()``, and the attribute name is unambiguous.
    """
    harvested: dict[str, Any] = {}
    for attribute in _TABLE:
        if attribute.api not in apis or attribute.name in skip:
            continue
        usd_attribute = prim.GetAttribute(attribute.name)
        if not usd_attribute or not usd_attribute.HasAuthoredValue():
            continue
        value = usd_attribute.Get()
        if value is None:
            continue
        if attribute.kind == "bool":
            harvested[attribute.name] = bool(value)
        elif attribute.kind in _INTEGER_RANGES:
            harvested[attribute.name] = int(value)
        elif attribute.kind == "float":
            number = float(value)
            if math.isfinite(number):
                harvested[attribute.name] = number
        else:
            harvested[attribute.name] = str(value)
    return harvested


def merge_physx(
    extensions: dict[str, Any], harvested: dict[str, Any]
) -> dict[str, Any]:
    """Fold harvested attributes into ``extensions["physx"]``.

    The stored (``customData``) copy is the base. A harvested float that is only
    the float32 rounding of the stored double keeps the stored value, so a
    Cyberwave-written stage reads back exactly what was exported.
    """
    if not harvested:
        return extensions
    merged = physx_extension(extensions)
    for name, value in harvested.items():
        if name in merged and _same_float32(merged[name], value):
            continue
        merged[name] = value
    extensions[t.PHYSX_EXTENSION_KEY] = merged
    return extensions


def _same_float32(stored: Any, harvested: Any) -> bool:
    if isinstance(stored, bool) or isinstance(harvested, bool):
        return stored == harvested
    if isinstance(stored, (int, float)) and isinstance(harvested, (int, float)):
        return bool(np.float32(stored) == np.float32(harvested))
    return bool(stored == harvested)


def derived_joint_physx(joint: Joint) -> dict[str, float]:
    """The ``PhysxJointAPI`` values a joint's schema fields imply.

    * ``limits.velocity`` (rad/s or m/s) is ``physxJoint:maxJointVelocity``
      (deg/s or m/s), as NVIDIA's URDF importer maps it. Zero or negative is
      skipped: URDF files use ``velocity="0"`` for "unspecified", and authoring it
      would freeze the joint.
    * ``dynamics.armature`` is ``physxJoint:armature`` in the same units.

    Joint friction is deliberately not derived: ``physxJoint:jointFriction`` is a
    dimensionless coefficient, while ``dynamics.friction`` is a torque/force.
    """
    derived: dict[str, float] = {}
    if joint.type not in _SINGLE_DOF_JOINTS:
        return derived
    limits = joint.limits
    if limits is not None and limits.velocity is not None:
        velocity = float(limits.velocity)
        if math.isfinite(velocity) and velocity > 0.0:
            derived[MAX_JOINT_VELOCITY] = (
                math.degrees(velocity)
                if joint.type in t.ANGULAR_JOINT_TYPES
                else velocity
            )
    dynamics = joint.dynamics
    if dynamics is not None and dynamics.armature is not None:
        armature = float(dynamics.armature)
        if math.isfinite(armature) and armature > 0.0:
            derived[JOINT_ARMATURE] = armature
    return derived


def native_mimic_problem(follower: Joint, leader: Joint | None) -> str | None:
    """Why ``follower``'s mimic cannot be a ``NewtonMimicAPI``, or ``None``.

    The constraints are NVIDIA's: the two joints share a type, and a revolute one
    needs finite limits ("NewtonMimicAPI ... is a revolute joint without a finite
    limit set" is a PhysX parse error). ``CONTINUOUS`` has no limits by definition.
    """
    assert follower.mimic is not None
    if leader is None:
        return f"leader joint {follower.mimic.joint!r} does not exist"
    if leader.name == follower.name:
        return "a joint cannot mimic itself"
    if follower.type not in (JointType.REVOLUTE, JointType.PRISMATIC):
        return f"{follower.type.value} joints have no native mimic"
    if leader.type != follower.type:
        return (
            f"the leader is {leader.type.value} and the follower "
            f"{follower.type.value}; a native mimic needs one joint type"
        )
    if follower.type == JointType.REVOLUTE:
        for joint in (follower, leader):
            if not _has_finite_limits(joint):
                return f"revolute joint {joint.name!r} has no finite limits"
    return None


def _has_finite_limits(joint: Joint) -> bool:
    limits = joint.limits
    return (
        limits is not None
        and limits.lower is not None
        and limits.upper is not None
        and math.isfinite(limits.lower)
        and math.isfinite(limits.upper)
    )


def write_newton_mimic(
    prim: Any, leader_path: Any, mimic: MimicJoint, *, angular: bool
) -> None:
    """Author ``joint = offset + multiplier * leader`` as a ``NewtonMimicAPI``.

    It is NVIDIA's non-deprecated mimic schema, and both Isaac's PhysX and Newton
    backends parse it. The offset is in degrees for an angular joint.
    """
    pxr = require_pxr()
    names = pxr.Sdf.ValueTypeNames
    prim.AddAppliedSchema(NEWTON_MIMIC_API)
    prim.CreateRelationship(NEWTON_MIMIC_JOINT, custom=False).SetTargets([leader_path])
    offset = float(mimic.offset)
    prim.CreateAttribute(NEWTON_MIMIC_COEF0, names.Float, custom=False).Set(
        math.degrees(offset) if angular else offset
    )
    prim.CreateAttribute(NEWTON_MIMIC_COEF1, names.Float, custom=False).Set(
        float(mimic.multiplier)
    )
    prim.CreateAttribute(NEWTON_MIMIC_ENABLED, names.Bool, custom=False).Set(True)


def read_native_mimic(prim: Any) -> tuple[Any, float, float] | None:
    """A native mimic on ``prim``: ``(leader path, multiplier, offset)``.

    The offset is in USD units (degrees for an angular joint). Reads
    ``NewtonMimicAPI`` and the deprecated ``PhysxMimicJointAPI``, whose relation
    ``joint + gearing * reference + offset = 0`` is the same line with both
    coefficients negated. A disabled ``NewtonMimicAPI`` is not a mimic.
    """
    relationship = prim.GetRelationship(NEWTON_MIMIC_JOINT)
    targets = relationship.GetTargets() if relationship else []
    if targets:
        enabled = prim.GetAttribute(NEWTON_MIMIC_ENABLED)
        if enabled and enabled.HasAuthoredValue() and not enabled.Get():
            return None
        return (
            targets[0],
            _authored_float(prim, NEWTON_MIMIC_COEF1, 1.0),
            _authored_float(prim, NEWTON_MIMIC_COEF0, 0.0),
        )
    for relationship in prim.GetRelationships():
        name = relationship.GetName()
        if not (
            name.startswith(_PHYSX_MIMIC_PREFIX)
            and name.endswith(_PHYSX_MIMIC_REFERENCE)
        ):
            continue
        targets = relationship.GetTargets()
        if not targets:
            continue
        base = name[: -len(_PHYSX_MIMIC_REFERENCE)]
        return (
            targets[0],
            -_authored_float(prim, f"{base}:gearing", 1.0),
            -_authored_float(prim, f"{base}:offset", 0.0),
        )
    return None


def _authored_float(prim: Any, name: str, default: float) -> float:
    attribute = prim.GetAttribute(name)
    if not attribute or not attribute.HasAuthoredValue():
        return default
    value = attribute.Get()
    return default if value is None else float(value)


__all__ = [
    "ARTICULATION_APIS",
    "COLLISION_APIS",
    "DERIVED_JOINT_KEYS",
    "JOINT_APIS",
    "LINK_APIS",
    "PHYSX_ATTRIBUTES",
    "SCENE_APIS",
    "PhysxAttribute",
    "coerce_value",
    "derived_joint_physx",
    "merge_physx",
    "native_mimic_problem",
    "physx_extension",
    "read_native_mimic",
    "read_physx",
    "write_newton_mimic",
    "write_physx",
]
