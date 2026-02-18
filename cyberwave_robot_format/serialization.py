"""
Serialization utilities using cattrs for CommonSchema (de)serialization.

This module provides a configured cattrs converter that handles:
- Enum types: serialize to .value, deserialize via EnumClass(value)
- Vector3/Quaternion/Pose: canonical dict-only format (no lists)
- All other dataclasses: automatic recursive conversion
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from cattrs import Converter

# Import schema types - these are needed for hook registration
from cyberwave_robot_format.schema import (
    ActuatorType,
    GeometryType,
    JointType,
    Pose,
    Quaternion,
    Vector3,
)


def _unstructure_enum(val: Enum) -> str:
    """Convert enum to its string value."""
    return val.value


def _structure_enum(val: Any, cls: type[Enum]) -> Enum:
    """Convert string value to enum."""
    if isinstance(val, cls):
        return val
    try:
        return cls(val)
    except ValueError as e:
        raise ValueError(f"Invalid value '{val}' for enum {cls.__name__}") from e


def _unstructure_vector3(val: Vector3) -> dict[str, float]:
    """Convert Vector3 to canonical dict format."""
    return {"x": val.x, "y": val.y, "z": val.z}


def _structure_vector3(val: Any, _: type[Vector3]) -> Vector3:
    """Convert dict to Vector3. Only accepts dict format (no lists)."""
    if val is None:
        return Vector3()
    if isinstance(val, Vector3):
        return val
    if not isinstance(val, dict):
        raise TypeError(
            f"Vector3 must be a dict with x/y/z keys, got {type(val).__name__}"
        )
    return Vector3(
        x=float(val.get("x", 0.0)),
        y=float(val.get("y", 0.0)),
        z=float(val.get("z", 0.0)),
    )


def _unstructure_quaternion(val: Quaternion) -> dict[str, float]:
    """Convert Quaternion to canonical dict format."""
    return {"x": val.x, "y": val.y, "z": val.z, "w": val.w}


def _structure_quaternion(val: Any, _: type[Quaternion]) -> Quaternion:
    """Convert dict to Quaternion. Only accepts dict format (no lists)."""
    if val is None:
        return Quaternion()
    if isinstance(val, Quaternion):
        return val
    if not isinstance(val, dict):
        raise TypeError(
            f"Quaternion must be a dict with x/y/z/w keys, got {type(val).__name__}"
        )
    return Quaternion(
        x=float(val.get("x", 0.0)),
        y=float(val.get("y", 0.0)),
        z=float(val.get("z", 0.0)),
        w=float(val.get("w", 1.0)),
    )


def _unstructure_pose(val: Pose) -> dict[str, Any]:
    """Convert Pose to dict format."""
    if val is None:
        raise TypeError("Pose cannot be None")
    return {
        "position": _unstructure_vector3(val.position),
        "orientation": _unstructure_quaternion(val.orientation),
    }


def _structure_pose(val: Any, _: type[Pose]) -> Pose:
    """Convert dict to Pose."""
    if val is None:
        raise TypeError("Pose cannot be None")
    if isinstance(val, Pose):
        return val
    if not isinstance(val, dict):
        raise TypeError(
            f"Pose must be a dict with position/orientation keys, got {type(val).__name__}"
        )
    return Pose(
        position=_structure_vector3(val.get("position"), Vector3),
        orientation=_structure_quaternion(val.get("orientation"), Quaternion),
    )


def _build_converter() -> Converter:
    """Build a configured cattrs converter for CommonSchema serialization."""
    converter = Converter()

    # Register enum hooks
    for enum_cls in (JointType, GeometryType, ActuatorType):
        converter.register_unstructure_hook(enum_cls, _unstructure_enum)
        converter.register_structure_hook(enum_cls, _structure_enum)

    # Register Vector3/Quaternion hooks (dict-only, no lists)
    converter.register_unstructure_hook(Vector3, _unstructure_vector3)
    converter.register_structure_hook(Vector3, _structure_vector3)
    converter.register_unstructure_hook(Quaternion, _unstructure_quaternion)
    converter.register_structure_hook(Quaternion, _structure_quaternion)

    # Register Pose hook
    converter.register_unstructure_hook(Pose, _unstructure_pose)
    converter.register_structure_hook(Pose, _structure_pose)

    return converter


# Single converter instance, built once at module load
converter = _build_converter()

__all__ = ["converter"]
