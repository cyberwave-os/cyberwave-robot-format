# Copyright [2025] Cyberwave
# Licensed under the Apache License, Version 2.0

"""What the URDF spec does and does not give meaning to.

These rules decide which elements an exporter may emit for a given joint type.
They were worked out once in the backend's schema -> URDF sync and are hoisted here
so the exporter and any caller share one definition instead of drifting apart.
"""

from __future__ import annotations

# Schema joint ``type`` values that are valid URDF ``<joint type=...>`` strings.
# Anything else (spherical/universal) has no native URDF element and is
# approximated by the exporter's type mapping.
URDF_JOINT_TYPES = frozenset(
    {"revolute", "continuous", "prismatic", "fixed", "floating", "planar"}
)

# URDF only gives ``<axis>`` a meaning for joints that move along or about one.
# Writing an axis onto a fixed or floating joint is noise that also breaks
# round-trip equality against the original document.
AXIS_JOINT_TYPES = frozenset({"revolute", "continuous", "prismatic", "planar"})

# Joint types for which URDF *requires* positional bounds. A ``continuous`` joint
# must not carry ``lower``/``upper``: consumers treat the presence of bounds as the
# signal that a hinge is limited, so emitting them silently converts the joint.
BOUNDED_JOINT_TYPES = frozenset({"revolute", "prismatic"})

# A missing ``<axis>`` means this, per spec.
URDF_DEFAULT_AXIS: tuple[float, float, float] = (1.0, 0.0, 0.0)

# Mesh filename schemes that are already absolute references and must never be
# rewritten relative to the output file's directory.
ABSOLUTE_MESH_SCHEMES: tuple[str, ...] = (
    "package://",
    "model://",
    "file://",
    "http://",
    "https://",
)

__all__ = [
    "ABSOLUTE_MESH_SCHEMES",
    "AXIS_JOINT_TYPES",
    "BOUNDED_JOINT_TYPES",
    "URDF_DEFAULT_AXIS",
    "URDF_JOINT_TYPES",
]
