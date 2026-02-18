"""
Math utilities for robot format conversion.

This module provides common mathematical types used across the format conversion
pipeline: vectors, quaternions, poses, and inertia tensors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Vector3:
    """3D vector representation."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def to_list(self) -> list[float]:
        """Convert to list format."""
        return [self.x, self.y, self.z]

    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.x, self.y, self.z])

    @classmethod
    def from_list(cls, values: list[float]) -> Vector3:
        """Create from list of values."""
        return cls(values[0], values[1], values[2])


@dataclass
class Quaternion:
    """Quaternion representation for rotations."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0

    def to_list(self) -> list[float]:
        """Convert to list format [x, y, z, w]."""
        return [self.x, self.y, self.z, self.w]

    @classmethod
    def from_rpy(cls, roll: float, pitch: float, yaw: float) -> Quaternion:
        """Create quaternion from roll-pitch-yaw angles (Fixed XYZ convention)."""
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return cls(x, y, z, w)


@dataclass
class Pose:
    """6DOF pose representation."""

    position: Vector3 = None  # type: ignore[assignment]
    orientation: Quaternion = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.position is None:
            self.position = Vector3()
        if self.orientation is None:
            self.orientation = Quaternion()

    @classmethod
    def from_xyzrpy(cls, xyz: list[float], rpy: list[float]) -> Pose:
        """Create pose from position and RPY orientation."""
        return cls(
            position=Vector3.from_list(xyz),
            orientation=Quaternion.from_rpy(rpy[0], rpy[1], rpy[2]),
        )


@dataclass
class Inertia:
    """Inertia tensor representation."""

    ixx: float = 0.0
    iyy: float = 0.0
    izz: float = 0.0
    ixy: float = 0.0
    ixz: float = 0.0
    iyz: float = 0.0

    def to_matrix(self) -> np.ndarray:
        """Convert to 3x3 inertia matrix."""
        return np.array(
            [
                [self.ixx, self.ixy, self.ixz],
                [self.ixy, self.iyy, self.iyz],
                [self.ixz, self.iyz, self.izz],
            ]
        )
