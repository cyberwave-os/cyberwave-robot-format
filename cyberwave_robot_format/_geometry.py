"""Lazy access to the shared geometry core.

This package holds no quaternion arithmetic of its own -- it lives in
``cyberwave_geometry``, a ctypes binding over the C++ core in
``common/geometry``. That package carries a native library, and importing it at
module scope makes it a hard requirement of ``import cyberwave_robot_format``,
including for the many consumers that only ever read a URDF.

So the import happens on first *use*. This is **not** a fallback: there is no
second implementation of the maths and there will not be one. A caller that
does rotation maths without the core installed gets an ImportError naming what
to install; a caller that never touches it is not made to care.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

__all__ = ["core"]

_CACHE: Any = None

_MISSING = (
    "cyberwave-geometry is not installed.\n"
    "\n"
    "cyberwave-robot-format does not implement quaternion or transform maths; "
    "it calls the shared geometry core, which ships a native library.\n"
    "\n"
    "Install it with:\n"
    "    pip install cyberwave-geometry\n"
    "\n"
    "or, from a monorepo checkout:\n"
    "    pip install common/geometry/bindings/python\n"
)


def core() -> Any:
    """Return the shared geometry core, importing it on first call."""
    global _CACHE
    if _CACHE is None:
        try:
            from cyberwave_geometry import GeometryError, Quaternion, Vector3, compat
            from cyberwave_geometry import quaternion as quaternion_module
            from cyberwave_geometry import transform as transform_module
        except ImportError as error:  # pragma: no cover - depends on the install
            raise ImportError(_MISSING) from error
        _CACHE = SimpleNamespace(
            quat=quaternion_module,
            transform=transform_module,
            compat=compat,
            Quaternion=Quaternion,
            Vector3=Vector3,
            GeometryError=GeometryError,
        )
    return _CACHE
