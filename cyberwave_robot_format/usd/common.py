# Copyright [2025] Cyberwave
# Licensed under the Apache License, Version 2.0

"""Shared ``pxr`` bridge helpers for the USD exporter and parser.

Three concerns live here, all of which the exporter and parser must agree on
exactly:

* **The deferred ``pxr`` import.** ``usd-core`` is an optional dependency
  (``pip install cyberwave-robot-format[usd]``); importing this module without it
  must succeed so that ``cyberwave_robot_format`` stays importable on platforms
  with no ``usd-core`` wheel. Only *calling* into USD fails, with an actionable
  message.
* **Optional-value authoring.** ``CommonSchema`` uses ``None`` to mean "not
  specified" all over (``mu_static``, ``limits.effort``, ``update_rate``, ...),
  and USD has the same notion in ``HasAuthoredValue()``. Every helper below maps
  one onto the other, so ``mu_static=None`` never round-trips as ``0.0``.
* **Name sanitization.** USD prim names must be valid identifiers; schema names
  routinely are not.
"""

from __future__ import annotations

import json
import logging
import math
import re
from types import SimpleNamespace
from typing import Any

import numpy as np

from cyberwave_robot_format.schema import Quaternion, Vector3

logger = logging.getLogger(__name__)

_PXR: SimpleNamespace | None = None


def require_pxr() -> SimpleNamespace:
    """Import and return the ``pxr`` modules the USD converters need.

    Cached after the first successful import. Raises a clear, actionable
    ``ImportError`` — distinct from the deliberately silent
    ``except ImportError: logger.debug(...)`` in ``core.py``'s optional-format
    registry — because a caller that reaches here genuinely needs USD.
    """
    global _PXR
    if _PXR is None:
        try:
            from pxr import Gf, Sdf, Tf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, UsdUtils, Vt
        except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
            raise ImportError(
                "USD support requires the 'pxr' module. Install it with "
                "`pip install cyberwave-robot-format[usd]` (or `pip install "
                "usd-core` directly) to read or write .usd/.usda/.usdc/.usdz files."
            ) from exc
        _PXR = SimpleNamespace(
            Gf=Gf,
            Sdf=Sdf,
            Tf=Tf,
            Usd=Usd,
            UsdGeom=UsdGeom,
            UsdLux=UsdLux,
            UsdPhysics=UsdPhysics,
            UsdShade=UsdShade,
            UsdUtils=UsdUtils,
            Vt=Vt,
        )
    return _PXR


# --------------------------------------------------------------------------
# Names
# --------------------------------------------------------------------------

_INVALID_NAME_CHARS = re.compile(r"[^A-Za-z0-9_]")


def sanitize_prim_name(name: str) -> str:
    """Return ``name`` reduced to a valid USD identifier.

    USD prim names must match ``[A-Za-z_][A-Za-z0-9_]*``. Schema names routinely
    contain ``-`` and ``.``, and ``CommonSchema.merge_in`` produces
    ``instance__link`` prefixes (double underscore is already valid, so merged
    names pass through untouched).

    This is intentionally **not** reversible. A reversible escape grammar
    (``-`` -> ``_x2D_``) is not injective unless ``_`` is escaped too, which turns
    every clean name into unreadable soup and defeats the point of authoring
    ``.usda``. Reversibility comes from the ``cyberwave:name`` sidecar instead,
    which :class:`PrimNamer` decides about.

    Note this is *not* ``utils.sanitize_name``, which preserves ``-`` and is
    therefore invalid for USD.
    """
    sanitized = _INVALID_NAME_CHARS.sub("_", name)
    if not sanitized or not (sanitized[0].isalpha() or sanitized[0] == "_"):
        sanitized = f"_{sanitized}"
    return sanitized


class PrimNamer:
    """Assigns collision-free prim names, one claim per entity.

    Uniqueness is scoped per parent prim path, since USD only requires *sibling*
    names to differ.

    Each call claims a **new** name; results are deliberately not cached by
    ``(parent, schema_name)``. Two sibling entities may legitimately share a schema
    name — two visuals both called ``"collision_geom"`` on one link, say — and
    handing them the same prim name would silently merge them into one prim.
    Callers therefore claim exactly once per entity and keep the returned path.
    """

    def __init__(self) -> None:
        self._taken: dict[str, set[str]] = {}

    def name_for(self, parent_path: str, schema_name: str) -> str:
        """Claim and return a unique prim name for ``schema_name`` under ``parent_path``."""
        taken = self._taken.setdefault(parent_path, set())
        base = sanitize_prim_name(schema_name)
        candidate = base
        suffix = 2
        while candidate in taken:
            candidate = f"{base}_{suffix}"
            suffix += 1
        taken.add(candidate)
        return candidate

    def needs_name_sidecar(self, prim_name: str, schema_name: str) -> bool:
        """Whether ``cyberwave:name`` must be authored to recover ``schema_name``."""
        return prim_name != schema_name


def schema_name_from_prim(prim: Any) -> str:
    """Return the schema name for ``prim``: the sidecar if authored, else its name."""
    sidecar = read_string(prim, "cyberwave:name")
    return sidecar if sidecar is not None else prim.GetName()


# --------------------------------------------------------------------------
# Matrix / vector / quaternion bridges
# --------------------------------------------------------------------------


def np_to_gf_matrix(matrix: np.ndarray) -> Any:
    """Convert a column-vector 4x4 numpy transform to ``Gf.Matrix4d``.

    USD matrices are row-major/row-vector (``v' = v * M``) while this package uses
    column-vector (``v' = M * v``), hence the transpose.
    """
    pxr = require_pxr()
    return pxr.Gf.Matrix4d(*np.asarray(matrix, dtype=np.float64).T.flatten().tolist())


def gf_matrix_to_np(matrix: Any) -> np.ndarray:
    """Convert a ``Gf.Matrix4d`` to a column-vector 4x4 numpy transform."""
    return np.array(matrix, dtype=np.float64).T


def vector3_to_gf(vector: Vector3, precision: str = "double") -> Any:
    """Convert :class:`Vector3` to ``Gf.Vec3d`` (or ``Gf.Vec3f``)."""
    pxr = require_pxr()
    cls = pxr.Gf.Vec3d if precision == "double" else pxr.Gf.Vec3f
    return cls(float(vector.x), float(vector.y), float(vector.z))


def gf_to_vector3(value: Any) -> Vector3:
    """Convert any ``Gf.Vec3*`` to :class:`Vector3`."""
    return Vector3(x=float(value[0]), y=float(value[1]), z=float(value[2]))


def quaternion_to_gf(quat: Quaternion, precision: str = "float") -> Any:
    """Convert :class:`Quaternion` ``(x, y, z, w)`` to a real-first ``Gf.Quat*``.

    ``Gf.Quatf``/``Gf.Quatd`` take ``(real, i, j, k)`` — the opposite component
    order from the schema — and that is also how they serialize into ``.usda``
    (``quatf physics:localRot0 = (w, x, y, z)``). Getting this backwards silently
    produces a different rotation, so the conversion lives here and nowhere else.
    """
    pxr = require_pxr()
    cls = pxr.Gf.Quatf if precision == "float" else pxr.Gf.Quatd
    return cls(float(quat.w), float(quat.x), float(quat.y), float(quat.z))


def gf_to_quaternion(value: Any) -> Quaternion:
    """Convert a real-first ``Gf.Quat*`` to :class:`Quaternion` ``(x, y, z, w)``."""
    imaginary = value.GetImaginary()
    return Quaternion(
        x=float(imaginary[0]),
        y=float(imaginary[1]),
        z=float(imaginary[2]),
        w=float(value.GetReal()),
    )


# --------------------------------------------------------------------------
# Optional custom-attribute authoring / reading
# --------------------------------------------------------------------------


def _value_type(kind: str) -> Any:
    pxr = require_pxr()
    names = pxr.Sdf.ValueTypeNames
    return {
        "double": names.Double,
        "bool": names.Bool,
        "int": names.Int,
        "string": names.String,
        "token": names.Token,
        "double2": names.Double2,
        "double3": names.Double3,
        "double4": names.Double4,
        "doubleArray": names.DoubleArray,
    }[kind]


def _authored_attribute(prim: Any, name: str) -> Any | None:
    attribute = prim.GetAttribute(name)
    if not attribute or not attribute.HasAuthoredValue():
        return None
    return attribute


def write_double(prim: Any, name: str, value: float | None) -> None:
    """Author a ``double`` custom attribute, skipping ``None``."""
    if value is None:
        return
    prim.CreateAttribute(name, _value_type("double"), custom=True).Set(float(value))


def write_bool(prim: Any, name: str, value: bool | None) -> None:
    """Author a ``bool`` custom attribute, skipping ``None``."""
    if value is None:
        return
    prim.CreateAttribute(name, _value_type("bool"), custom=True).Set(bool(value))


def write_int(prim: Any, name: str, value: int | None) -> None:
    """Author an ``int`` custom attribute, skipping ``None``."""
    if value is None:
        return
    prim.CreateAttribute(name, _value_type("int"), custom=True).Set(int(value))


def write_string(prim: Any, name: str, value: str | None) -> None:
    """Author a ``string`` custom attribute, skipping ``None``."""
    if value is None:
        return
    prim.CreateAttribute(name, _value_type("string"), custom=True).Set(str(value))


def write_token(prim: Any, name: str, value: str | None) -> None:
    """Author a ``token`` custom attribute, skipping ``None``."""
    if value is None:
        return
    prim.CreateAttribute(name, _value_type("token"), custom=True).Set(str(value))


def write_vector3(prim: Any, name: str, value: Vector3 | None) -> None:
    """Author a ``double3`` custom attribute from a :class:`Vector3`."""
    if value is None:
        return
    prim.CreateAttribute(name, _value_type("double3"), custom=True).Set(
        vector3_to_gf(value)
    )


def write_quaternion(prim: Any, name: str, value: Quaternion | None) -> None:
    """Author a ``double4`` custom attribute as ``(x, y, z, w)``.

    Schema component order on purpose, not USD's real-first order: this sidecar
    exists to reproduce the dataclass exactly, so it mirrors the dataclass.
    """
    if value is None:
        return
    pxr = require_pxr()
    prim.CreateAttribute(name, _value_type("double4"), custom=True).Set(
        pxr.Gf.Vec4d(float(value.x), float(value.y), float(value.z), float(value.w))
    )


def write_float_pair(
    prim: Any, name: str, value: tuple[float, float] | list[float] | None
) -> None:
    """Author a ``double2`` custom attribute, skipping ``None``."""
    if value is None:
        return
    pxr = require_pxr()
    prim.CreateAttribute(name, _value_type("double2"), custom=True).Set(
        pxr.Gf.Vec2d(float(value[0]), float(value[1]))
    )


def write_float_list(prim: Any, name: str, value: list[float] | None) -> None:
    """Author a ``double[]`` custom attribute, skipping ``None``.

    An empty list is authored as an empty array (distinguishable from ``None``,
    which is not authored at all).
    """
    if value is None:
        return
    pxr = require_pxr()
    prim.CreateAttribute(name, _value_type("doubleArray"), custom=True).Set(
        pxr.Vt.DoubleArray([float(v) for v in value])
    )


def read_double(prim: Any, name: str) -> float | None:
    """Read a ``double``-ish custom attribute, or ``None`` if unauthored."""
    attribute = _authored_attribute(prim, name)
    return None if attribute is None else float(attribute.Get())


def read_bool(prim: Any, name: str) -> bool | None:
    """Read a ``bool`` custom attribute, or ``None`` if unauthored."""
    attribute = _authored_attribute(prim, name)
    return None if attribute is None else bool(attribute.Get())


def read_int(prim: Any, name: str) -> int | None:
    """Read an ``int`` custom attribute, or ``None`` if unauthored."""
    attribute = _authored_attribute(prim, name)
    return None if attribute is None else int(attribute.Get())


def read_string(prim: Any, name: str) -> str | None:
    """Read a ``string``/``token`` custom attribute, or ``None`` if unauthored."""
    attribute = _authored_attribute(prim, name)
    return None if attribute is None else str(attribute.Get())


def read_vector3(prim: Any, name: str) -> Vector3 | None:
    """Read a ``double3`` custom attribute as :class:`Vector3`."""
    attribute = _authored_attribute(prim, name)
    return None if attribute is None else gf_to_vector3(attribute.Get())


def read_quaternion(prim: Any, name: str) -> Quaternion | None:
    """Read a ``double4`` ``(x, y, z, w)`` custom attribute as :class:`Quaternion`."""
    attribute = _authored_attribute(prim, name)
    if attribute is None:
        return None
    value = attribute.Get()
    return Quaternion(
        x=float(value[0]), y=float(value[1]), z=float(value[2]), w=float(value[3])
    )


def read_float_pair(prim: Any, name: str) -> tuple[float, float] | None:
    """Read a ``double2`` custom attribute as a **tuple**.

    A tuple, not a list: ``Actuator.control_range`` and ``force_range`` are typed
    ``tuple[float, float]`` and ``[a, b] != (a, b)``, so returning a list would
    silently break round-trip equality.
    """
    attribute = _authored_attribute(prim, name)
    if attribute is None:
        return None
    value = attribute.Get()
    return (float(value[0]), float(value[1]))


def read_float_list(prim: Any, name: str) -> list[float] | None:
    """Read a ``double[]`` custom attribute as a list of floats."""
    attribute = _authored_attribute(prim, name)
    if attribute is None:
        return None
    return [float(v) for v in attribute.Get()]


def read_native(attribute: Any) -> Any | None:
    """Return a *native schema* attribute's value, or ``None`` if unauthored.

    Native UsdPhysics attributes carry meaningful fallbacks that must not be
    mistaken for data — ``physics:mass = 0`` means "derive from density",
    ``physics:centerOfMass = (-inf, -inf, -inf)`` means "derive from geometry",
    ``physics:lowerLimit = -inf`` means "unlimited". Gating on
    ``HasAuthoredValue()`` keeps those sentinels out of the schema.
    """
    if not attribute or not attribute.HasAuthoredValue():
        return None
    return attribute.Get()


# --------------------------------------------------------------------------
# JSON payloads (extensions, noise, parameters, calibration, ...)
# --------------------------------------------------------------------------


def _json_default(obj: Any) -> Any:
    """Fall back to cattrs for values JSON cannot encode directly.

    Needed because both ``URDFParser`` and ``MJCFParser`` stash a
    ``parse_context`` into ``CommonSchema.extensions`` that contains live
    ``Material`` dataclass instances, so a plain ``json.dumps`` would raise on the
    very first ``URDFParser -> USDExporter`` conversion. Converting them via the
    package's own cattrs converter keeps the conversion working; the cost is a
    genuine asymmetry (a dataclass goes out and a dict comes back), which is why
    the round-trip tests drop ``parse_context`` as parser diagnostics rather than
    model data.
    """
    from cyberwave_robot_format.serialization import converter

    try:
        unstructured = converter.unstructure(obj)
    except Exception as exc:
        raise TypeError(
            f"Value of type {type(obj).__name__!r} inside extensions is not "
            "JSON-serializable and could not be unstructured."
        ) from exc

    # cattrs returns unregistered objects unchanged rather than raising, which
    # would send json.dumps straight back here and surface as a confusing
    # "Circular reference detected" instead of naming the offending type.
    if unstructured is obj and not isinstance(
        obj, (dict, list, tuple, str, int, float, bool)
    ):
        raise TypeError(
            f"Value of type {type(obj).__name__!r} inside extensions is not "
            "JSON-serializable and has no cattrs unstructure hook."
        )
    return unstructured


# Bound decoding independently of CPython's changing recursion limits. This
# applies only to embedded USD metadata, not the scene hierarchy or mesh data.
_JSON_MAX_DEPTH = 256


def _check_json_nesting(raw: str) -> None:
    """Reject excessive nesting before allocating the decoded object tree."""
    depth = 0
    quoted = False
    escaped = False
    for char in raw:
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "[{":
            depth += 1
            if depth > _JSON_MAX_DEPTH:
                raise RecursionError("USD JSON metadata is nested too deeply")
        elif char in "]}":
            depth -= 1


def _read_json_metadata(raw: Any, location: str, prim: Any) -> Any | None:
    if raw is None:
        return None
    try:
        if isinstance(raw, str):
            _check_json_nesting(raw)
        return json.loads(raw)
    except (ValueError, TypeError):
        logger.warning("%s on %s is not valid JSON; ignoring it.", location, prim.GetPath())
    except RecursionError:
        logger.warning(
            "%s on %s is nested too deeply to decode; ignoring it.", location, prim.GetPath()
        )
    return None


def dumps_json(value: Any) -> str:
    """Serialize a JSON-ish payload deterministically, preserving key order.

    Keys are deliberately **not** sorted. Dict ordering inside ``extensions`` is
    observable downstream — ``MJCFExporter`` expands ``extensions["mjcf_size"]``
    straight into XML attributes, so reordering it changes the emitted
    ``<size .../>`` — and a USD hop must not perturb that. Determinism does not
    require sorting: a given input dict has a fixed insertion order, and
    ``json.loads`` rebuilds it in document order, so export -> parse -> export is
    still byte-stable.

    The compact separators keep the embedded payloads readable in the ``.usda``.
    """
    try:
        raw = json.dumps(value, separators=(",", ":"), default=_json_default)
        _check_json_nesting(raw)
    except RecursionError as exc:
        raise ValueError(f"USD JSON metadata exceeds {_JSON_MAX_DEPTH} nesting levels") from exc
    return raw


def write_json_attr(prim: Any, name: str, value: Any) -> None:
    """Author a JSON payload as a ``string`` attribute, skipping ``None``.

    A JSON string rather than a USD ``dictionary``: USD's value model cannot
    represent ``None`` at all, cannot hold heterogeneous or empty lists
    (``VtArray`` is homogeneous and needs an element type), and silently coerces
    ``int``/``int64`` and ``tuple``/array. JSON round-trips every type
    ``CommonSchema.to_dict()`` can already produce.
    """
    if value is None:
        return
    write_string(prim, name, dumps_json(value))


def read_json_attr(prim: Any, name: str) -> Any | None:
    """Read a JSON payload from a ``string`` attribute, or ``None``."""
    return _read_json_metadata(read_string(prim, name), f"Attribute {name}", prim)


def write_custom_data_json(prim: Any, key: str, value: Any) -> None:
    """Store a JSON payload in ``customData`` under ``key``, skipping empties.

    ``key`` may contain ``:``, which USD reads as a dictionary path — so
    ``"cyberwave:extensions"`` nests as ``{"cyberwave": {"extensions": ...}}`` and
    reads back through the same key.
    """
    if value is None or (isinstance(value, dict) and not value):
        return
    prim.SetCustomDataByKey(key, dumps_json(value))


def _to_custom_data_value(value: Any) -> Any | None:
    """Convert a scalar/list/dict payload into the value types USD stores natively."""
    vt = require_pxr().Vt
    if value is None:
        return None
    if isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        converted = {
            str(key): item
            for key, item in ((k, _to_custom_data_value(v)) for k, v in value.items())
            if item is not None
        }
        return converted or None
    if isinstance(value, list | tuple):
        if all(isinstance(item, str) for item in value):
            return vt.StringArray([str(item) for item in value])
        if all(
            isinstance(item, int | float) and not isinstance(item, bool)
            for item in value
        ):
            return vt.DoubleArray([float(item) for item in value])
    raise TypeError(f"Unsupported customData value: {value!r}")


def write_custom_data_value(prim: Any, key: str, value: Any) -> None:
    """Store a plain USD value (not JSON) in ``customData`` under ``key``.

    Used for the portable identity keys (``cyberwave:linkName``,
    ``cyberwave:sensorParams`` ...) that a non-Cyberwave consumer reads directly.
    Strings, ints, bools and finite floats are stored as-is, number/string lists
    as ``double[]``/``string[]``, dicts as nested USD dictionaries (which
    ``.usda`` writes in sorted key order). ``None`` and empty dicts are skipped.
    """
    converted = _to_custom_data_value(value)
    if converted is None:
        return
    prim.SetCustomDataByKey(key, converted)


def read_custom_data_json(prim: Any, key: str) -> Any | None:
    """Read a JSON payload from ``customData``, or ``None`` if absent/invalid."""
    return _read_json_metadata(prim.GetCustomDataByKey(key), f"customData[{key}]", prim)


def read_extensions(prim: Any, key: str) -> dict[str, Any]:
    """Read an ``extensions`` dict from ``customData``, defaulting to ``{}``."""
    value = read_custom_data_json(prim, key)
    return value if isinstance(value, dict) else {}


# --------------------------------------------------------------------------
# UsdPhysics-native properties with no schema field
# --------------------------------------------------------------------------


def _native_value_type(kind: str) -> Any:
    pxr = require_pxr()
    names = pxr.Sdf.ValueTypeNames
    return {
        "bool": names.Bool,
        "float": names.Float,
        "double": names.Double,
        "string": names.String,
        "vector3f": names.Vector3f,
    }[kind]


def read_usd_passthrough(
    prim: Any, table: tuple[tuple[str, str, str], ...]
) -> dict[str, Any]:
    """Harvest authored UsdPhysics properties that the schema cannot express.

    Returns a JSON-able dict keyed by the short name in ``table``, containing only
    the attributes actually authored on ``prim`` — so an unauthored ``breakForce``
    (whose fallback is ``inf``) contributes nothing.
    """
    harvested: dict[str, Any] = {}
    for key, attribute_name, kind in table:
        attribute = prim.GetAttribute(attribute_name)
        if not attribute or not attribute.HasAuthoredValue():
            continue
        value = attribute.Get()
        if value is None:
            continue
        if kind == "bool":
            harvested[key] = bool(value)
        elif kind in ("float", "double"):
            number = float(value)
            if not math.isfinite(number):
                # inf/-inf are UsdPhysics "no limit" fallbacks, not data.
                continue
            harvested[key] = number
        elif kind == "vector3f":
            harvested[key] = [float(value[0]), float(value[1]), float(value[2])]
        else:
            harvested[key] = str(value)
    return harvested


def write_usd_passthrough(
    prim: Any, table: tuple[tuple[str, str, str], ...], extensions: dict[str, Any]
) -> None:
    """Author back the UsdPhysics properties stored under ``extensions["usd"]``.

    Written as schema-declared (non-``custom``) attributes so they land in the
    ``physics:`` namespace a USD consumer reads, rather than as a second copy in the
    ``cyberwave:`` namespace.
    """
    from cyberwave_robot_format.usd.tokens import USD_EXTENSION_KEY

    stored = extensions.get(USD_EXTENSION_KEY)
    if not isinstance(stored, dict):
        return
    pxr = require_pxr()
    for key, attribute_name, kind in table:
        if key not in stored:
            continue
        value = stored[key]
        attribute = prim.CreateAttribute(
            attribute_name, _native_value_type(kind), custom=False
        )
        if kind == "bool":
            attribute.Set(bool(value))
        elif kind in ("float", "double"):
            attribute.Set(float(value))
        elif kind == "vector3f":
            attribute.Set(pxr.Gf.Vec3f(*[float(v) for v in value]))
        else:
            attribute.Set(str(value))


def merge_usd_passthrough(
    extensions: dict[str, Any], harvested: dict[str, Any]
) -> dict[str, Any]:
    """Fold harvested USD properties into an ``extensions`` dict under ``"usd"``.

    The stored ``customData`` copy is the base and the harvested native attributes
    overlay it. For a Cyberwave-written stage the two agree by construction (the
    natives were derived from the stored dict), so the merge is a no-op there.
    """
    from cyberwave_robot_format.usd.tokens import USD_EXTENSION_KEY

    if not harvested:
        return extensions
    existing = extensions.get(USD_EXTENSION_KEY)
    merged = dict(existing) if isinstance(existing, dict) else {}
    merged.update(harvested)
    extensions[USD_EXTENSION_KEY] = merged
    return extensions


__all__ = [
    "PrimNamer",
    "dumps_json",
    "gf_matrix_to_np",
    "gf_to_quaternion",
    "gf_to_vector3",
    "merge_usd_passthrough",
    "np_to_gf_matrix",
    "quaternion_to_gf",
    "read_bool",
    "read_custom_data_json",
    "read_double",
    "read_extensions",
    "read_float_list",
    "read_float_pair",
    "read_int",
    "read_json_attr",
    "read_native",
    "read_quaternion",
    "read_string",
    "read_usd_passthrough",
    "read_vector3",
    "require_pxr",
    "sanitize_prim_name",
    "schema_name_from_prim",
    "vector3_to_gf",
    "write_bool",
    "write_custom_data_json",
    "write_double",
    "write_float_list",
    "write_float_pair",
    "write_int",
    "write_json_attr",
    "write_quaternion",
    "write_string",
    "write_token",
    "write_usd_passthrough",
    "write_vector3",
]
