"""Canonical field-of-view naming and unit convention for this package.

Full policy: cyberwave-backend/docs/FOV_NAMING_CONVENTION.md. This module is a
self-contained local copy of the same recognition rule, not an import of it --
this package does not depend on cyberwave-backend.

The gold standard, one stored/interchanged key per meaningful axis, radians:

|            | Radians    | Degrees    |
|------------|------------|------------|
| Horizontal | hfov_rad   | hfov_deg   |
| Vertical   | vfov_rad   | vfov_deg   |
| Diagonal   | dfov_rad   | dfov_deg   |

The axis and unit of a value are properties of its *name*, never inferred
from its *magnitude* -- a magnitude-based heuristic misreads a legitimate
narrow FOV, which is the historical bug this convention exists to prevent.
"""

from __future__ import annotations

import math
from typing import Literal, NamedTuple

Axis = Literal["horizontal", "vertical", "diagonal"]
Unit = Literal["rad", "deg"]


class FovKey(NamedTuple):
    axis: Axis
    unit: Unit


CANONICAL_KEY: dict[tuple[Axis, Unit], str] = {
    ("horizontal", "rad"): "hfov_rad",
    ("horizontal", "deg"): "hfov_deg",
    ("vertical", "rad"): "vfov_rad",
    ("vertical", "deg"): "vfov_deg",
    ("diagonal", "rad"): "dfov_rad",
    ("diagonal", "deg"): "dfov_deg",
}

# Cross-checked against sensor_fov.py / CONVENTION.md §7, which this package
# cannot import. `vfov` is DEGREES here -- unlike hfov/horizontal_fov, which
# are already radians -- and that asymmetry is deliberate, not a typo.
KNOWN_FOV_KEYS: dict[str, FovKey] = {
    "hfov_rad": FovKey("horizontal", "rad"),
    "vfov_rad": FovKey("vertical", "rad"),
    "dfov_rad": FovKey("diagonal", "rad"),
    "hfov_deg": FovKey("horizontal", "deg"),
    "vfov_deg": FovKey("vertical", "deg"),
    "dfov_deg": FovKey("diagonal", "deg"),
    "fovy": FovKey("vertical", "deg"),
    "fov_degrees": FovKey("vertical", "deg"),
    "fov_vertical": FovKey("vertical", "deg"),
    "vertical_fov": FovKey("vertical", "deg"),
    "vertical_fov_degrees": FovKey("vertical", "deg"),
    "vfov": FovKey("vertical", "deg"),
    "fov_horizontal": FovKey("horizontal", "deg"),
    "horizontal_fov_degrees": FovKey("horizontal", "deg"),
    "diagonal_fov_deg": FovKey("diagonal", "deg"),
    "apriltag_diagonal_fov_deg": FovKey("diagonal", "deg"),
    "camera_diagonal_fov_deg": FovKey("diagonal", "deg"),
    "horizontal_fov": FovKey("horizontal", "rad"),
    "hfov": FovKey("horizontal", "rad"),
}

# Fallback for a name with no separator at all (e.g. "HorizontalFOV"), which
# the primary lookup's case/hyphen/space normalization does not catch.
_STRIPPED_LOOKUP: dict[str, FovKey] = {
    key.replace("_", ""): value for key, value in KNOWN_FOV_KEYS.items()
}


def _lookup_name(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def normalize_fov_key(name: str, value: float) -> tuple[str, float] | None:
    """Identify an arbitrarily-spelled (case/separator-insensitive) FOV name
    and return its canonical ``(key, radians)`` pair, or ``None`` if
    unrecognized — never a guess; axis and unit come from the name, not the
    value."""
    normalized = _lookup_name(name)
    key = KNOWN_FOV_KEYS.get(normalized) or _STRIPPED_LOOKUP.get(
        normalized.replace("_", "")
    )
    if key is None:
        return None
    canonical_name = CANONICAL_KEY[(key.axis, "rad")]
    radians = value if key.unit == "rad" else math.radians(value)
    return canonical_name, radians


def find_fov_value(
    params: dict[str, float], candidate_names: tuple[str, ...]
) -> tuple[str, float] | None:
    """Return the canonical ``(key, radians)`` pair for the first of
    ``candidate_names`` present in ``params``, or ``None``."""
    for name in candidate_names:
        if name in params:
            resolved = normalize_fov_key(name, params[name])
            if resolved is not None:
                return resolved
    return None
