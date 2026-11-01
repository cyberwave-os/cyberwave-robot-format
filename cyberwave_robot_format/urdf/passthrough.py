# Copyright [2025] Cyberwave
# Licensed under the Apache License, Version 2.0

"""Verbatim capture and replay of URDF content the schema cannot type.

A URDF carries more than kinematics. ``<ros2_control>`` declares the hardware
plugin and per-joint command/state interfaces, ``<transmission>`` describes the
actuator coupling, and ``<gazebo>`` carries simulation plugins. The schema models
none of that, so a parse that ignored them followed by an export that regenerated
the file from the schema silently deleted a robot's entire control configuration.

The fix is a passthrough section: every top-level element the schema does not own
is stored verbatim as XML text under ``schema.extensions["urdf_passthrough"]`` and
re-emitted by the exporter. That gives two properties:

* an untouched block round-trips URDF -> schema -> URDF unchanged, and
* a block *edited in the schema* exports edited, because the exporter renders only
  what the schema holds. The schema stays the single source of truth even for
  content it cannot interpret.

Each entry carries a ``key`` so a JSON-Pointer patch can address one block out of
several (``/extensions/urdf_passthrough/ros2_control/0/xml``). Keys come from the
element's natural identity (``name`` for ros2_control/transmission, ``reference``
for gazebo) and are advisory: position in the list is what identifies an entry.

Two known limitations, both deliberate:

* ``xml.etree`` discards comments, so comments *inside* a captured block are lost.
  Preserving them needs a comment-aware parser and would change every other parse
  path in this package.
* A ``<gazebo><sensor>`` block is both captured here and parsed into
  ``schema.sensors``. The captured block is the Gazebo-side truth; ``schema.sensors``
  is the Cyberwave/MJCF-side truth. Editing one does not update the other — see
  :func:`gazebo_sensor_drift` for the check that says so out loud.

Scope note: passthrough is *per-asset*. The block text references joint and link
names verbatim, and :meth:`CommonSchema.merge_in` namespaces those names when it
composes a scene, so replaying a captured block into a composed scene would point a
robot's controllers at joints that no longer exist under those names. ``merge_in``
deliberately does not propagate ``extensions``, which is what keeps that from
happening; a composed scene therefore carries no passthrough at all.
"""

from __future__ import annotations

import copy
import logging
import xml.etree.ElementTree as ET
from typing import Any

from cyberwave_robot_format.utils import sanitize_name

logger = logging.getLogger(__name__)

EXTENSION_KEY = "urdf_passthrough"

# Top-level ``<robot>`` children the schema models natively. Everything else is
# passthrough. ``gazebo`` is deliberately NOT here: sensors are lifted out of it
# into ``schema.sensors``, but the block also carries plugins the schema cannot
# express, so the whole block is captured verbatim as well.
_SCHEMA_OWNED_TAGS = frozenset({"link", "joint", "material"})

# Sections with a dedicated bucket, in the order they are re-emitted. Anything
# else lands in ``unknown`` rather than being dropped.
NAMED_SECTIONS: tuple[str, ...] = ("ros2_control", "transmission", "gazebo")
UNKNOWN_SECTION = "unknown"
SECTIONS: tuple[str, ...] = (*NAMED_SECTIONS, UNKNOWN_SECTION)

# Attribute that carries an element's natural identity, per section.
_KEY_ATTRIBUTE: dict[str, str] = {
    "ros2_control": "name",
    "transmission": "name",
    "gazebo": "reference",
}


def _strip_formatting_whitespace(elem: ET.Element) -> None:
    """Drop indentation-only text nodes from ``elem``, recursively, in place.

    Capture must be *canonical*, not merely faithful. XML keeps the indentation
    between tags as text nodes; the exporter pretty-prints the document it builds,
    which adds indentation around those existing nodes. Store the raw string and
    every rebuild -> export cycle nests one more level of blank lines inside each
    block, so a robot's ``<ros2_control>`` grows without bound and no two exports of
    an unchanged schema are ever equal.

    An element with children cannot have meaningful text of its own in URDF/SDF, so
    its ``text`` is formatting. A leaf's text is its *value*, so it is kept — only
    the surrounding whitespace goes.
    """
    children = list(elem)
    if children:
        elem.text = None
    elif elem.text is not None:
        stripped = elem.text.strip()
        elem.text = stripped or None
    for child in children:
        child.tail = None
        _strip_formatting_whitespace(child)


# Attributes inside a captured block that name a link or joint the schema also owns.
# ``<gazebo reference="...">`` attaches to a link; ``<ros2_control><joint name>`` and
# ``<transmission><joint name>`` bind to a joint. Anything else in these blocks is
# plugin configuration the schema does not model and must stay verbatim.
_NAME_REFERENCE_ATTRIBUTES: tuple[tuple[str, str], ...] = (
    ("gazebo", "reference"),
    ("joint", "name"),
    ("link", "name"),
)


def _sanitize_name_references(elem: ET.Element) -> None:
    """Rewrite link/joint references in ``elem`` the way the parser rewrote the names.

    The parser runs every link and joint name through ``sanitize_name``, which rewrites
    ``.``, ``/``, ``:`` and spaces — all legal in URDF and routinely produced by xacro
    namespaces (``ur5:shoulder``). Captured blocks are stored verbatim, so a
    ``<transmission>`` or ``<ros2_control>`` entry kept pointing at the pre-sanitized
    spelling while the exported ``<joint>`` carried the sanitized one: the replayed
    block referenced a joint that does not exist in the file it ships in, and
    ros2_control fails to load the controller.

    Only the attributes in :data:`_NAME_REFERENCE_ATTRIBUTES` are touched, and only
    when sanitizing actually changes the value — a robot whose names need no
    sanitizing (almost all of them) still round-trips byte-identical.
    """
    for tag, attribute in _NAME_REFERENCE_ATTRIBUTES:
        candidates = [elem] if elem.tag == tag else []
        candidates.extend(elem.iter(tag))
        for node in candidates:
            value = node.get(attribute)
            if not value:
                continue
            sanitized = sanitize_name(value)
            if sanitized != value:
                node.set(attribute, sanitized)


def _element_to_xml(elem: ET.Element) -> str:
    """Serialize ``elem`` canonically: content preserved, formatting normalized.

    Serialized from a copy so the caller's parse tree is untouched.
    """
    clone = copy.deepcopy(elem)
    clone.tail = None
    _strip_formatting_whitespace(clone)
    _sanitize_name_references(clone)
    return ET.tostring(clone, encoding="unicode")


def capture_urdf_passthrough(root: ET.Element) -> dict[str, list[dict[str, Any]]]:
    """Return the passthrough section for a parsed ``<robot>`` element.

    Only direct children are considered: nested content travels with its parent
    block. Document order is preserved within each section.
    """
    captured: dict[str, list[dict[str, Any]]] = {}

    for index, elem in enumerate(root):
        tag = elem.tag
        if not isinstance(tag, str):
            continue  # comments/PIs surface as callables, not tag names
        if tag in _SCHEMA_OWNED_TAGS:
            continue

        section = tag if tag in NAMED_SECTIONS else UNKNOWN_SECTION
        entry: dict[str, Any] = {"xml": _element_to_xml(elem)}

        key_attribute = _KEY_ATTRIBUTE.get(section)
        key = elem.get(key_attribute) if key_attribute else None
        entry["key"] = key or f"idx{index}"
        if section == UNKNOWN_SECTION:
            # The bucket is heterogeneous, so the tag is the only way a consumer
            # (or a reviewer reading the stored schema) can tell entries apart.
            entry["tag"] = tag

        captured.setdefault(section, []).append(entry)

    if captured:
        logger.debug(
            "Captured URDF passthrough: %s",
            {section: len(entries) for section, entries in captured.items()},
        )
    return captured


def passthrough_from_extensions(extensions: Any) -> dict[str, list[dict[str, Any]]]:
    """Read the passthrough section out of a schema ``extensions`` mapping.

    Tolerant of a missing or malformed section (returns ``{}``) because it may have
    been hand-authored by a patch. Malformed *entries* are not tolerated — see
    :func:`render_urdf_passthrough`.
    """
    if not isinstance(extensions, dict):
        return {}
    section = extensions.get(EXTENSION_KEY)
    if not isinstance(section, dict):
        return {}
    return {
        name: entries
        for name, entries in section.items()
        if isinstance(entries, list) and entries
    }


def render_urdf_passthrough(
    robot: ET.Element,
    passthrough: dict[str, list[dict[str, Any]]],
) -> int:
    """Append every passthrough block to ``robot``, returning how many were added.

    Raises ``ValueError`` on an entry that is not well-formed XML. A malformed
    entry means authored data is corrupt, and emitting a URDF that is silently
    missing a robot's ``ros2_control`` block is worse than failing the export —
    the robot would come up with no controllers and no explanation. Callers that
    must not fail (the served-URDF regeneration) already swallow-and-log.
    """
    emitted = 0
    for section in (*NAMED_SECTIONS, UNKNOWN_SECTION):
        for position, entry in enumerate(passthrough.get(section) or []):
            if not isinstance(entry, dict):
                raise ValueError(
                    f"urdf_passthrough/{section}/{position} must be an object with an 'xml' key"
                )
            xml = entry.get("xml")
            if not isinstance(xml, str) or not xml.strip():
                raise ValueError(
                    f"urdf_passthrough/{section}/{position} (key={entry.get('key')!r}) "
                    "has no 'xml' string"
                )
            try:
                element = ET.fromstring(xml)
            except ET.ParseError as exc:
                raise ValueError(
                    f"urdf_passthrough/{section}/{position} (key={entry.get('key')!r}) "
                    f"is not well-formed XML: {exc}"
                ) from exc
            robot.append(element)
            emitted += 1
    return emitted


def gazebo_sensor_drift(
    passthrough: dict[str, list[dict[str, Any]]],
    sensor_names: set[str],
) -> tuple[str, ...]:
    """Return sensor names present in captured ``<gazebo>`` XML but not in the schema.

    A one-way informational check for the documented boundary: ``schema.sensors``
    drives Cyberwave and the MJCF export, the captured ``<gazebo>`` block drives
    Gazebo, and nothing keeps them in step. Naming the divergence beats discovering
    it when a simulated camera publishes nothing.
    """
    drifted: list[str] = []
    for entry in passthrough.get("gazebo") or []:
        xml = entry.get("xml") if isinstance(entry, dict) else None
        if not isinstance(xml, str):
            continue
        try:
            element = ET.fromstring(xml)
        except ET.ParseError:
            continue
        for sensor in (*element.findall("sensor"), *element.findall("sensor_camera")):
            name = sensor.get("name")
            # Compare on the sanitized spelling: ``schema.sensors`` holds names the
            # parser already ran through ``sanitize_name``, so a raw comparison
            # reported every sensor whose name merely needed sanitizing as "absent
            # from the schema" — a false alarm in a check whose whole value is that
            # it only fires on real divergence.
            if name and sanitize_name(name) not in sensor_names:
                drifted.append(name)
    return tuple(drifted)


__all__ = [
    "EXTENSION_KEY",
    "NAMED_SECTIONS",
    "SECTIONS",
    "UNKNOWN_SECTION",
    "capture_urdf_passthrough",
    "gazebo_sensor_drift",
    "passthrough_from_extensions",
    "render_urdf_passthrough",
]
