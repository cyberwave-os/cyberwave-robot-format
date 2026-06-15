"""Infer and annotate URDF mimic joints from kinematic structure.

Detects coupled finger / gripper joint pairs that are not explicitly tagged with
``<mimic>`` in the source URDF. When confidence is high enough, writes a new
``{stem}-mimic-joint.urdf`` file without modifying the original.
"""

from __future__ import annotations

import logging
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import defusedxml.ElementTree as DefusedET

from cyberwave_robot_format.schema import CommonSchema, Joint, JointType
from cyberwave_robot_format.urdf.parser import URDFParser

logger = logging.getLogger(__name__)

CONTROLLABLE_TYPES = {JointType.REVOLUTE, JointType.PRISMATIC, JointType.CONTINUOUS}

DEFAULT_MIN_CONFIDENCE = 0.85
MIMIC_URDF_SUFFIX = "-mimic-joint"

_GRIPPER_NAME_RE = re.compile(
    r"(gripper|finger|jaw|claw|knuckle|slider|mimic|ee_|eef)",
    re.IGNORECASE,
)


@dataclass
class InferredMimic:
    """A single inferred or existing mimic relationship."""

    driver_joint: str
    slave_joint: str
    multiplier: float = 1.0
    offset: float = 0.0
    confidence: float = 1.0
    evidence: list[str] = field(default_factory=list)


@dataclass
class MimicInferenceResult:
    """Outcome of evaluating a URDF for mimic joints."""

    source_path: Path
    existing_mimics: list[InferredMimic]
    inferred_mimics: list[InferredMimic]
    should_patch: bool
    output_path: Path | None = None


def infer_mimic_joints(
    urdf_path: str | Path,
    *,
    schema: CommonSchema | None = None,
    mjcf_path: str | Path | None = None,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> MimicInferenceResult:
    """Evaluate a URDF and infer mimic joint pairs with confidence scores."""
    path = Path(urdf_path)
    if schema is None:
        schema = URDFParser().parse(path)

    existing = _existing_mimics_from_schema(schema)
    if existing:
        return MimicInferenceResult(
            source_path=path,
            existing_mimics=existing,
            inferred_mimics=[],
            should_patch=False,
        )

    mjcf_pairs = _infer_from_mjcf_equality(mjcf_path) if mjcf_path else []
    kinematic_pairs = _infer_from_kinematics(schema)

    inferred_by_slave: dict[str, InferredMimic] = {}
    for mimic in mjcf_pairs + kinematic_pairs:
        prev = inferred_by_slave.get(mimic.slave_joint)
        if prev is None or mimic.confidence > prev.confidence:
            inferred_by_slave[mimic.slave_joint] = mimic

    inferred = _resolve_conflicts(list(inferred_by_slave.values()))
    inferred = [m for m in inferred if m.confidence >= min_confidence]
    should_patch = len(inferred) > 0
    output_path = _default_output_path(path) if should_patch else None

    return MimicInferenceResult(
        source_path=path,
        existing_mimics=[],
        inferred_mimics=inferred,
        should_patch=should_patch,
        output_path=output_path,
    )


def write_mimic_patched_urdf(
    urdf_path: str | Path,
    inferred: list[InferredMimic],
    output_path: str | Path | None = None,
) -> Path:
    """Write a new URDF with ``<mimic>`` tags on slave joints. Never touches the input."""
    source = Path(urdf_path)
    if not inferred:
        raise ValueError("No inferred mimic joints to patch")

    dest = Path(output_path) if output_path else _default_output_path(source)
    tree = DefusedET.parse(source)
    root = tree.getroot()

    comment = ET.Comment(
        f" mimic joints inferred by cyberwave-robot-format; source={source.name}; pairs={len(inferred)} "
    )
    root.insert(0, comment)

    slaves = {m.slave_joint: m for m in inferred}
    for joint_elem in root.iter("joint"):
        name = joint_elem.get("name")
        if not name or name not in slaves:
            continue
        if joint_elem.find("mimic") is not None:
            continue

        mimic = slaves[name]
        mimic_elem = ET.Element("mimic")
        mimic_elem.set("joint", mimic.driver_joint)
        mimic_elem.set("multiplier", _format_float(mimic.multiplier))
        mimic_elem.set("offset", _format_float(mimic.offset))
        joint_elem.append(mimic_elem)

    dest.parent.mkdir(parents=True, exist_ok=True)
    tree.write(dest, encoding="utf-8", xml_declaration=True)
    logger.info("Wrote mimic-patched URDF to %s", dest)
    return dest


def infer_and_patch_if_needed(
    urdf_path: str | Path,
    *,
    mjcf_path: str | Path | None = None,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    output_path: str | Path | None = None,
) -> MimicInferenceResult:
    """Full pipeline: infer mimic pairs and write patched URDF when confident."""
    result = infer_mimic_joints(
        urdf_path,
        mjcf_path=mjcf_path,
        min_confidence=min_confidence,
    )
    if result.should_patch:
        dest = write_mimic_patched_urdf(
            urdf_path,
            result.inferred_mimics,
            output_path=output_path or result.output_path,
        )
        result.output_path = dest
    return result


def _default_output_path(source: Path) -> Path:
    return source.with_name(f"{source.stem}{MIMIC_URDF_SUFFIX}{source.suffix}")


def _format_float(value: float) -> str:
    if value == int(value):
        return str(int(value))
    return f"{value:g}"


def _existing_mimics_from_schema(schema: CommonSchema) -> list[InferredMimic]:
    mimics: list[InferredMimic] = []
    for joint in schema.joints:
        if joint.mimic is None:
            continue
        mimics.append(
            InferredMimic(
                driver_joint=joint.mimic.joint,
                slave_joint=joint.name,
                multiplier=joint.mimic.multiplier,
                offset=joint.mimic.offset,
                confidence=1.0,
                evidence=["explicit_urdf_mimic"],
            )
        )
    return mimics


def _infer_from_mjcf_equality(mjcf_path: str | Path) -> list[InferredMimic]:
    """Map MuJoCo ``<equality><joint polycoef=.../>`` to linear URDF mimic."""
    path = Path(mjcf_path)
    if not path.is_file():
        return []

    root = DefusedET.parse(path).getroot()
    mimics: list[InferredMimic] = []
    for eq in root.findall(".//equality/joint"):
        j1 = eq.get("joint1")
        j2 = eq.get("joint2")
        polycoef = eq.get("polycoef", "")
        if not j1 or not j2:
            continue

        coeffs = [float(c) for c in polycoef.split()]
        if len(coeffs) < 2:
            continue

        offset, multiplier = coeffs[0], coeffs[1]
        if len(coeffs) > 2 and any(abs(c) > 1e-6 for c in coeffs[2:]):
            continue

        if abs(abs(multiplier) - 1.0) > 0.05:
            continue

        mimics.append(
            InferredMimic(
                driver_joint=_strip_mjcf_prefix(j2),
                slave_joint=_strip_mjcf_prefix(j1),
                multiplier=multiplier,
                offset=offset,
                confidence=0.95,
                evidence=["mjcf_equality_linear"],
            )
        )
    return mimics


def _strip_mjcf_prefix(name: str) -> str:
    if "_" in name:
        suffix = name.rsplit("_", 1)[-1]
        if suffix.startswith("joint"):
            return suffix
    return name


def _infer_from_kinematics(schema: CommonSchema) -> list[InferredMimic]:
    joints = [j for j in schema.joints if j.type in CONTROLLABLE_TYPES]
    if len(joints) < 2:
        return []

    by_parent: dict[str, list[Joint]] = {}
    for joint in joints:
        by_parent.setdefault(joint.parent_link, []).append(joint)

    actuator_joints = {a.joint for a in schema.actuators}
    tip_links = _tip_links(joints)
    arm_travels = sorted(_joint_travel(j) for j in joints if _joint_travel(j) > 0)
    median_arm_travel = arm_travels[len(arm_travels) // 2] if arm_travels else math.pi

    candidates: list[InferredMimic] = []
    for parent, siblings in by_parent.items():
        if len(siblings) < 2:
            continue
        for i, joint_a in enumerate(siblings):
            for joint_b in siblings[i + 1 :]:
                candidate = _score_joint_pair(
                    joint_a,
                    joint_b,
                    parent=parent,
                    actuator_joints=actuator_joints,
                    tip_links=tip_links,
                    median_arm_travel=median_arm_travel,
                )
                if candidate is not None:
                    candidates.append(candidate)
    return candidates


def _tip_links(controllable_joints: list[Joint]) -> set[str]:
    """Child links with no further controllable joints downstream."""
    parents_of_controllable = {j.parent_link for j in controllable_joints}
    return {j.child_link for j in controllable_joints if j.child_link not in parents_of_controllable}


def _joint_travel(joint: Joint) -> float:
    if joint.limits is None or joint.limits.lower is None or joint.limits.upper is None:
        return 0.0
    return abs(joint.limits.upper - joint.limits.lower)


def _axis_dot(joint_a: Joint, joint_b: Joint) -> float:
    ax = joint_a.axis
    bx = joint_b.axis
    mag_a = math.sqrt(ax.x**2 + ax.y**2 + ax.z**2)
    mag_b = math.sqrt(bx.x**2 + bx.y**2 + bx.z**2)
    if mag_a < 1e-9 or mag_b < 1e-9:
        return 0.0
    return (ax.x * bx.x + ax.y * bx.y + ax.z * bx.z) / (mag_a * mag_b)


def _complementary_limits(joint_a: Joint, joint_b: Joint, tol: float = 0.02) -> bool:
    if (
        joint_a.limits is None
        or joint_b.limits is None
        or joint_a.limits.lower is None
        or joint_a.limits.upper is None
        or joint_b.limits.lower is None
        or joint_b.limits.upper is None
    ):
        return False
    return (
        abs(joint_a.limits.lower + joint_b.limits.upper) <= tol
        and abs(joint_a.limits.upper + joint_b.limits.lower) <= tol
    )


def _symmetric_travel(joint_a: Joint, joint_b: Joint, tol: float = 0.02) -> bool:
    travel_a = _joint_travel(joint_a)
    travel_b = _joint_travel(joint_b)
    if travel_a <= 0 or travel_b <= 0:
        return False
    return abs(travel_a - travel_b) <= max(tol, 0.05 * max(travel_a, travel_b))


def _gripper_name_bonus(joint_a: Joint, joint_b: Joint) -> float:
    names = f"{joint_a.name} {joint_a.child_link} {joint_b.name} {joint_b.child_link}"
    return 0.05 if _GRIPPER_NAME_RE.search(names) else 0.0


def _estimate_mimic_params(
    driver: Joint,
    slave: Joint,
    axis_dot: float,
) -> tuple[float, float] | None:
    """Return (multiplier, offset) or None if no valid linear coupling."""
    # Opposing axes + complementary limits → canonical parallel-jaw coupling.
    if axis_dot < -0.9 and _complementary_limits(driver, slave):
        if _validate_mimic_limits(driver, slave, -1.0, 0.0):
            return -1.0, 0.0

    preferred_multipliers = (-1.0, 1.0) if axis_dot < 0 else (1.0, -1.0)
    if _complementary_limits(driver, slave):
        preferred_multipliers = (-1.0, 1.0)

    if (
        driver.limits is None
        or slave.limits is None
        or driver.limits.lower is None
        or driver.limits.upper is None
        or slave.limits.lower is None
        or slave.limits.upper is None
    ):
        return preferred_multipliers[0], 0.0

    best: tuple[float, float] | None = None
    best_offset_mag = float("inf")
    for mult in preferred_multipliers:
        for offset in _offset_candidates(driver, slave, mult):
            if not _validate_mimic_limits(driver, slave, mult, offset):
                continue
            if abs(offset) < best_offset_mag:
                best = (mult, offset)
                best_offset_mag = abs(offset)
    return best


def _offset_candidates(driver: Joint, slave: Joint, multiplier: float) -> list[float]:
    assert driver.limits and slave.limits
    assert driver.limits.lower is not None and driver.limits.upper is not None
    assert slave.limits.lower is not None and slave.limits.upper is not None

    driver_mid = 0.5 * (driver.limits.lower + driver.limits.upper)
    slave_mid = 0.5 * (slave.limits.lower + slave.limits.upper)
    fitted = slave_mid - multiplier * driver_mid
    return [0.0, fitted]


def _validate_mimic_limits(
    driver: Joint,
    slave: Joint,
    multiplier: float,
    offset: float,
    samples: int = 5,
) -> bool:
    assert driver.limits and slave.limits
    assert driver.limits.lower is not None and driver.limits.upper is not None
    assert slave.limits.lower is not None and slave.limits.upper is not None

    for i in range(samples):
        t = i / (samples - 1) if samples > 1 else 0.5
        driver_pos = driver.limits.lower + t * (driver.limits.upper - driver.limits.lower)
        slave_pos = multiplier * driver_pos + offset
        if slave_pos < slave.limits.lower - 1e-4 or slave_pos > slave.limits.upper + 1e-4:
            return False
    return True


def _pick_driver(
    joint_a: Joint,
    joint_b: Joint,
    actuator_joints: set[str],
) -> tuple[Joint, Joint]:
    a_actuated = joint_a.name in actuator_joints
    b_actuated = joint_b.name in actuator_joints
    if a_actuated and not b_actuated:
        return joint_a, joint_b
    if b_actuated and not a_actuated:
        return joint_b, joint_a

    # Prefer joint whose lower limit is closer to zero (active finger convention).
    a_lower = joint_a.limits.lower if joint_a.limits and joint_a.limits.lower is not None else 0.0
    b_lower = joint_b.limits.lower if joint_b.limits and joint_b.limits.lower is not None else 0.0
    if abs(a_lower) < abs(b_lower) - 1e-6:
        return joint_a, joint_b
    if abs(b_lower) < abs(a_lower) - 1e-6:
        return joint_b, joint_a

    return (joint_a, joint_b) if joint_a.name <= joint_b.name else (joint_b, joint_a)


def _score_joint_pair(
    joint_a: Joint,
    joint_b: Joint,
    *,
    parent: str,
    actuator_joints: set[str],
    tip_links: set[str],
    median_arm_travel: float,
) -> InferredMimic | None:
    if joint_a.type != joint_b.type:
        return None
    if joint_a.type not in {JointType.PRISMATIC, JointType.REVOLUTE}:
        return None

    dot = _axis_dot(joint_a, joint_b)
    parallel = abs(abs(dot) - 1.0) < 0.05
    if not parallel:
        return None

    driver, slave = _pick_driver(joint_a, joint_b, actuator_joints)
    mimic_params = _estimate_mimic_params(driver, slave, dot)
    if mimic_params is None:
        return None
    multiplier, offset = mimic_params

    evidence: list[str] = []
    score = 0.0

    if dot < -0.9:
        score += 0.30
        evidence.append("opposing_parallel_axes")
    elif dot > 0.9:
        score += 0.15
        evidence.append("aligned_parallel_axes")

    if _complementary_limits(driver, slave):
        score += 0.30
        evidence.append("complementary_limits")

    if _symmetric_travel(driver, slave):
        score += 0.10
        evidence.append("symmetric_travel")

    max_travel = max(_joint_travel(driver), _joint_travel(slave))
    if max_travel > 0 and max_travel < 0.25 * median_arm_travel:
        score += 0.15
        evidence.append("small_travel_vs_arm")

    if driver.child_link in tip_links and slave.child_link in tip_links:
        score += 0.20
        evidence.append("terminal_fork")

    driver_actuated = driver.name in actuator_joints
    slave_actuated = slave.name in actuator_joints
    if driver_actuated and not slave_actuated:
        score += 0.25
        evidence.append("actuator_gap")
    elif driver_actuated and slave_actuated:
        score -= 0.10

    name_bonus = _gripper_name_bonus(driver, slave)
    if name_bonus:
        score += name_bonus
        evidence.append("gripper_naming_hint")

    score = min(score, 1.0)
    if score < 0.70:
        return None

    return InferredMimic(
        driver_joint=driver.name,
        slave_joint=slave.name,
        multiplier=multiplier,
        offset=offset,
        confidence=score,
        evidence=evidence,
    )


def _resolve_conflicts(candidates: list[InferredMimic]) -> list[InferredMimic]:
    """Drop lower-confidence pairs that share a slave or create driver/slave cycles."""
    ordered = sorted(candidates, key=lambda m: m.confidence, reverse=True)
    accepted: list[InferredMimic] = []
    used_slaves: set[str] = set()
    used_drivers_as_slave: set[str] = set()

    for mimic in ordered:
        if mimic.slave_joint in used_slaves:
            continue
        if mimic.driver_joint in used_slaves or mimic.slave_joint in used_drivers_as_slave:
            continue
        accepted.append(mimic)
        used_slaves.add(mimic.slave_joint)
        used_drivers_as_slave.add(mimic.driver_joint)
    return accepted
