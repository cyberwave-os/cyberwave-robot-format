# Copyright [2025] Cyberwave
# Licensed under the Apache License, Version 2.0

"""Frame, axis and inertia math for the USD round trip.

Everything in this module is plain ``numpy`` — **no ``pxr`` import** — for two
reasons: the delicate parts of the USD mapping (joint frames, degenerate axes,
inertia eigendecomposition, world-transform accumulation) are then covered by
tests that run on every machine whether or not ``usd-core`` is installed, and
the exporter and parser are guaranteed to derive their frames from the *same*
formulas rather than from two hand-inlined copies.

Quaternion convention here is the schema's: :class:`Quaternion` stores
``(x, y, z, w)``. USD's ``Gf.Quatf``/``Gf.Quatd`` are real-first
(``(w, x, y, z)``); that conversion is confined to
:mod:`cyberwave_robot_format.usd.common`.

Rotation matrices here are **column-vector** (``v' = R @ v``), like the rest of
the package. USD matrices are row-vector; that transpose is likewise confined to
:mod:`~cyberwave_robot_format.usd.common`.
"""

from __future__ import annotations

import logging
import math
from collections import deque

import numpy as np

from cyberwave_robot_format.schema import (
    CommonSchema,
    Inertia,
    Pose,
    Quaternion,
    Vector3,
)
from cyberwave_robot_format.usd.tokens import AXIS_TOKENS, REF_AXIS_TOKEN, WORLD_LINK

logger = logging.getLogger(__name__)

_EPS = 1e-9

REF_AXIS_VECTORS: dict[str, np.ndarray] = {
    "X": np.array([1.0, 0.0, 0.0]),
    "Y": np.array([0.0, 1.0, 0.0]),
    "Z": np.array([0.0, 0.0, 1.0]),
}


# --------------------------------------------------------------------------
# Quaternion / matrix conversions
# --------------------------------------------------------------------------


def quat_to_matrix(q: Quaternion) -> np.ndarray:
    """Return the 3x3 rotation matrix for ``q`` (normalizing defensively)."""
    x, y, z, w = q.x, q.y, q.z, q.w
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < _EPS:
        # A zero quaternion is meaningless; identity is the only safe reading and
        # is what a caller comparing against Quaternion() would expect.
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ]
    )


def matrix_to_quat(matrix: np.ndarray) -> Quaternion:
    """Return the unit quaternion for a 3x3 rotation matrix.

    Uses Shepperd's method: pick the branch whose divisor is largest so the
    square root never operates on a near-zero value. The sign is canonicalized
    to ``w >= 0`` because ``q`` and ``-q`` are the same rotation but compare
    unequal as dataclasses, and a deterministic choice is what makes
    export -> parse -> export byte-stable.
    """
    m = np.asarray(matrix, dtype=np.float64)
    trace = m[0, 0] + m[1, 1] + m[2, 2]

    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s

    if w < 0.0:
        x, y, z, w = -x, -y, -z, -w
    return normalize_quat(Quaternion(x=x, y=y, z=z, w=w))


def normalize_quat(q: Quaternion) -> Quaternion:
    """Return ``q`` scaled to unit length (identity if degenerate)."""
    n = math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
    if n < _EPS:
        return Quaternion()
    return Quaternion(x=q.x / n, y=q.y / n, z=q.z / n, w=q.w / n)


def quat_multiply(a: Quaternion, b: Quaternion) -> Quaternion:
    """Return the Hamilton product ``a ⊗ b`` (apply ``b`` first, then ``a``)."""
    return Quaternion(
        x=a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        y=a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        z=a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        w=a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
    )


def pose_to_matrix(pose: Pose) -> np.ndarray:
    """Return the 4x4 homogeneous transform for ``pose``."""
    matrix = np.eye(4)
    matrix[:3, :3] = quat_to_matrix(pose.orientation)
    matrix[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
    return matrix


def matrix_to_pose(matrix: np.ndarray) -> Pose:
    """Return the :class:`Pose` for a 4x4 homogeneous transform."""
    m = np.asarray(matrix, dtype=np.float64)
    return Pose(
        position=Vector3(x=float(m[0, 3]), y=float(m[1, 3]), z=float(m[2, 3])),
        orientation=matrix_to_quat(m[:3, :3]),
    )


# --------------------------------------------------------------------------
# Kinematic-tree accumulation
# --------------------------------------------------------------------------


def accumulated_world_transforms(schema: CommonSchema) -> dict[str, np.ndarray]:
    """Return each link's world transform at the **zero configuration**.

    UsdPhysics forbids a ``PhysicsRigidBodyAPI`` prim from having a
    ``PhysicsRigidBodyAPI`` ancestor, so — unlike MJCF's nested ``<body>`` tree —
    USD link prims must be flat siblings, each carrying its full world placement.
    This walks the kinematic tree accumulating

        ``T_world(child) = T_world(parent) @ T_local(joint)``

    with every joint at position zero. ``home_position`` is deliberately *not*
    baked in: it would have to be un-baked on import, and it would make the link
    prim transforms disagree with the ``joint.pose`` values the joint prims carry.

    Links unreachable from a root (dangling ``parent_link``, or a kinematic loop —
    both of which ``CommonSchema.validate()`` only *reports*) get identity and a
    warning. Export never raises on a structurally invalid schema; one bad asset
    must not take down a whole scene export.
    """
    joints_by_child: dict[str, list] = {}
    for joint in schema.joints:
        joints_by_child.setdefault(joint.child_link, []).append(joint)

    children_by_parent: dict[str, list] = {}
    for joint in schema.joints:
        children_by_parent.setdefault(joint.parent_link, []).append(joint)

    transforms: dict[str, np.ndarray] = {WORLD_LINK: np.eye(4)}

    # Roots: links that are no joint's child. Joints parented to "world" still
    # position their child, so their children are *not* roots here (unlike
    # CommonSchema.get_root_links, which is about topology, not placement).
    queue: deque[str] = deque([WORLD_LINK])
    for link in schema.links:
        if link.name not in joints_by_child:
            transforms[link.name] = np.eye(4)
            queue.append(link.name)

    visited: set[str] = set()
    while queue:
        parent = queue.popleft()
        if parent in visited:
            continue
        visited.add(parent)
        for joint in children_by_parent.get(parent, []):
            child = joint.child_link
            if child in transforms and child != WORLD_LINK:
                # Two joints claiming the same child: keep the first (deterministic)
                # and let validate() report the real problem.
                continue
            transforms[child] = transforms[parent] @ pose_to_matrix(joint.pose)
            queue.append(child)

    for link in schema.links:
        if link.name not in transforms:
            logger.warning(
                "Link %r is not reachable from any root link (dangling parent or "
                "kinematic loop); exporting it at the world origin.",
                link.name,
            )
            transforms[link.name] = np.eye(4)

    return transforms


# --------------------------------------------------------------------------
# Joint axis <-> USD reference axis
# --------------------------------------------------------------------------


def axis_to_ref_quat(axis: Vector3, ref_token: str = REF_AXIS_TOKEN) -> Quaternion:
    """Return the rotation mapping the USD reference axis onto ``axis``.

    ``UsdPhysics`` restricts ``physics:axis`` to the tokens ``"X"``/``"Y"``/``"Z"``,
    while ``CommonSchema`` (like URDF) allows an arbitrary axis vector. The
    difference is absorbed by rotating the *joint frame* so that the reference
    axis points along the schema axis, and authoring that rotation into
    ``physics:localRot0``/``localRot1``.

    The shortest-arc rotation is used, which makes the result unique and hence the
    export deterministic. The import formula
    (:func:`joint_pose_axis_from_frames`) is invariant to which valid rotation
    was chosen, so foreign files using a different one still parse correctly.
    """
    reference = REF_AXIS_VECTORS[ref_token]
    vector = np.array([axis.x, axis.y, axis.z], dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm < _EPS:
        logger.warning(
            "Joint axis has zero length; using an identity joint frame. The exact "
            "value is preserved in the cyberwave:joint:axis sidecar."
        )
        return Quaternion()

    unit = vector / norm
    dot = float(np.dot(reference, unit))

    if dot > 1.0 - _EPS:
        return Quaternion()
    if dot < -1.0 + _EPS:
        # Anti-parallel: the shortest-arc formula degenerates to the zero
        # quaternion (w = 1 + dot = 0, cross = 0). Any pi rotation about an axis
        # orthogonal to the reference works; pick a fixed one so output is stable.
        orthogonal = _orthogonal_unit(reference)
        return Quaternion(
            x=float(orthogonal[0]),
            y=float(orthogonal[1]),
            z=float(orthogonal[2]),
            w=0.0,
        )

    cross = np.cross(reference, unit)
    return normalize_quat(
        Quaternion(
            x=float(cross[0]), y=float(cross[1]), z=float(cross[2]), w=1.0 + dot
        )
    )


def _orthogonal_unit(reference: np.ndarray) -> np.ndarray:
    """Return a deterministic unit vector orthogonal to ``reference``."""
    candidate = (
        np.array([0.0, 0.0, 1.0])
        if abs(reference[2]) < 0.9
        else np.array([0.0, 1.0, 0.0])
    )
    orthogonal = np.cross(reference, candidate)
    return orthogonal / float(np.linalg.norm(orthogonal))


def ref_quat_to_axis(
    quat: Quaternion, ref_token: str = REF_AXIS_TOKEN
) -> Vector3:
    """Return the schema axis encoded by a joint frame rotation.

    Inverse of :func:`axis_to_ref_quat`: the schema axis is the reference axis
    carried through the child-side joint frame rotation, ``a = R(q) @ e``.
    """
    token = ref_token if ref_token in AXIS_TOKENS else REF_AXIS_TOKEN
    axis = quat_to_matrix(quat) @ REF_AXIS_VECTORS[token]
    return Vector3(x=float(axis[0]), y=float(axis[1]), z=float(axis[2]))


# --------------------------------------------------------------------------
# Joint frames
# --------------------------------------------------------------------------


def joint_frames_for_export(
    pose: Pose, axis: Vector3, ref_token: str = REF_AXIS_TOKEN
) -> tuple[Vector3, Quaternion, Vector3, Quaternion]:
    """Return ``(localPos0, localRot0, localPos1, localRot1)`` for a joint.

    ``pose`` is the schema/URDF joint origin: the child link frame expressed in
    the parent link frame. ``axis`` is in the child frame.

    The child-side frame is placed at the child origin (``localPos1 = 0``) and
    rotated so the USD reference axis aligns with ``axis``; the parent-side frame
    is the same joint frame expressed in the parent, i.e. ``pose`` composed with
    that same rotation.

    This is provably consistent with
    :func:`accumulated_world_transforms`: the UsdPhysics requirement that the two
    joint frames coincide in world space,

        ``T_world(P) @ [R_j R_a, p_j] == T_world(C) @ [R_a, 0]``

    reduces to ``T_world(C) == T_world(P) @ T_local(joint)``, which is exactly
    the accumulation rule used to place the link prims.
    """
    ref_quat = axis_to_ref_quat(axis, ref_token)
    local_pos0 = Vector3(
        x=pose.position.x, y=pose.position.y, z=pose.position.z
    )
    local_rot0 = normalize_quat(quat_multiply(pose.orientation, ref_quat))
    return local_pos0, local_rot0, Vector3(), ref_quat


def joint_pose_axis_from_frames(
    local_pos0: Vector3,
    local_rot0: Quaternion,
    local_pos1: Vector3,
    local_rot1: Quaternion,
    ref_token: str = REF_AXIS_TOKEN,
) -> tuple[Pose, Vector3]:
    """Return ``(pose, axis)`` from a UsdPhysics joint's two local frames.

    The exact inverse of :func:`joint_frames_for_export`, written in the general
    form so it also handles foreign files that place the child-side joint frame
    away from the child origin (``localPos1 != 0``):

        ``T_PC = [R0, p0] @ [R1, p1]^-1``  =>  ``R_j = R0 @ R1^T``,
        ``p_j = p0 - R_j @ p1``,  ``a = R1 @ e``

    Any such child-origin offset is folded into ``joint.pose``, which is the only
    place ``CommonSchema`` can express it.
    """
    rot0 = quat_to_matrix(local_rot0)
    rot1 = quat_to_matrix(local_rot1)
    rot_joint = rot0 @ rot1.T
    pos1 = np.array([local_pos1.x, local_pos1.y, local_pos1.z], dtype=np.float64)
    pos0 = np.array([local_pos0.x, local_pos0.y, local_pos0.z], dtype=np.float64)
    position = pos0 - rot_joint @ pos1

    pose = Pose(
        position=Vector3(
            x=float(position[0]), y=float(position[1]), z=float(position[2])
        ),
        orientation=matrix_to_quat(rot_joint),
    )
    return pose, ref_quat_to_axis(local_rot1, ref_token)


# --------------------------------------------------------------------------
# Inertia
# --------------------------------------------------------------------------


def inertia_to_principal(
    inertia: Inertia,
) -> tuple[tuple[float, float, float], Quaternion, bool]:
    """Return ``(diagonal_inertia, principal_axes, is_exact)`` for USD.

    ``UsdPhysicsMassAPI`` stores inertia as a diagonal plus the orientation of
    the principal-axis frame, so a tensor with off-diagonal terms has to be
    eigendecomposed. ``numpy.linalg.eigh`` is used because the tensor is
    symmetric by construction, which guarantees real eigenvalues and an
    orthonormal basis.

    ``is_exact`` is ``False`` whenever a decomposition was needed, which is the
    exporter's cue to also write the double-precision ``cyberwave:link:inertia``
    sidecar (``physics:principalAxes`` is ``quatf``, i.e. float32).
    """
    if inertia.ixy == 0.0 and inertia.ixz == 0.0 and inertia.iyz == 0.0:
        return (inertia.ixx, inertia.iyy, inertia.izz), Quaternion(), True

    tensor = inertia.to_matrix()
    eigenvalues, eigenvectors = np.linalg.eigh(tensor)
    if float(np.linalg.det(eigenvectors)) < 0.0:
        # eigh may return a left-handed basis; flipping one column makes it a
        # proper rotation without changing V @ diag(w) @ V.T.
        eigenvectors = eigenvectors.copy()
        eigenvectors[:, 0] = -eigenvectors[:, 0]

    diagonal = (
        float(eigenvalues[0]),
        float(eigenvalues[1]),
        float(eigenvalues[2]),
    )
    return diagonal, matrix_to_quat(eigenvectors), False


def inertia_from_principal(
    diagonal: tuple[float, float, float] | list[float],
    principal_axes: Quaternion,
) -> Inertia:
    """Recompose a full inertia tensor from ``diag`` and ``principal_axes``.

    ``I = R @ diag(w) @ R^T``, symmetrized defensively against float32 drift in
    ``physics:principalAxes``. Eigenvalue *ordering* is irrelevant here — only
    the recomposed tensor matters.
    """
    rotation = quat_to_matrix(principal_axes)
    tensor = rotation @ np.diag(np.asarray(diagonal, dtype=np.float64)) @ rotation.T
    tensor = 0.5 * (tensor + tensor.T)
    return Inertia(
        ixx=float(tensor[0, 0]),
        iyy=float(tensor[1, 1]),
        izz=float(tensor[2, 2]),
        ixy=float(tensor[0, 1]),
        ixz=float(tensor[0, 2]),
        iyz=float(tensor[1, 2]),
    )


__all__ = [
    "REF_AXIS_VECTORS",
    "accumulated_world_transforms",
    "axis_to_ref_quat",
    "inertia_from_principal",
    "inertia_to_principal",
    "joint_frames_for_export",
    "joint_pose_axis_from_frames",
    "matrix_to_pose",
    "matrix_to_quat",
    "normalize_quat",
    "pose_to_matrix",
    "quat_multiply",
    "quat_to_matrix",
    "ref_quat_to_axis",
]
