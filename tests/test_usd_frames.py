# Copyright [2025] Cyberwave
# Licensed under the Apache License, Version 2.0

"""Unit tests for ``cyberwave_robot_format.usd.frames``.

This module deliberately imports **no** ``pxr`` and uses no ``importorskip``: the
whole reason ``frames.py`` is kept free of USD imports is that the delicate math
behind the mapping — joint frames, degenerate axes, inertia eigendecomposition,
kinematic-tree accumulation — then gets covered on every machine, whether or not
``usd-core`` is installed.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pytest

from cyberwave_robot_format.schema import (
    CommonSchema,
    Inertia,
    Joint,
    JointType,
    Link,
    Metadata,
    Pose,
    Quaternion,
    Vector3,
)
from cyberwave_robot_format.usd.frames import (
    REF_AXIS_VECTORS,
    accumulated_world_transforms,
    axis_to_ref_quat,
    inertia_from_principal,
    inertia_to_principal,
    joint_frames_for_export,
    joint_pose_axis_from_frames,
    matrix_to_pose,
    matrix_to_quat,
    pose_to_matrix,
    quat_multiply,
    quat_to_matrix,
    ref_quat_to_axis,
)

X_AXIS = np.array([1.0, 0.0, 0.0])


def _rotate(quat: Quaternion, vector) -> np.ndarray:
    return quat_to_matrix(quat) @ np.asarray(vector, dtype=np.float64)


def _random_unit_quaternion(rng: np.random.Generator) -> Quaternion:
    values = rng.normal(size=4)
    values /= np.linalg.norm(values)
    if values[3] < 0.0:
        values = -values
    return Quaternion(x=values[0], y=values[1], z=values[2], w=values[3])


class TestAxisToRefQuat:
    """The reference axis is ``"X"``; an arbitrary schema axis is baked into it."""

    def test_axis_parallel_to_reference_is_identity(self):
        assert axis_to_ref_quat(Vector3(x=1.0, y=0.0, z=0.0)) == Quaternion()

    @pytest.mark.parametrize(
        "axis",
        [
            Vector3(x=0.0, y=1.0, z=0.0),
            Vector3(x=0.0, y=0.0, z=1.0),
            Vector3(x=0.0, y=-1.0, z=0.0),
            Vector3(x=1.0, y=1.0, z=1.0),
            Vector3(x=-0.3, y=0.5, z=0.81),
        ],
    )
    def test_reference_axis_maps_onto_the_schema_axis(self, axis):
        expected = np.array([axis.x, axis.y, axis.z])
        expected = expected / np.linalg.norm(expected)
        assert np.allclose(_rotate(axis_to_ref_quat(axis), X_AXIS), expected)

    def test_antiparallel_axis_uses_a_deterministic_pi_rotation(self):
        # The shortest-arc formula degenerates here (w = 1 + dot = 0, cross = 0),
        # so a NaN or a zero quaternion is the failure mode being guarded against.
        quat = axis_to_ref_quat(Vector3(x=-1.0, y=0.0, z=0.0))
        assert not any(math.isnan(v) for v in quat.to_list())
        assert math.isclose(quat.w, 0.0, abs_tol=1e-12)
        assert np.allclose(_rotate(quat, X_AXIS), [-1.0, 0.0, 0.0])
        assert axis_to_ref_quat(Vector3(x=-1.0, y=0.0, z=0.0)) == quat

    def test_unnormalized_axis_is_normalized_in_the_frame(self):
        assert axis_to_ref_quat(Vector3(x=0.0, y=0.0, z=2.0)) == axis_to_ref_quat(
            Vector3(x=0.0, y=0.0, z=1.0)
        )

    def test_zero_axis_yields_identity_and_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            assert axis_to_ref_quat(Vector3()) == Quaternion()
        assert "zero length" in caplog.text

    def test_axis_round_trip_over_random_unit_vectors(self):
        rng = np.random.default_rng(20260810)
        for _ in range(500):
            vector = rng.normal(size=3)
            norm = np.linalg.norm(vector)
            if norm < 1e-6:
                continue
            unit = vector / norm
            axis = Vector3(x=unit[0], y=unit[1], z=unit[2])
            recovered = ref_quat_to_axis(axis_to_ref_quat(axis))
            assert np.allclose(recovered.to_list(), unit, atol=1e-12)

    @pytest.mark.parametrize("token", ["X", "Y", "Z"])
    def test_ref_quat_to_axis_honours_the_token(self, token):
        assert np.allclose(
            ref_quat_to_axis(Quaternion(), token).to_list(), REF_AXIS_VECTORS[token]
        )

    def test_unknown_axis_token_falls_back_to_the_reference(self):
        assert ref_quat_to_axis(Quaternion(), "bogus") == ref_quat_to_axis(
            Quaternion(), "X"
        )


class TestJointFrames:
    """``joint_frames_for_export`` and its exact inverse."""

    def test_child_frame_sits_at_the_child_origin(self):
        pose = Pose(position=Vector3(x=1.0, y=2.0, z=3.0))
        _, _, local_pos1, _ = joint_frames_for_export(pose, Vector3(z=1.0))
        assert local_pos1 == Vector3()

    def test_parent_frame_position_is_the_joint_origin(self):
        pose = Pose(position=Vector3(x=1.0, y=2.0, z=3.0))
        local_pos0, _, _, _ = joint_frames_for_export(pose, Vector3(z=1.0))
        assert local_pos0 == pose.position

    def test_round_trip_over_random_poses_and_axes(self):
        rng = np.random.default_rng(4242)
        for _ in range(300):
            pose = Pose(
                position=Vector3(*rng.uniform(-5.0, 5.0, size=3)),
                orientation=_random_unit_quaternion(rng),
            )
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            axis = Vector3(*direction)

            frames = joint_frames_for_export(pose, axis)
            recovered_pose, recovered_axis = joint_pose_axis_from_frames(*frames)

            assert np.allclose(
                recovered_pose.position.to_list(), pose.position.to_list(), atol=1e-12
            )
            assert np.allclose(
                quat_to_matrix(recovered_pose.orientation),
                quat_to_matrix(pose.orientation),
                atol=1e-12,
            )
            assert np.allclose(recovered_axis.to_list(), direction, atol=1e-12)

    def test_import_folds_a_nonzero_child_offset_into_the_pose(self):
        # A foreign stage may place the child-side joint frame away from the child
        # origin. CommonSchema has nowhere to put that, so it must land in the pose.
        rot0 = Quaternion.from_rpy(0.3, -0.2, 0.7)
        rot1 = Quaternion.from_rpy(-0.1, 0.4, 0.2)
        pos0 = Vector3(x=0.5, y=-0.25, z=1.0)
        pos1 = Vector3(x=0.1, y=0.2, z=-0.3)

        pose, _ = joint_pose_axis_from_frames(pos0, rot0, pos1, rot1)

        frame0 = np.eye(4)
        frame0[:3, :3] = quat_to_matrix(rot0)
        frame0[:3, 3] = pos0.to_list()
        frame1 = np.eye(4)
        frame1[:3, :3] = quat_to_matrix(rot1)
        frame1[:3, 3] = pos1.to_list()
        expected = frame0 @ np.linalg.inv(frame1)

        assert np.allclose(pose_to_matrix(pose), expected, atol=1e-12)

    @pytest.mark.parametrize("token", ["X", "Y", "Z"])
    def test_import_reads_the_axis_from_the_child_frame(self, token):
        rot1 = Quaternion.from_rpy(0.2, 0.3, -0.4)
        _, axis = joint_pose_axis_from_frames(
            Vector3(), Quaternion(), Vector3(), rot1, token
        )
        assert np.allclose(
            axis.to_list(), quat_to_matrix(rot1) @ REF_AXIS_VECTORS[token], atol=1e-12
        )


class TestAccumulatedWorldTransforms:
    """Link prims are flat siblings, so each carries its full world placement."""

    @staticmethod
    def _chain() -> CommonSchema:
        return CommonSchema(
            metadata=Metadata(name="chain"),
            links=[Link(name="a"), Link(name="b"), Link(name="c")],
            joints=[
                Joint(
                    name="ab",
                    type=JointType.REVOLUTE,
                    parent_link="a",
                    child_link="b",
                    pose=Pose(
                        position=Vector3(z=0.5),
                        orientation=Quaternion.from_rpy(0.0, math.pi / 2, 0.0),
                    ),
                ),
                Joint(
                    name="bc",
                    type=JointType.REVOLUTE,
                    parent_link="b",
                    child_link="c",
                    pose=Pose(position=Vector3(x=0.25)),
                ),
            ],
        )

    def test_three_link_chain_matches_hand_computed_matrices(self):
        transforms = accumulated_world_transforms(self._chain())

        assert np.allclose(transforms["a"], np.eye(4))

        expected_b = np.eye(4)
        expected_b[:3, :3] = quat_to_matrix(Quaternion.from_rpy(0.0, math.pi / 2, 0.0))
        expected_b[:3, 3] = [0.0, 0.0, 0.5]
        assert np.allclose(transforms["b"], expected_b, atol=1e-12)

        local_c = np.eye(4)
        local_c[:3, 3] = [0.25, 0.0, 0.0]
        assert np.allclose(transforms["c"], expected_b @ local_c, atol=1e-12)

    def test_rotation_is_not_transposed(self):
        # A 90-degree pitch maps the parent's +X onto the world's -Z. If the matrix
        # were transposed this would come out as +Z and nothing else would notice.
        transforms = accumulated_world_transforms(self._chain())
        assert np.allclose(transforms["b"][:3, 0], [0.0, 0.0, -1.0], atol=1e-12)

    def test_world_is_identity_and_its_children_are_placed_by_the_joint(self):
        schema = CommonSchema(
            metadata=Metadata(name="spawned"),
            links=[Link(name="base")],
            joints=[
                Joint(
                    name="spawn",
                    type=JointType.FIXED,
                    parent_link="world",
                    child_link="base",
                    pose=Pose(position=Vector3(x=2.0, y=1.0)),
                )
            ],
        )
        transforms = accumulated_world_transforms(schema)
        assert np.allclose(transforms["world"], np.eye(4))
        assert np.allclose(transforms["base"][:3, 3], [2.0, 1.0, 0.0])

    def test_dangling_parent_gets_identity_and_warns(self, caplog):
        schema = CommonSchema(
            metadata=Metadata(name="dangling"),
            links=[Link(name="orphan")],
            joints=[
                Joint(
                    name="j",
                    type=JointType.FIXED,
                    parent_link="nope",
                    child_link="orphan",
                    pose=Pose(position=Vector3(x=9.0)),
                )
            ],
        )
        with caplog.at_level(logging.WARNING):
            transforms = accumulated_world_transforms(schema)
        assert np.allclose(transforms["orphan"], np.eye(4))
        assert "not reachable" in caplog.text

    def test_kinematic_loop_terminates(self, caplog):
        schema = CommonSchema(
            metadata=Metadata(name="loop"),
            links=[Link(name="p"), Link(name="q")],
            joints=[
                Joint(name="pq", type=JointType.FIXED, parent_link="p", child_link="q"),
                Joint(name="qp", type=JointType.FIXED, parent_link="q", child_link="p"),
            ],
        )
        with caplog.at_level(logging.WARNING):
            transforms = accumulated_world_transforms(schema)
        assert set(transforms) >= {"p", "q"}
        assert "not reachable" in caplog.text


class TestQuaternionHelpers:
    def test_round_trip_over_random_rotations(self):
        rng = np.random.default_rng(7)
        for _ in range(200):
            quat = _random_unit_quaternion(rng)
            recovered = matrix_to_quat(quat_to_matrix(quat))
            assert np.allclose(
                quat_to_matrix(recovered), quat_to_matrix(quat), atol=1e-12
            )

    def test_matrix_to_quat_canonicalizes_the_sign(self):
        # q and -q are the same rotation but compare unequal as dataclasses, so a
        # fixed sign is what makes export -> parse -> export byte-stable.
        negated = Quaternion(x=0.5, y=0.5, z=0.5, w=-0.5)
        recovered = matrix_to_quat(quat_to_matrix(negated))
        assert recovered.w >= 0.0
        assert np.allclose(
            quat_to_matrix(recovered), quat_to_matrix(negated), atol=1e-12
        )

    @pytest.mark.parametrize(
        "matrix",
        [
            np.diag([1.0, -1.0, -1.0]),  # pi about X: trace <= 0, m00 largest
            np.diag([-1.0, 1.0, -1.0]),  # pi about Y: m11 largest
            np.diag([-1.0, -1.0, 1.0]),  # pi about Z: m22 largest
        ],
    )
    def test_all_shepperd_branches_are_correct(self, matrix):
        assert np.allclose(quat_to_matrix(matrix_to_quat(matrix)), matrix, atol=1e-12)

    def test_zero_quaternion_reads_as_identity(self):
        assert np.allclose(quat_to_matrix(Quaternion(w=0.0)), np.eye(3))

    def test_quat_multiply_matches_matrix_composition(self):
        rng = np.random.default_rng(11)
        for _ in range(100):
            first = _random_unit_quaternion(rng)
            second = _random_unit_quaternion(rng)
            assert np.allclose(
                quat_to_matrix(quat_multiply(first, second)),
                quat_to_matrix(first) @ quat_to_matrix(second),
                atol=1e-12,
            )

    def test_pose_matrix_round_trip(self):
        pose = Pose(
            position=Vector3(x=1.5, y=-2.0, z=0.25),
            orientation=Quaternion.from_rpy(0.3, -0.7, 1.1),
        )
        recovered = matrix_to_pose(pose_to_matrix(pose))
        assert np.allclose(recovered.position.to_list(), pose.position.to_list())
        assert np.allclose(
            quat_to_matrix(recovered.orientation),
            quat_to_matrix(pose.orientation),
            atol=1e-12,
        )


class TestInertia:
    def test_diagonal_tensor_round_trips_exactly(self):
        inertia = Inertia(ixx=1.25, iyy=2.5, izz=0.75)
        diagonal, principal_axes, is_exact = inertia_to_principal(inertia)

        assert is_exact is True
        assert principal_axes == Quaternion()
        assert diagonal == (1.25, 2.5, 0.75)
        assert inertia_from_principal(diagonal, principal_axes) == inertia

    def test_full_tensor_recomposes_to_the_original(self):
        inertia = Inertia(
            ixx=1.0, iyy=2.0, izz=3.0, ixy=0.15, ixz=-0.25, iyz=0.05
        )
        diagonal, principal_axes, is_exact = inertia_to_principal(inertia)

        assert is_exact is False
        recomposed = inertia_from_principal(diagonal, principal_axes)
        for field in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz"):
            assert math.isclose(
                getattr(recomposed, field), getattr(inertia, field), abs_tol=1e-12
            )

    @pytest.mark.parametrize(
        "inertia",
        [
            Inertia(ixx=1.0, iyy=2.0, izz=3.0, ixy=0.5),
            Inertia(ixx=3.0, iyy=1.0, izz=2.0, ixz=-0.4, iyz=0.2),
            # Chosen because numpy.linalg.eigh returns a left-handed basis for it,
            # exercising the determinant fix-up.
            Inertia(ixx=2.0, iyy=2.0, izz=5.0, ixy=1.0, ixz=0.5, iyz=-0.5),
        ],
    )
    def test_principal_axes_is_always_a_proper_rotation(self, inertia):
        _, principal_axes, _ = inertia_to_principal(inertia)
        rotation = quat_to_matrix(principal_axes)
        assert math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-9)
        assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)

    def test_recomposed_tensor_is_symmetric(self):
        recomposed = inertia_from_principal(
            (1.0, 2.0, 3.0), Quaternion.from_rpy(0.4, 0.2, -0.6)
        )
        matrix = recomposed.to_matrix()
        assert np.allclose(matrix, matrix.T)
