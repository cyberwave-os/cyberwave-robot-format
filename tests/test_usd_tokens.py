# Copyright [2025] Cyberwave
# Licensed under the Apache License, Version 2.0

"""Validates ``usd/tokens.py``'s hardcoded strings against the installed schema.

``tokens.py`` spells every USD type name, token value and attribute name as a
plain string so that it can be imported (and diffed) without ``usd-core``. The
cost of that choice is that a ``usd-core`` release which renames something would
otherwise be discovered as silently-wrong output rather than as a failure — so
these tests check each table against what ``pxr`` actually provides.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pxr")

from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

from cyberwave_robot_format.schema import GeometryType, JointType  # noqa: E402
from cyberwave_robot_format.usd import tokens as t  # noqa: E402


@pytest.fixture
def stage():
    return Usd.Stage.CreateInMemory("tokens.usda")


class TestJointTables:
    @pytest.mark.parametrize("joint_type", list(JointType))
    def test_every_joint_type_has_both_table_entries(self, joint_type):
        assert joint_type in t.JOINT_TYPE_TO_PRIM_TYPE
        assert joint_type in t.JOINT_TYPE_TO_USDPHYSICS_CLASS

    @pytest.mark.parametrize("joint_type", list(JointType))
    def test_the_class_and_prim_type_tables_agree(self, joint_type, stage):
        # The two tables are keyed identically but used in different places — the
        # class name to Define() the prim, the prim-type name to assert on and to
        # invert on import — so they must not drift apart.
        class_name = t.JOINT_TYPE_TO_USDPHYSICS_CLASS[joint_type]
        schema_class = getattr(UsdPhysics, class_name, None)
        assert schema_class is not None, class_name
        prim = schema_class.Define(stage, f"/j_{joint_type.value}").GetPrim()
        assert str(prim.GetTypeName()) == t.JOINT_TYPE_TO_PRIM_TYPE[joint_type]

    def test_prim_type_to_joint_type_inverts_the_unambiguous_entries(self):
        for prim_type, joint_type in t.PRIM_TYPE_TO_JOINT_TYPE.items():
            assert t.JOINT_TYPE_TO_PRIM_TYPE[joint_type] == prim_type

    def test_reference_axis_token_is_the_schema_default(self, stage):
        # REF_AXIS_TOKEN is chosen precisely because it is the default, so that
        # authoring it never surprises a consumer reading an unauthored value.
        joint = UsdPhysics.RevoluteJoint.Define(stage, "/rev")
        assert joint.GetAxisAttr().Get() == t.REF_AXIS_TOKEN

    def test_lock_encoding_is_low_above_high(self):
        assert t.LOCK_LOW > t.LOCK_HIGH

    def test_locked_dof_tokens_are_real_usdphysics_tokens(self):
        valid = {
            UsdPhysics.Tokens.transX,
            UsdPhysics.Tokens.transY,
            UsdPhysics.Tokens.transZ,
            UsdPhysics.Tokens.rotX,
            UsdPhysics.Tokens.rotY,
            UsdPhysics.Tokens.rotZ,
        }
        for dofs in t.LOCKED_DOFS_FOR_JOINT_TYPE.values():
            assert set(dofs) <= valid

    def test_reduced_dof_joints_use_the_base_joint_prim(self):
        # A locked-DOF encoding only makes sense on the generic PhysicsJoint; the
        # typed joints already constrain their own DOFs.
        for joint_type in t.LOCKED_DOFS_FOR_JOINT_TYPE:
            assert t.JOINT_TYPE_TO_PRIM_TYPE[joint_type] == t.PRIM_JOINT

    def test_axis_attribute_table_matches_the_actual_schema(self, stage):
        for joint_type in JointType:
            class_name = t.JOINT_TYPE_TO_USDPHYSICS_CLASS[joint_type]
            prim = getattr(UsdPhysics, class_name).Define(
                stage, f"/axis_{joint_type.value}"
            ).GetPrim()
            has_axis = bool(prim.GetAttribute("physics:axis"))
            assert has_axis == (joint_type in t.JOINT_TYPES_WITH_AXIS_ATTR)

    def test_drive_instances_are_only_defined_for_drivable_joints(self):
        # FIXED has no DOF to drive and FLOATING's six cannot be expressed by one
        # drive instance, so both are deliberately absent.
        assert JointType.FIXED not in t.DRIVE_INSTANCE_FOR_JOINT_TYPE
        assert JointType.FLOATING not in t.DRIVE_INSTANCE_FOR_JOINT_TYPE
        assert set(t.DRIVE_INSTANCE_FOR_JOINT_TYPE.values()) == {
            t.DRIVE_ANGULAR,
            t.DRIVE_LINEAR,
        }

    def test_angular_and_linear_joint_types_do_not_overlap(self):
        assert not (t.ANGULAR_JOINT_TYPES & t.LINEAR_JOINT_TYPES)


class TestGeometryTables:
    @pytest.mark.parametrize("geometry_type", list(GeometryType))
    def test_every_geometry_type_maps_to_a_real_usdgeom_class(
        self, geometry_type, stage
    ):
        prim_type = t.GEOMETRY_TYPE_TO_PRIM_TYPE[geometry_type]
        schema_class = getattr(UsdGeom, prim_type, None)
        assert schema_class is not None, prim_type
        prim = schema_class.Define(stage, f"/g_{geometry_type.value}").GetPrim()
        assert str(prim.GetTypeName()) == prim_type

    def test_prim_type_to_geometry_type_inverts_the_unambiguous_entries(self):
        for prim_type, geometry_type in t.PRIM_TYPE_TO_GEOMETRY_TYPE.items():
            assert t.GEOMETRY_TYPE_TO_PRIM_TYPE[geometry_type] == prim_type


class TestTokenValues:
    def test_axis_tokens(self):
        assert (t.TOK_X, t.TOK_Y, t.TOK_Z) == (
            UsdGeom.Tokens.x,
            UsdGeom.Tokens.y,
            UsdGeom.Tokens.z,
        )
        assert t.AXIS_TOKENS == (UsdGeom.Tokens.x, UsdGeom.Tokens.y, UsdGeom.Tokens.z)

    def test_purpose_and_visibility_tokens(self):
        assert t.TOK_PURPOSE_RENDER == UsdGeom.Tokens.render
        assert t.TOK_PURPOSE_GUIDE == UsdGeom.Tokens.guide
        assert t.TOK_PURPOSE_PROXY == UsdGeom.Tokens.proxy
        assert t.TOK_INVISIBLE == UsdGeom.Tokens.invisible

    def test_drive_type_tokens(self):
        assert t.TOK_DRIVE_FORCE == UsdPhysics.Tokens.force
        assert t.TOK_DRIVE_ACCELERATION == UsdPhysics.Tokens.acceleration

    def test_mesh_approximation_tokens(self):
        assert t.TOK_APPROX_NONE == UsdPhysics.Tokens.none
        assert t.TOK_APPROX_CONVEX_HULL == UsdPhysics.Tokens.convexHull

    def test_drive_instance_names_produce_the_expected_attributes(self, stage):
        prim = UsdPhysics.RevoluteJoint.Define(stage, "/drv").GetPrim()
        UsdPhysics.DriveAPI.Apply(prim, t.DRIVE_ANGULAR)
        assert prim.GetAttribute(
            f"drive:{t.DRIVE_ANGULAR}:physics:stiffness"
        ).IsDefined()

    def test_collision_group_collection_name(self, stage):
        group = UsdPhysics.CollisionGroup.Define(stage, "/grp")
        includes = group.GetCollidersCollectionAPI().CreateIncludesRel()
        assert includes.GetName() == "collection:colliders:includes"


class TestAttributeNames:
    @staticmethod
    def _attribute_constants() -> dict[str, str]:
        """Discover the ``A_*`` constants by introspection, not a hardcoded list."""
        return {
            name: value
            for name, value in vars(t).items()
            if name.startswith("A_") and isinstance(value, str)
        }

    def test_there_are_attribute_constants_to_check(self):
        assert len(self._attribute_constants()) > 40

    def test_every_attribute_name_is_cyberwave_namespaced(self):
        for name, value in self._attribute_constants().items():
            assert value.startswith(f"{t.CW}:"), name

    def test_no_two_attribute_constants_share_a_spelling(self):
        # A duplicated spelling would make one schema field silently overwrite
        # another on the same prim — exactly the class of bug that cost
        # Geometry.extensions its own customData key.
        constants = self._attribute_constants()
        seen: dict[str, str] = {}
        for name, value in constants.items():
            assert value not in seen, f"{name} duplicates {seen.get(value)}: {value}"
            seen[value] = name

    def test_every_attribute_name_is_a_valid_usd_property_name(self):
        from pxr import Sdf

        for name, value in self._attribute_constants().items():
            assert Sdf.Path.IsValidNamespacedIdentifier(value), f"{name}: {value}"

    def test_custom_data_keys_are_distinct_and_namespaced(self):
        keys = {
            name: value
            for name, value in vars(t).items()
            if name.startswith("CD_") and isinstance(value, str)
        }
        assert len(set(keys.values())) == len(keys)
        for name, value in keys.items():
            assert value.startswith(f"{t.CW}:"), name

    def test_contact_surface_prefixes_do_not_collide(self):
        # Both prefixes feed the same writer/reader pair, so overlapping ones would
        # make Physics.contact and Contact.surface clobber each other.
        contact = {
            f"{t.CONTACT_SURFACE_PREFIX}{suffix}" for suffix in t.SURFACE_SUFFIXES
        }
        physics = {
            f"{t.PHYSICS_CONTACT_PREFIX}{suffix}" for suffix in t.SURFACE_SUFFIXES
        }
        assert not (contact & physics)


class TestBothSidesUseEveryConstant:
    """A constant written but never read (or vice versa) means a dropped field."""

    @staticmethod
    def _sources() -> tuple[str, str]:
        from pathlib import Path

        import cyberwave_robot_format.usd as usd_package

        directory = Path(usd_package.__file__).parent
        # The scene exporter authors the scene-level keys on the composed stage,
        # so it counts as part of the writing side.
        writers = ("exporter.py", "scene.py")
        return (
            "\n".join((directory / name).read_text() for name in writers),
            (directory / "parser.py").read_text(),
        )

    @pytest.mark.parametrize(
        "constant_name",
        sorted(
            name
            for name in vars(t)
            if name.startswith(("A_", "CD_"))
            and isinstance(getattr(t, name), str)
        ),
    )
    def test_constant_is_referenced_by_exporter_and_parser(self, constant_name):
        exporter_source, parser_source = self._sources()
        reference = f"t.{constant_name}"
        # A_NAME and the ContactSurface fields are reached through helpers rather
        # than by name, so allow the helper's own module to count.
        assert reference in exporter_source or reference in parser_source, constant_name

    def test_surface_suffixes_are_used_by_both_sides(self):
        exporter_source, parser_source = self._sources()
        for source in (exporter_source, parser_source):
            assert "SURFACE_PREFIX" in source or "CONTACT_PREFIX" in source
