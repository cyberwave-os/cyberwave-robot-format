"""Default inheritance and free roots must survive import on any embodiment."""

import pytest

from cyberwave_robot_format.core import ParseError
from cyberwave_robot_format.mjcf import MJCFExporter, MJCFParser
from cyberwave_robot_format.schema import JointType

SOURCE = """<mujoco model="defaulted_mobile_robot">
  <compiler angle="radian"/>
  <default>
    <geom type="box" size=".1 .2 .3" friction=".6 .01 .001"/>
    <joint axis="0 1 0" damping="2" armature=".01"/>
    <default class="limb">
      <joint range="-1 2"/>
      <default class="hinge"><joint axis="1 0 0"/></default>
      <default class="foot"><geom size=".02 .03 .04" group="3" condim="6"/></default>
    </default>
    <default class="sibling"><joint range="-.2 .2"/></default>
  </default>
  <worldbody>
    <body name="chassis" childclass="limb" pos="0 0 1">
      <freejoint/>
      <geom name="body_geometry"/>
      <body name="first">
        <joint name="axis_a" class="hinge"/>
        <geom name="foot_geometry" class="foot" friction=".8 .02 .01"/>
        <body name="second" childclass="sibling">
          <joint name="axis_b" damping=".5"/>
          <geom name="second_geometry" size=".2 .2 .2"/>
        </body>
      </body>
      <body name="third"><joint name="axis_c"/><geom/></body>
    </body>
  </worldbody>
</mujoco>"""


def test_geom_derived_inertia_and_ellipsoids_survive_training_import(tmp_path):
    import numpy as np

    from cyberwave_robot_format.schema import CommonSchema, GeometryType

    mujoco = pytest.importorskip("mujoco")
    source = tmp_path / "source.xml"
    source.write_text("""<mujoco><option timestep=".01"/>
      <default><geom mass="0"/><default class="rotor"><geom type="ellipsoid" size=".13 .13 .01" mass=".25" group="3"/></default></default>
      <worldbody><body name="airframe" pos="0 0 1"><freejoint/>
        <geom class="rotor" pos="-.14 -.18 .05"/><geom class="rotor" pos="-.14 .18 .05"/>
        <geom class="rotor" pos=".14 .18 .08"/><geom class="rotor" pos=".14 -.18 .08"/>
        <geom type="ellipsoid" size=".16 .04 .02" mass=".325" pos="0 0 .02" group="2"/>
      </body></worldbody></mujoco>""")
    reference = mujoco.MjModel.from_xml_path(str(source))
    schema = CommonSchema.from_dict(MJCFParser().parse(source, resolve_inertia=True).to_dict())
    body = schema.get_link("airframe")
    assert body.mass == pytest.approx(1.325)
    assert len(body.collisions) == 4 and len(body.visuals) == 1
    assert body.collisions[0].geometry.type == GeometryType.ELLIPSOID
    assert body.collisions[0].geometry.size.to_list() == [0.13, 0.13, 0.01]
    destination = tmp_path / "roundtrip.xml"
    MJCFExporter().export(schema, destination)
    compiled = mujoco.MjModel.from_xml_path(str(destination))
    np.testing.assert_allclose(compiled.body_mass, reference.body_mass)
    np.testing.assert_allclose(compiled.body_ipos, reference.body_ipos)
    a, b = mujoco.MjData(reference), mujoco.MjData(compiled)
    a.qvel[:] = b.qvel[:] = [0.1, -0.2, 0.3, 0.4, -0.5, 0.6]
    for _ in range(25):
        mujoco.mj_step(reference, a)
        mujoco.mj_step(compiled, b)
        np.testing.assert_allclose(a.qpos, b.qpos, atol=1e-9, rtol=0)
        np.testing.assert_allclose(a.qvel, b.qvel, atol=1e-9, rtol=0)


@pytest.mark.parametrize(
    "options",
    [
        "",
        '<option integrator="Euler"/>',
        '<option integrator="RK4"/>',
        '<option cone="elliptic" impratio="100"/>',
        '<option gravity="0 0 -3.7" wind="1 0 0" density=".8" viscosity=".0002"/>',
        '<option timestep=".004" integrator="implicitfast" iterations="31" tolerance="1e-6" '
        'solver="CG" cone="elliptic" impratio="8" noslip_iterations="4" ls_iterations="9"/>',
    ],
)
def test_mjcf_solver_options_and_defaults_survive_compiled_roundtrip(tmp_path, options):
    mujoco = pytest.importorskip("mujoco")
    source = tmp_path / "source.xml"
    source.write_text(f'<mujoco>{options}<worldbody><body name="body"><geom size=".1"/></body></worldbody></mujoco>')
    expected = mujoco.MjModel.from_xml_path(str(source))
    schema = MJCFParser().parse(source)
    destination = tmp_path / "export.xml"
    MJCFExporter().export(schema, destination)
    actual = mujoco.MjModel.from_xml_path(str(destination))
    for field in (
        "timestep",
        "integrator",
        "iterations",
        "tolerance",
        "solver",
        "cone",
        "impratio",
        "noslip_iterations",
        "ls_iterations",
        "gravity",
        "wind",
        "density",
        "viscosity",
    ):
        assert getattr(actual.opt, field) == pytest.approx(getattr(expected.opt, field)), field


def test_new_mjcf_defaults_do_not_change_existing_common_schema_defaults(tmp_path):
    import xml.etree.ElementTree as ET

    from cyberwave_robot_format.schema import CommonSchema, Metadata, Physics

    for physics in (None, Physics()):
        schema = CommonSchema(metadata=Metadata(name="legacy"), physics=physics)
        destination = tmp_path / "legacy.xml"
        MJCFExporter().export(schema, destination)
        options = ET.parse(destination).find("option")
        assert options.get("integrator") == "implicitfast"
        assert options.get("noslip_iterations") == "10"


def test_portable_physics_edits_override_extension_attributes(tmp_path):
    import xml.etree.ElementTree as ET

    source = tmp_path / "source.xml"
    source.write_text('<mujoco><worldbody><body name="body"><geom size=".1"/></body></worldbody></mujoco>')
    schema = MJCFParser().parse(source)
    schema.physics.timestep = 0.005
    schema.physics.solver.type = "rk4"
    schema.physics.extensions["mjcf_option"].update(timestep=".009", integrator="Euler")
    destination = tmp_path / "export.xml"
    MJCFExporter().export(schema, destination)
    option = ET.parse(destination).find("option")
    assert option.get("timestep") == "0.005" and option.get("integrator") == "RK4"


@pytest.mark.parametrize("attributes", [[], {"impratio": []}, {"cone": True}, {1: "elliptic"}])
def test_invalid_mjcf_option_extensions_are_not_silently_stringified(tmp_path, attributes):
    from cyberwave_robot_format.schema import CommonSchema, Metadata, Physics

    schema = CommonSchema(metadata=Metadata(name="invalid"), physics=Physics())
    schema.physics.extensions["mjcf_option"] = attributes
    with pytest.raises(ValueError, match="scalar XML attributes"):
        MJCFExporter().export(schema, tmp_path / "invalid.xml")


def test_nested_defaults_childclass_and_explicit_overrides(tmp_path):
    path = tmp_path / "robot.xml"
    path.write_text(SOURCE)
    schema = MJCFParser().parse(path)
    assert schema.validate() == []
    assert schema.extensions["parse_context"]["errors"] == []
    assert schema.extensions["parse_context"]["warnings"] == []
    joints = {joint.name: joint for joint in schema.joints}
    assert joints["axis_a"].axis.x == 1
    assert joints["axis_a"].limits.lower == -1
    assert joints["axis_a"].dynamics.damping == 2
    assert joints["axis_a"].dynamics.armature == 0.01
    assert joints["axis_b"].axis.y == 1
    assert joints["axis_b"].limits.lower == -0.2
    assert joints["axis_b"].dynamics.damping == 0.5
    # A nested body's override never leaks back into its parent's later sibling.
    assert joints["axis_c"].limits.upper == 2
    assert joints["axis_c"].dynamics.damping == 2
    root = joints["chassis_joint"]
    assert root.type == JointType.FLOATING
    assert root.dynamics is None and root.limits is None
    links = {link.name: link for link in schema.links}
    assert len(links["first"].visuals) == 0  # inherited collision-only group
    foot = links["first"].collisions[0]
    assert foot.geometry.size.x == 0.04  # MuJoCo half-size to schema full size
    assert foot.extensions["friction"] == [0.8, 0.02, 0.01]
    assert foot.extensions["condim"] == "6"
    assert links["chassis"].collisions[0].geometry.size.z == 0.6


def test_defaulted_joint_limits_and_free_dofs_match_mujoco_after_export(tmp_path):
    mujoco = pytest.importorskip("mujoco")
    source = tmp_path / "robot.xml"
    source.write_text(SOURCE)
    reference = mujoco.MjModel.from_xml_path(str(source))
    schema = MJCFParser().parse(source)
    output = tmp_path / "export.xml"
    MJCFExporter().export(schema, output)
    compiled = mujoco.MjModel.from_xml_path(str(output))
    assert (compiled.nq, compiled.nv) == (reference.nq, reference.nv) == (10, 9)
    for name in ("axis_a", "axis_b", "axis_c"):
        expected, actual = reference.joint(name), compiled.joint(name)
        assert actual.range == pytest.approx(expected.range)
        assert actual.axis == pytest.approx(expected.axis)
        assert compiled.dof_damping[actual.dofadr[0]] == reference.dof_damping[expected.dofadr[0]]
        assert compiled.dof_armature[actual.dofadr[0]] == reference.dof_armature[expected.dofadr[0]]
    assert compiled.dof_damping[:6] == pytest.approx([0] * 6)
    assert compiled.dof_armature[:6] == pytest.approx([0] * 6)
    foot = compiled.geom("first_collision_0")
    assert foot.friction == pytest.approx([0.8, 0.02, 0.01])
    assert foot.condim == 6


@pytest.mark.parametrize("selector", ['childclass="missing"', 'class="missing"'])
def test_unknown_default_references_do_not_produce_partial_valid_schema(tmp_path, selector):
    element = (
        f'<body name="body" {selector}><geom size=".1"/></body>'
        if selector.startswith("child")
        else (f'<body name="body"><geom size=".1" {selector}/></body>')
    )
    path = tmp_path / "invalid.xml"
    path.write_text(f"<mujoco><worldbody>{element}</worldbody></mujoco>")
    with pytest.raises(ParseError, match="Unknown MJCF defaults"):
        MJCFParser().parse(path)


def test_raw_default_class_names_are_not_sanitized_or_merged(tmp_path):
    path = tmp_path / "names.xml"
    path.write_text("""<mujoco><default>
      <default class="shape/a"><geom size=".1"/></default>
      <default class="shape_a"><geom size=".2"/></default>
    </default><worldbody><body name="body">
      <geom class="shape/a"/><geom class="shape_a"/>
    </body></worldbody></mujoco>""")
    schema = MJCFParser().parse(path)
    body = next(link for link in schema.links if link.name == "body")
    assert [visual.geometry.radius for visual in body.visuals] == [0.1, 0.2]


def test_parser_reuse_does_not_leak_previous_defaults(tmp_path):
    path = tmp_path / "robot.xml"
    path.write_text(SOURCE)
    parser = MJCFParser()
    parser.parse(path)
    path.write_text('<mujoco><worldbody><body name="fixed"><geom size=".1"/></body></worldbody></mujoco>')
    schema = parser.parse(path)
    assert schema.joints == []
    body = next(link for link in schema.links if link.name == "fixed")
    assert body.visuals[0].geometry.radius == 0.1


@pytest.mark.parametrize("free_root", ['<freejoint name="floating"/>', '<joint name="floating" type="free"/>'])
def test_only_freejoint_shortcut_ignores_joint_defaults(tmp_path, free_root):
    path = tmp_path / "root.xml"
    path.write_text(f"""<mujoco><default><joint damping="2"/></default>
    <worldbody><body name="body">{free_root}<geom size=".1"/></body></worldbody></mujoco>""")
    root = MJCFParser().parse(path).joints[0]
    assert root.type == JointType.FLOATING
    if free_root.startswith("<freejoint"):
        assert root.dynamics is None
    else:
        assert root.dynamics.damping == 2


@pytest.mark.parametrize("directory_attribute", ["meshdir", "assetdir"])
def test_implicit_mesh_names_and_compiler_directory_preserve_visuals(tmp_path, directory_attribute):
    assets = tmp_path / "assets"
    assets.mkdir()
    mesh = assets / "body.obj"
    mesh.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    path = tmp_path / "robot.xml"
    path.write_text(f"""<mujoco><compiler {directory_attribute}="assets"/>
    <default><default class="visual"><geom type="mesh" group="2"/></default></default>
    <asset><mesh file="body.obj"/></asset>
    <worldbody><body name="chassis"><geom class="visual" mesh="body"/></body></worldbody></mujoco>""")
    schema = MJCFParser().parse(path)
    body = next(link for link in schema.links if link.name == "chassis")
    assert len(body.visuals) == 1
    assert body.visuals[0].geometry.filename == str(mesh)
    assert schema.extensions["parse_context"]["warnings"] == []


def test_missing_mesh_reference_is_reported_instead_of_silently_disappearing(tmp_path):
    path = tmp_path / "robot.xml"
    path.write_text("""<mujoco><worldbody><body name="body">
    <geom type="mesh" mesh="absent"/></body></worldbody></mujoco>""")
    schema = MJCFParser().parse(path)
    assert "Unknown mesh reference 'absent'" in schema.extensions["parse_context"]["warnings"][0]


@pytest.mark.parametrize(
    ("defaults", "mesh_attributes", "expected_scale"),
    [
        ('<mesh scale=".01 .02 .03"/>', "", [0.01, 0.02, 0.03]),
        ('<default class="robot"><mesh scale=".01 .01 .01"/></default>', 'class="robot"', [0.01] * 3),
        (
            '<default class="robot"><mesh scale=".01 .02 .03"/><default class="child"/></default>'
            '<default class="other"><mesh scale="5 5 5"/></default>',
            'class="child"',
            [0.01, 0.02, 0.03],
        ),
        ('<mesh scale=".01 .01 .01"/>', 'scale=".1 .2 .3"', [0.1, 0.2, 0.3]),
    ],
)
def test_mesh_default_scale_preserves_compiled_geometry_after_json_roundtrip(
    tmp_path, defaults, mesh_attributes, expected_scale
):
    import numpy as np

    mujoco = pytest.importorskip("mujoco")
    from cyberwave_robot_format.schema import CommonSchema

    (tmp_path / "body.obj").write_text("v 0 0 0\nv 10 0 0\nv 0 20 0\nv 0 0 30\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n")
    source = tmp_path / "robot.xml"
    source.write_text(f"""<mujoco><default>{defaults}</default>
      <asset><mesh file="body.obj" {mesh_attributes}/></asset>
      <worldbody><body name="chassis"><freejoint/>
        <geom type="mesh" mesh="body" mass="1"/>
      </body></worldbody></mujoco>""")
    schema = CommonSchema.from_dict(MJCFParser().parse(source, resolve_inertia=True).to_dict())
    chassis = next(link for link in schema.links if link.name == "chassis")
    assert chassis.visuals[0].geometry.scale.to_list() == pytest.approx(expected_scale)
    exported = tmp_path / "export.xml"
    MJCFExporter().export(schema, exported)

    def world_bounds(path):
        model = mujoco.MjModel.from_xml_path(str(path))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        geom_id = next(i for i in range(model.ngeom) if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH)
        mesh_id = model.geom_dataid[geom_id]
        first, count = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
        vertices = (
            model.mesh_vert[first : first + count] @ data.geom_xmat[geom_id].reshape(3, 3).T + data.geom_xpos[geom_id]
        )
        return np.stack([vertices.min(axis=0), vertices.max(axis=0)])

    np.testing.assert_allclose(world_bounds(exported), world_bounds(source), atol=1e-6)


def test_inertial_frame_and_rotation_without_translation_match_compiled_model(tmp_path):
    mujoco = pytest.importorskip("mujoco")
    import numpy as np

    path = tmp_path / "rotations.xml"
    path.write_text("""<mujoco><worldbody><body name="chassis"><freejoint/>
      <inertial mass="2" pos=".1 .2 .3" quat=".9238795325 0 0 .3826834324" diaginertia="1 2 2.5"/>
      <geom type="box" size=".1 .2 .3" quat=".7071067812 .7071067812 0 0"/>
    </body></worldbody></mujoco>""")
    source = mujoco.MjModel.from_xml_path(str(path))
    schema = MJCFParser().parse(path)
    output = tmp_path / "export.xml"
    MJCFExporter().export(schema, output)
    imported = mujoco.MjModel.from_xml_path(str(output))

    def inertia_in_link_frame(model):
        body = model.body("chassis")
        matrix = np.empty(9)
        mujoco.mju_quat2Mat(matrix, body.iquat)
        rotation = matrix.reshape(3, 3)
        return rotation @ np.diag(body.inertia) @ rotation.T

    np.testing.assert_allclose(inertia_in_link_frame(imported), inertia_in_link_frame(source), atol=1e-8)
    assert imported.body("chassis").ipos == pytest.approx(source.body("chassis").ipos)
    assert imported.body("chassis").mass == pytest.approx(source.body("chassis").mass)
    assert imported.geom("chassis_collision_0").quat == pytest.approx(source.geom(0).quat)


@pytest.mark.parametrize(
    "defaults",
    [
        '<default><default class="same"/><default class="same"/></default>',
        "<default><default/></default>",
    ],
)
def test_invalid_default_definitions_fail_explicitly(tmp_path, defaults):
    path = tmp_path / "invalid.xml"
    path.write_text(f"<mujoco>{defaults}<worldbody/></mujoco>")
    with pytest.raises(ParseError):
        MJCFParser().parse(path)


@pytest.mark.parametrize("quaternion", ["0 0 0 0", "nan 0 0 1", "1 0 0"])
def test_invalid_inertial_orientation_is_reported(tmp_path, quaternion):
    path = tmp_path / "invalid.xml"
    path.write_text(f"""<mujoco><worldbody><body name="body"><inertial mass="2"
    pos="0 0 0" diaginertia="1 2 2.5" quat="{quaternion}"/>
    <geom size=".1"/></body></worldbody></mujoco>""")
    schema = MJCFParser().parse(path)
    assert schema.extensions["parse_context"]["errors"] == ["Invalid inertia values for link body"]
