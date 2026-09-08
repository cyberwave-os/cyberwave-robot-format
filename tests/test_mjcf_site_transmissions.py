"""Native thrust/torque frames must survive storage and multi-robot export."""

import json

import numpy as np
import pytest

from cyberwave_robot_format.core import ParseError
from cyberwave_robot_format.mjcf import MJCFExporter, MJCFParser
from cyberwave_robot_format.schema import CommonSchema, Metadata, Pose, Vector3

# Same transmission convention as Menagerie's Crazyflie: negative 1e-5
# moment gears are physical mappings, not scalar joint reduction ratios.
SOURCE = """<mujoco model="site_motor_robot">
  <compiler angle="radian"/>
  <option timestep=".002" integrator="RK4"/>
  <default>
    <site size=".002" group="4"/>
    <motor ctrllimited="true"/>
    <default class="vehicle"><site rgba=".1 .2 .3 1"/>
      <default class="moment"><motor ctrlrange="-1 1"/></default>
    </default>
  </default>
  <worldbody>
    <body name="vehicle" childclass="vehicle" pos="0 0 1">
      <freejoint/>
      <inertial pos="0 0 0" mass=".027" diaginertia=".000023951 .000023951 .000032347"/>
      <geom type="box" size=".02 .02 .01"/>
      <site name="actuation"/>
      <site name="imu" pos=".001 .002 .003" quat=".9238795325 .3826834324 0 0"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="thrust" site="actuation" ctrlrange="0 .35" gear="0 0 1 0 0 0"/>
    <motor name="roll" class="moment" site="actuation" gear="0 0 0 -.00001 0 0"/>
    <motor name="pitch" class="moment" site="actuation" gear="0 0 0 0 -.00001 0"/>
    <motor name="yaw" class="moment" site="actuation" gear="0 0 0 0 0 -.00001"/>
  </actuator>
  <sensor>
    <gyro name="gyro" site="imu"/>
    <accelerometer name="accel" site="imu"/>
    <framequat name="attitude" objtype="site" objname="imu"/>
  </sensor>
</mujoco>"""


def import_schema(tmp_path, source=SOURCE):
    path = tmp_path / "source.xml"
    path.write_text(source)
    return path, MJCFParser().parse(path)


def test_storage_preserves_site_bindings_and_full_gear_vectors(tmp_path):
    _, schema = import_schema(tmp_path)
    assert schema.validate() == []
    restored = CommonSchema.from_dict(json.loads(json.dumps(schema.to_dict())))
    assert restored.to_dict() == schema.to_dict()
    assert len(restored.actuators) == 4
    assert all(not actuator.joint and actuator.gear_ratio is None for actuator in restored.actuators)
    assert restored.get_actuator("pitch").get_mjcf_site_transmission()["gear"] == "0 0 0 0 -.00001 0"
    assert all(
        sensor.parent_link == "vehicle" and sensor.extensions["mjcf_site"] == "imu" for sensor in restored.sensors
    )
    assert restored.get_link("vehicle").get_mjcf_sites()[0]["rgba"] == ".1 .2 .3 1"


@pytest.mark.parametrize("ctrl", [[0.26487, 0, 0, 0], [0.28, 0.7, -0.4, 0.2]])
def test_compiled_transmissions_sensors_and_unforced_dynamics_match_source(tmp_path, ctrl):
    mujoco = pytest.importorskip("mujoco")
    source, schema = import_schema(tmp_path)
    destination = tmp_path / "roundtrip.xml"
    MJCFExporter().export(CommonSchema.from_dict(schema.to_dict()), destination)
    expected = mujoco.MjModel.from_xml_path(str(source))
    actual = mujoco.MjModel.from_xml_path(str(destination))
    assert (actual.nq, actual.nv, actual.nu, actual.nsite, actual.nsensor) == (7, 6, 4, 2, 3)
    for field in (
        "actuator_trntype",
        "actuator_gear",
        "actuator_ctrlrange",
        "actuator_ctrllimited",
        "actuator_gainprm",
        "actuator_biasprm",
    ):
        np.testing.assert_allclose(getattr(actual, field), getattr(expected, field))
    for name in ("actuation", "imu"):
        np.testing.assert_allclose(actual.site(name).pos, expected.site(name).pos)
        np.testing.assert_allclose(actual.site(name).quat, expected.site(name).quat)
    reference, replay = mujoco.MjData(expected), mujoco.MjData(actual)
    reference.ctrl[:] = replay.ctrl[:] = ctrl
    for _ in range(100):
        mujoco.mj_step(expected, reference)
        mujoco.mj_step(actual, replay)
        np.testing.assert_allclose(replay.qfrc_actuator, reference.qfrc_actuator, atol=1e-12)
        np.testing.assert_allclose(replay.qpos, reference.qpos, atol=1e-10)
        np.testing.assert_allclose(replay.qvel, reference.qvel, atol=1e-10)
        np.testing.assert_allclose(replay.sensordata, reference.sensordata, atol=1e-9)
        assert not np.any(replay.xfrc_applied) and not np.any(replay.qfrc_applied)


def test_two_instances_namespace_transmissions_and_shared_sensor_frames(tmp_path):
    mujoco = pytest.importorskip("mujoco")
    _, robot = import_schema(tmp_path)
    # Asset composition owns the world/spawn joint, as for existing imports.
    robot.links = [link for link in robot.links if link.name != "world"]
    robot.joints = []
    robot.actuators[0].extensions["mjcf_site_transmission"]["refsite"] = "imu"
    original = robot.to_dict()
    scene = CommonSchema(metadata=Metadata(name="two_vehicles"))
    for name, x in (("first", 0), ("second", 2)):
        scene.merge_in(robot, name, Pose(position=Vector3(x, 0, 1)), fixed_base=False)
    assert robot.to_dict() == original
    assert scene.validate() == []
    destination = tmp_path / "scene.xml"
    MJCFExporter().export(scene, destination)
    model = mujoco.MjModel.from_xml_path(str(destination))
    assert model.nu == 8 and model.nsite == 4 and model.nsensor == 6
    for name in ("first", "second"):
        actuator = model.actuator(f"{name}__thrust")
        assert actuator.trnid[0] == model.site(f"{name}__actuation").id
        assert actuator.trnid[1] == model.site(f"{name}__imu").id
        assert model.sensor(f"{name}__gyro").objid == model.site(f"{name}__imu").id


@pytest.mark.parametrize("gear", ["nan 0 1", "0 0 inf", "0 0 0 0 0 0 0", "", "nonsense"])
def test_invalid_motor_vectors_fail_import(tmp_path, gear):
    with pytest.raises(ParseError, match="gear vector"):
        import_schema(tmp_path, SOURCE.replace('gear="0 0 1 0 0 0"', f'gear="{gear}"'))


@pytest.mark.parametrize("limits", ["nan .35", ".4 .3", "0", "0 inf"])
def test_invalid_control_limits_are_not_imported_as_unlimited(tmp_path, limits):
    with pytest.raises(ParseError, match="invalid ctrlrange"):
        import_schema(tmp_path, SOURCE.replace('ctrlrange="0 .35"', f'ctrlrange="{limits}"'))


@pytest.mark.parametrize(
    "sites", [None, {}, [None], [{"name": "frame", "quat": []}], [{"name": "frame", "include": "file.xml"}]]
)
def test_malformed_site_extensions_fail_validation_and_export(tmp_path, sites):
    _, schema = import_schema(tmp_path)
    schema.get_link("vehicle").extensions["mjcf_sites"] = sites
    assert any("invalid MJCF site" in issue for issue in schema.validate())
    with pytest.raises(ValueError, match="invalid MJCF site"):
        MJCFExporter().export(schema, tmp_path / "invalid.xml")


@pytest.mark.parametrize(
    "transmission", [[], {"site": "actuation", "gear": []}, {"site": "actuation", "joint": "fake"}]
)
def test_invalid_extension_does_not_silently_turn_into_a_joint_motor(tmp_path, transmission):
    _, schema = import_schema(tmp_path)
    schema.actuators[0].extensions["mjcf_site_transmission"] = transmission
    assert any("invalid MJCF" in issue for issue in schema.validate())
    with pytest.raises(ValueError, match="invalid MJCF"):
        MJCFExporter().export(schema, tmp_path / "invalid.xml")


@pytest.mark.parametrize("reference", ["site", "refsite"])
def test_dangling_frames_fail_export_instead_of_emitting_passive_robot(tmp_path, reference):
    _, schema = import_schema(tmp_path)
    schema.actuators[0].extensions["mjcf_site_transmission"][reference] = "missing"
    assert any(f"unknown {reference}" in issue for issue in schema.validate())
    with pytest.raises(ValueError, match="undeclared"):
        MJCFExporter().export(schema, tmp_path / "invalid.xml")


def test_unsupported_site_servo_is_not_reinterpreted_as_a_motor(tmp_path):
    with pytest.raises(ParseError, match="stateless motor"):
        import_schema(tmp_path, SOURCE.replace('<motor name="thrust"', '<position name="thrust"'))


def test_duplicate_site_names_fail_schema_and_export(tmp_path):
    _, schema = import_schema(tmp_path, SOURCE.replace('name="imu" pos=', 'name="actuation" pos='))
    assert "Duplicate MJCF site names found." in schema.validate()
    with pytest.raises(ValueError, match="Duplicate MJCF site"):
        MJCFExporter().export(schema, tmp_path / "duplicate.xml")
