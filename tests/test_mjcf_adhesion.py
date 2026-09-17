"""Suction contact actuators survive catalog storage and docked scene export."""

import json

import numpy as np
import pytest

from cyberwave_robot_format.core import ParseError
from cyberwave_robot_format.mjcf import MJCFExporter, MJCFParser
from cyberwave_robot_format.schema import CommonSchema, Metadata, Pose, Vector3

SOURCE = """<mujoco model="suction_tool">
  <compiler angle="radian"/>
  <option timestep=".002" integrator="implicitfast"/>
  <default>
    <adhesion ctrlrange="0 1"/>
    <default class="vacuum"><adhesion gain="20"/></default>
  </default>
  <worldbody>
    <body name="cup">
      <geom pos="0 0 .5" type="cylinder" size=".025 .005"/>
    </body>
    <body name="payload" pos="0 0 .4701">
      <freejoint name="payload_free"/>
      <inertial pos="0 0 0" mass=".1" diaginertia=".00004167 .00004167 .00004167"/>
      <geom type="box" size=".025 .025 .025"/>
    </body>
  </worldbody>
  <actuator><adhesion name="vacuum" class="vacuum" body="cup"/></actuator>
</mujoco>"""


def import_schema(tmp_path, source=SOURCE):
    path = tmp_path / "source.xml"
    path.write_text(source)
    return MJCFParser().parse(path)


def test_catalog_storage_and_compiled_body_force_contract(tmp_path):
    mujoco = pytest.importorskip("mujoco")
    schema = import_schema(tmp_path)
    assert schema.validate() == []
    restored = CommonSchema.from_dict(json.loads(json.dumps(schema.to_dict())))
    assert restored.to_dict() == schema.to_dict()
    assert restored.actuators[0].joint == ""
    destination = tmp_path / "roundtrip.xml"
    MJCFExporter().export(restored, destination)
    source = mujoco.MjModel.from_xml_string(SOURCE)
    exported = mujoco.MjModel.from_xml_path(str(destination))
    assert exported.njnt == 1  # Only the payload moves; the vacuum has no joint.
    for field in ("actuator_trntype", "actuator_ctrlrange", "actuator_gainprm", "actuator_biasprm"):
        np.testing.assert_allclose(getattr(exported, field), getattr(source, field))
    assert exported.actuator("vacuum").trnid[0] == exported.body("cup").id

    # The contact physics, not a kinematic attachment, must hold and release.
    for model in (source, exported):
        data = mujoco.MjData(model)
        data.ctrl[0] = 1
        for _ in range(500):
            mujoco.mj_step(model, data)
        held_z = float(data.body("payload").xpos[2])
        assert held_z > 0.46
        assert data.ncon > 0
        data.ctrl[0] = 0
        for _ in range(150):
            mujoco.mj_step(model, data)
        assert data.body("payload").xpos[2] < held_z - 0.2
        assert np.isfinite(data.qpos).all()
        assert sum(w.number for w in data.warning) == 0
        assert not np.any(data.xfrc_applied) and not np.any(data.qfrc_applied)


def test_no_contact_means_no_attachment(tmp_path):
    mujoco = pytest.importorskip("mujoco")
    schema = import_schema(tmp_path, SOURCE.replace(".4701", ".4"))
    destination = tmp_path / "separated.xml"
    MJCFExporter().export(schema, destination)
    model = mujoco.MjModel.from_xml_path(str(destination))
    data = mujoco.MjData(model)
    data.ctrl[0] = 1
    for _ in range(150):
        mujoco.mj_step(model, data)
    assert data.ncon == 0
    assert data.body("payload").xpos[2] < 0.2


def test_composition_namespaces_each_tool_body_without_mutating_the_asset(tmp_path):
    mujoco = pytest.importorskip("mujoco")
    tool = import_schema(tmp_path)
    tool.links = [link for link in tool.links if link.name == "cup"]
    tool.joints = []
    original = tool.to_dict()
    scene = CommonSchema(metadata=Metadata(name="two_tools"))
    for prefix, x in (("first", 0), ("second", 2)):
        scene.merge_in(tool, prefix, Pose(position=Vector3(x, 0, 1)), fixed_base=True)
    assert tool.to_dict() == original
    assert scene.validate() == []
    destination = tmp_path / "two.xml"
    MJCFExporter().export(scene, destination)
    model = mujoco.MjModel.from_xml_path(str(destination))
    assert model.nu == 2 and model.njnt == 0
    for prefix in ("first", "second"):
        assert model.actuator(f"{prefix}__vacuum").trnid[0] == model.body(f"{prefix}__cup").id


@pytest.mark.parametrize("value", ["-1 1", "nan 1", "0 inf", "1 0", "0", ""])
def test_invalid_limits_fail_import(tmp_path, value):
    with pytest.raises(ParseError, match="adhesion"):
        import_schema(tmp_path, SOURCE.replace('ctrlrange="0 1"', f'ctrlrange="{value}"'))


@pytest.mark.parametrize("value", ["nan", "inf", "-1", "0", "bad"])
def test_invalid_gain_fails_import(tmp_path, value):
    with pytest.raises(ParseError, match="adhesion"):
        import_schema(tmp_path, SOURCE.replace('gain="20"', f'gain="{value}"'))


@pytest.mark.parametrize("attribute", ['delay=".2"', 'joint="fake"', 'gear="2"'])
def test_unsupported_semantics_are_not_silently_discarded(tmp_path, attribute):
    with pytest.raises(ParseError, match="adhesion"):
        import_schema(tmp_path, SOURCE.replace('body="cup"', f'body="cup" {attribute}'))


@pytest.mark.parametrize("binding", [None, [], {}, {"body": "cup", "gain": []}, {"body": "cup", "unknown": "1"}])
def test_malformed_catalog_binding_fails_validation_and_export(tmp_path, binding):
    schema = import_schema(tmp_path)
    schema.actuators[0].extensions["mjcf_adhesion"] = binding
    assert any("adhesion" in issue for issue in schema.validate())
    with pytest.raises(ValueError, match="adhesion"):
        MJCFExporter().export(schema, tmp_path / "invalid.xml")


def test_unknown_body_fails_validation_and_export(tmp_path):
    schema = import_schema(tmp_path)
    schema.actuators[0].extensions["mjcf_adhesion"]["body"] = "missing"
    assert any("unknown body" in issue for issue in schema.validate())
    with pytest.raises(ValueError, match="unknown body"):
        MJCFExporter().export(schema, tmp_path / "invalid.xml")
