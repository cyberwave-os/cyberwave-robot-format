"""Closed-chain mechanics survive composition and constrain real dynamics."""

from copy import deepcopy
import xml.etree.ElementTree as ET

import pytest

from cyberwave_robot_format import (
    CommonSchema,
    Inertia,
    Joint,
    JointType,
    Link,
    Metadata,
    Pose,
    Vector3,
)
from cyberwave_robot_format.mjcf.exporter import MJCFExporter
from cyberwave_robot_format.schema import MimicJoint


def _schema():
    return CommonSchema(
        metadata=Metadata(name="parallel_slides"),
        links=[
            Link(name="base"),
            *[
                Link(name=name, mass=1, inertia=Inertia(ixx=0.01, iyy=0.01, izz=0.01))
                for name in ("a", "b")
            ],
        ],
        joints=[
            Joint(
                name="drive",
                type=JointType.PRISMATIC,
                parent_link="base",
                child_link="a",
                axis=Vector3(0, 0, 1),
                pose=Pose(position=Vector3(0, 0, 0.4)),
            ),
            Joint(
                name="follower",
                type=JointType.PRISMATIC,
                parent_link="base",
                child_link="b",
                axis=Vector3(0, 0, 1),
                pose=Pose(position=Vector3(0, 0, 0.6)),
                extensions={"passive": True},
            ),
        ],
        extensions={"loop_closures": [{"frame_a": "a", "frame_b": "b"}]},
    )


def test_closure_survives_serialization_composition_and_moves_passive_joint(tmp_path):
    mujoco = pytest.importorskip("mujoco")
    schema = _schema()
    before = deepcopy(schema.to_dict())
    schema = CommonSchema.from_dict(before)
    scene = CommonSchema(metadata=Metadata(name="two_robots"))
    for i in range(2):
        scene.merge_in(schema, f"robot{i}", Pose(position=Vector3(i, 0, 0)))
    assert schema.to_dict() == before
    assert scene.validate() == []
    xml = tmp_path / "scene.xml"
    MJCFExporter().export(scene, xml)
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    assert model.neq == 2
    assert model.nu == 2
    assert {model.joint(int(i)).name for i in model.actuator_trnid[:, 0]} == {
        "robot0__drive",
        "robot1__drive",
    }
    mujoco.mj_forward(model, data)
    initial = (
        data.xpos[model.body("robot0__b").id, 2]
        - data.xpos[model.body("robot0__a").id, 2]
    )
    assert initial == pytest.approx(0.2)
    # Distinct controls exercise each namespaced constraint. Both endpoints must
    # converge despite the open reference pose, without driving passive joints.
    data.ctrl[:] = [1, -1]
    for _ in range(500):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    for i in range(2):
        assert data.xpos[model.body(f"robot{i}__a").id, 2] == pytest.approx(
            data.xpos[model.body(f"robot{i}__b").id, 2], abs=0.001
        )
    assert sum(w.number for w in data.warning) == 0


def test_closures_coexist_with_mimic_and_authored_sites(tmp_path):
    schema = _schema()
    schema.joints[1].mimic = MimicJoint(joint="drive", offset=-0.2)
    schema.links[1].extensions["mjcf_sites"] = [
        {"name": "a__loop_0_frame_a", "pos": "0 0 1"}
    ]
    before = deepcopy(schema.to_dict())
    xml = tmp_path / "robot.xml"
    MJCFExporter().export(schema, xml)
    root = ET.parse(xml).getroot()
    assert len(root.findall("./equality/connect")) == 1
    assert len(root.findall("./equality/joint")) == 1
    connect = root.find("./equality/connect")
    assert connect.get("site1") != "a__loop_0_frame_a"
    assert root.find(".//site[@name='a__loop_0_frame_a']").get("pos") == "0 0 1"
    assert schema.to_dict() == before


@pytest.mark.parametrize(
    "closures",
    [
        {},
        [None],
        [{"frame_a": "a", "frame_b": "missing"}],
        [{"frame_a": "a", "frame_b": "a"}],
        [{"frame_a": "a", "frame_b": "b", "axes": "xx"}],
        [{"frame_a": "a", "frame_b": "b", "typo": True}],
    ],
)
def test_invalid_closures_fail_validation_and_export(closures, tmp_path):
    schema = _schema()
    schema.extensions["loop_closures"] = closures
    assert schema.validate()
    with pytest.raises(ValueError, match="loop_closures"):
        MJCFExporter().export(schema, tmp_path / "robot.xml")


def test_partial_axes_are_not_silently_converted_to_full_closure(tmp_path):
    schema = _schema()
    schema.extensions["loop_closures"][0]["axes"] = "xy"
    assert schema.validate() == []
    with pytest.raises(ValueError, match="require xyz axes"):
        MJCFExporter().export(schema, tmp_path / "robot.xml")


def test_legacy_schema_has_no_added_extension_or_constraint(tmp_path):
    schema = _schema()
    schema.extensions.clear()
    scene = CommonSchema(metadata=Metadata(name="scene"))
    scene.merge_in(schema, "legacy", Pose())
    assert scene.to_dict()["extensions"] == {}
    xml = tmp_path / "scene.xml"
    MJCFExporter().export(scene, xml)
    assert ET.parse(xml).getroot().find("equality") is None
