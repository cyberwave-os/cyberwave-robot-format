# Copyright [2025] Tomáš Macháček <tomasmachacekw@gmail.com>

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""URDF -> schema -> URDF must not lose what the schema cannot type.

The schema is the platform's source of truth, and the served URDF is exported from
it. Anything the exporter cannot emit is therefore content the platform deletes
from a user's robot: ``<ros2_control>`` is a robot's entire controller
configuration, ``<transmission>`` its actuator coupling, ``<gazebo>`` its
simulation plugins. These tests pin that they survive, that an *edit* to the schema
wins over the original, and that repeated cycles are a fixed point.
"""

from __future__ import annotations

import math
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from cyberwave_robot_format.schema import CommonSchema
from cyberwave_robot_format.urdf import (
    URDF_PASSTHROUGH_KEY,
    URDFExporter,
    URDFParser,
)

ROS_URDF = """<?xml version="1.0"?>
<robot name="ros_robot">
  <material name="grey">
    <color rgba="0.5 0.5 0.5 1"/>
  </material>
  <link name="base_link">
    <inertial>
      <origin xyz="0 0 0.1" rpy="0 0.3 0"/>
      <mass value="2.0"/>
      <inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/>
    </inertial>
    <visual>
      <geometry><mesh filename="../meshes/base.stl" scale="1 1 1"/></geometry>
      <material name="grey"/>
    </visual>
  </link>
  <link name="arm_link"/>
  <link name="tip_link"/>
  <joint name="joint_1" type="revolute">
    <parent link="base_link"/>
    <child link="arm_link"/>
    <origin xyz="0 0 1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="10" velocity="2"/>
    <safety_controller soft_lower_limit="-0.9" soft_upper_limit="0.9"
                       k_position="20" k_velocity="10"/>
    <calibration rising="0.1"/>
  </joint>
  <joint name="joint_2" type="continuous">
    <parent link="arm_link"/>
    <child link="tip_link"/>
    <axis xyz="0 1 0"/>
    <limit effort="5" velocity="1"/>
  </joint>
  <transmission name="trans_1">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="joint_1"><hardwareInterface>PositionJointInterface</hardwareInterface></joint>
    <actuator name="motor_1"><mechanicalReduction>1</mechanicalReduction></actuator>
  </transmission>
  <ros2_control name="GazeboSimSystem" type="system">
    <hardware><plugin>gz_ros2_control/GazeboSimSystem</plugin></hardware>
    <joint name="joint_1">
      <command_interface name="position"/>
      <state_interface name="position"/>
      <state_interface name="velocity"/>
    </joint>
  </ros2_control>
  <gazebo reference="base_link">
    <sensor name="wrist_camera" type="camera">
      <update_rate>30</update_rate>
      <camera><horizontal_fov>1.0</horizontal_fov></camera>
    </sensor>
    <plugin filename="libgazebo_ros_camera.so" name="camera_plugin"/>
  </gazebo>
  <gazebo>
    <plugin filename="libign_ros2_control.so" name="control_plugin">
      <parameters>/opt/config/controllers.yaml</parameters>
    </plugin>
  </gazebo>
</robot>
"""


def _parse(urdf_text: str, *, filename: str = "robot.urdf") -> CommonSchema:
    directory = Path(tempfile.mkdtemp())
    path = directory / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(urdf_text)
    return URDFParser().parse(path)


def _export(schema: CommonSchema, **kwargs) -> str:
    return URDFExporter(**kwargs).export_string(schema)


def _roundtrip(urdf_text: str) -> tuple[CommonSchema, str, CommonSchema]:
    """``parse -> export -> parse``, returning both schemas and the exported text."""
    original = _parse(urdf_text)
    exported = _export(original)
    return original, exported, _parse(exported, filename="reexported.urdf")


class TestPassthroughCapture:
    def test_ros_blocks_are_captured(self):
        schema = _parse(ROS_URDF)
        passthrough = schema.extensions[URDF_PASSTHROUGH_KEY]

        assert [entry["key"] for entry in passthrough["ros2_control"]] == [
            "GazeboSimSystem"
        ]
        assert [entry["key"] for entry in passthrough["transmission"]] == ["trans_1"]
        # Two <gazebo> blocks: one keyed by its reference, one anonymous.
        assert passthrough["gazebo"][0]["key"] == "base_link"
        assert passthrough["gazebo"][1]["key"].startswith("idx")
        assert (
            "gz_ros2_control/GazeboSimSystem" in passthrough["ros2_control"][0]["xml"]
        )

    def test_links_joints_and_materials_are_not_captured(self):
        """Content the schema models natively must not be duplicated into passthrough."""
        passthrough = _parse(ROS_URDF).extensions[URDF_PASSTHROUGH_KEY]
        assert "link" not in passthrough
        assert "joint" not in passthrough
        assert "material" not in passthrough
        assert "unknown" not in passthrough

    def test_unknown_element_is_captured_with_its_tag(self):
        urdf = ROS_URDF.replace(
            "</robot>", '  <custom_vendor_block foo="bar"/>\n</robot>'
        )
        unknown = _parse(urdf).extensions[URDF_PASSTHROUGH_KEY]["unknown"]
        assert len(unknown) == 1
        assert unknown[0]["tag"] == "custom_vendor_block"
        assert 'foo="bar"' in unknown[0]["xml"]

    def test_schema_with_no_extra_blocks_has_no_passthrough_section(self):
        urdf = '<?xml version="1.0"?><robot name="bare"><link name="base"/></robot>'
        assert URDF_PASSTHROUGH_KEY not in _parse(urdf).extensions


class TestPassthroughRoundTrip:
    def test_ros_blocks_survive_export(self):
        _, exported, _ = _roundtrip(ROS_URDF)
        root = ET.fromstring(exported)

        assert root.find("ros2_control") is not None
        assert root.find("transmission") is not None
        assert len(root.findall("gazebo")) == 2

        ros2_control = root.find("ros2_control")
        assert ros2_control.get("name") == "GazeboSimSystem"
        assert ros2_control.find("hardware/plugin").text == (
            "gz_ros2_control/GazeboSimSystem"
        )
        interfaces = ros2_control.findall("joint/state_interface")
        assert [elem.get("name") for elem in interfaces] == ["position", "velocity"]

    def test_transmission_nested_joint_is_not_confused_for_a_kinematic_joint(self):
        """``<transmission><joint>`` must not become a schema joint nor be rewritten."""
        original, exported, reparsed = _roundtrip(ROS_URDF)
        assert [joint.name for joint in original.joints] == ["joint_1", "joint_2"]
        assert [joint.name for joint in reparsed.joints] == ["joint_1", "joint_2"]

        transmission = ET.fromstring(exported).find("transmission")
        assert transmission.find("joint").get("name") == "joint_1"
        assert transmission.find("joint/hardwareInterface").text == (
            "PositionJointInterface"
        )

    def test_passthrough_is_a_fixed_point(self):
        """Repeated parse/export cycles must not grow or alter the stored blocks.

        XML keeps indentation as text nodes and the exporter pretty-prints, so an
        uncanonicalized capture nests one more level of blank lines per cycle — the
        stored schema and the served file would both drift forever.
        """
        original, exported, reparsed = _roundtrip(ROS_URDF)
        assert (
            original.extensions[URDF_PASSTHROUGH_KEY]
            == reparsed.extensions[URDF_PASSTHROUGH_KEY]
        )
        assert _export(reparsed) == exported

    def test_kinematics_and_materials_survive_export(self):
        original, _, reparsed = _roundtrip(ROS_URDF)
        before, after = original.to_dict(), reparsed.to_dict()
        assert before["joints"] == after["joints"]
        assert before["links"] == after["links"]
        assert before["materials"] == after["materials"]
        assert before["sensors"] == after["sensors"]


class TestPassthroughIsSchemaDriven:
    """An edit to the schema must win: the exporter renders only what the schema holds."""

    def test_overwritten_block_exports_overwritten(self):
        schema = _parse(ROS_URDF)
        schema.extensions[URDF_PASSTHROUGH_KEY]["ros2_control"][0]["xml"] = (
            '<ros2_control name="RealHardware" type="system">'
            "<hardware><plugin>kortex_driver/KortexHardware</plugin></hardware>"
            "</ros2_control>"
        )

        root = ET.fromstring(_export(schema))
        ros2_control = root.find("ros2_control")
        assert ros2_control.get("name") == "RealHardware"
        assert ros2_control.find("hardware/plugin").text == (
            "kortex_driver/KortexHardware"
        )
        assert "gz_ros2_control" not in _export(schema)

    def test_removed_block_is_not_exported(self):
        schema = _parse(ROS_URDF)
        del schema.extensions[URDF_PASSTHROUGH_KEY]["transmission"]
        assert ET.fromstring(_export(schema)).find("transmission") is None

    def test_added_block_is_exported(self):
        schema = _parse(ROS_URDF)
        schema.extensions[URDF_PASSTHROUGH_KEY]["ros2_control"].append(
            {"key": "Second", "xml": '<ros2_control name="Second" type="actuator"/>'}
        )
        names = [
            elem.get("name")
            for elem in ET.fromstring(_export(schema)).findall("ros2_control")
        ]
        assert names == ["GazeboSimSystem", "Second"]

    def test_malformed_block_raises_rather_than_exporting_a_gutted_robot(self):
        """Shipping a URDF silently missing its controllers is worse than failing."""
        schema = _parse(ROS_URDF)
        schema.extensions[URDF_PASSTHROUGH_KEY]["ros2_control"][0]["xml"] = (
            "<ros2_control><hardware>"
        )
        with pytest.raises(ValueError, match="not well-formed XML"):
            _export(schema)

    def test_entry_without_xml_raises(self):
        schema = _parse(ROS_URDF)
        schema.extensions[URDF_PASSTHROUGH_KEY]["gazebo"][0] = {"key": "base_link"}
        with pytest.raises(ValueError, match="has no 'xml' string"):
            _export(schema)


class TestTypedFidelityGaps:
    """Sections the parser used to drop even though the schema had fields for them."""

    def test_safety_controller_and_calibration_round_trip(self):
        original, exported, reparsed = _roundtrip(ROS_URDF)
        joint = original.get_joint("joint_1")
        assert joint.safety_controller == {
            "soft_lower_limit": -0.9,
            "soft_upper_limit": 0.9,
            "k_position": 20.0,
            "k_velocity": 10.0,
        }
        assert joint.calibration == {"rising": 0.1}
        assert (
            reparsed.get_joint("joint_1").safety_controller == joint.safety_controller
        )
        assert reparsed.get_joint("joint_1").calibration == joint.calibration

        elem = ET.fromstring(exported).find("joint[@name='joint_1']")
        assert elem.find("safety_controller") is not None
        assert elem.find("calibration").get("rising") == "0.1"

    def test_non_finite_attributes_are_dropped_not_carried(self):
        """``float("inf")`` parses, so these have to be rejected explicitly.

        Kept them and the schema serialises to JSON containing the bare ``Infinity``
        token, which no strict parser accepts — and a NaN soft limit silently disables
        the safety controller the author was configuring.
        """
        urdf = """<?xml version="1.0"?>
<robot name="nonfinite">
  <link name="base_link"/>
  <link name="arm_link"/>
  <joint name="joint_1" type="revolute">
    <parent link="base_link"/>
    <child link="arm_link"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="10" velocity="2"/>
    <safety_controller soft_lower_limit="-0.9" soft_upper_limit="inf"
                       k_position="NaN" k_velocity="10.0"/>
    <calibration rising="Infinity"/>
  </joint>
</robot>"""

        joint = _parse(urdf).get_joint("joint_1")

        # The finite siblings survive; only the bad attributes are dropped.
        assert joint.safety_controller == {
            "soft_lower_limit": -0.9,
            "k_velocity": 10.0,
        }
        # Every attribute was unusable, so the section reads as "not authored".
        assert joint.calibration is None

    def test_dropped_attributes_are_each_reported(self):
        urdf = ROS_URDF.replace('k_position="20"', 'k_position="stiff"').replace(
            'rising="0.1"', 'rising="-inf"'
        )
        schema = _parse(urdf)

        joint = schema.get_joint("joint_1")
        assert "k_position" not in joint.safety_controller
        assert joint.calibration is None
        warnings = schema.extensions["parse_context"]["warnings"]
        assert any("<safety_controller k_position='stiff'>" in w for w in warnings)
        assert any("<calibration rising='-inf'>" in w for w in warnings)

    @pytest.mark.parametrize(
        "safety_xml",
        [
            # k_velocity dropped by the parser as non-finite...
            '<safety_controller soft_lower_limit="-0.5" soft_upper_limit="0.5" '
            'k_position="1" k_velocity="nan"/>',
            # ...or simply not authored.
            '<safety_controller soft_lower_limit="-0.5" soft_upper_limit="0.5" '
            'k_position="1"/>',
        ],
    )
    def test_safety_controller_without_k_velocity_is_omitted(self, safety_xml, caplog):
        """URDF requires ``k_velocity``; urdfdom rejects the document without it.

        It used to be written anyway, turning a URDF that loaded (the element was
        simply dropped before) into one that does not. A placeholder is no better:
        ``k_velocity=0`` makes the safety controller cap the joint at zero effort.
        """
        authored = (
            '<safety_controller soft_lower_limit="-0.9" soft_upper_limit="0.9"\n'
            '                       k_position="20" k_velocity="10"/>'
        )
        assert authored in ROS_URDF
        urdf = ROS_URDF.replace(authored, safety_xml)
        schema = _parse(urdf)
        assert "k_velocity" not in schema.get_joint("joint_1").safety_controller

        with caplog.at_level("WARNING"):
            exported = _export(schema)

        joint = ET.fromstring(exported).find("joint[@name='joint_1']")
        assert joint.find("safety_controller") is None
        # Only the incomplete section goes; its sibling is still written.
        assert joint.find("calibration").get("rising") == "0.1"
        assert "safety_controller without k_velocity" in caplog.text
        # The schema keeps what was authored, so nothing is lost for other formats.
        assert schema.get_joint("joint_1").safety_controller["k_position"] == 1.0

    def test_schema_safety_controller_without_k_velocity_is_omitted(self, caplog):
        """A USD-sourced schema can carry any subset of the gains."""
        schema = _parse(ROS_URDF)
        schema.get_joint("joint_1").safety_controller = {"k_position": 10.0}

        with caplog.at_level("WARNING"):
            exported = _export(schema)

        assert (
            ET.fromstring(exported).find("joint[@name='joint_1']/safety_controller")
            is None
        )
        assert "safety_controller without k_velocity" in caplog.text

    def test_non_finite_mimic_coefficients_are_refused(self):
        """A non-finite coupling is worse than none: it poisons everything derived.

        ``slave = multiplier * driver + offset`` yields NaN for every position, the
        MJCF ``<equality>`` carries ``nan`` polycoefs, and the schema's JSON emits a
        bare ``NaN``. The backend's runtime coercion already drops these, so admitting
        them here only makes ingest, export and execution disagree.
        """
        urdf = """<?xml version="1.0"?>
<robot name="nonfinite_mimic">
  <link name="base_link"/>
  <link name="a_link"/>
  <link name="b_link"/>
  <joint name="driver" type="revolute">
    <parent link="base_link"/><child link="a_link"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="10" velocity="2"/>
  </joint>
  <joint name="follower" type="revolute">
    <parent link="a_link"/><child link="b_link"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="10" velocity="2"/>
    <mimic joint="driver" multiplier="inf" offset="0"/>
  </joint>
</robot>"""

        schema = _parse(urdf)

        # Dropped, not admitted with a poisoned coefficient.
        assert schema.get_joint("follower").mimic is None
        # ...and the drop is reported rather than silent. It used to sit unread in
        # `extensions["parse_context"]["errors"]`, so a coupling the parser refused left
        # no trace anywhere a consumer would look.
        assert any("Invalid mimic values" in issue for issue in schema.parse_issues())
        # Reported, not gated. And the schema does not then claim to be
        # valid-with-a-coupling either.
        assert not [issue for issue in schema.validate() if "mimic" in issue.lower()]

    def test_joint_without_the_sections_keeps_them_none(self):
        """``None`` (not ``{}``) so the backend can tell "not authored" from "empty"."""
        joint = _parse(ROS_URDF).get_joint("joint_2")
        assert joint.safety_controller is None
        assert joint.calibration is None

    def test_inertial_orientation_round_trips(self):
        original, exported, reparsed = _roundtrip(ROS_URDF)
        roll, pitch, yaw = original.get_link("base_link").inertial_orientation.to_rpy()
        assert math.isclose(pitch, 0.3, abs_tol=1e-9)
        assert math.isclose(roll, 0.0, abs_tol=1e-9)
        assert math.isclose(yaw, 0.0, abs_tol=1e-9)

        emitted = ET.fromstring(exported).find(
            "link[@name='base_link']/inertial/origin"
        )
        assert [float(v) for v in emitted.get("rpy").split()] == pytest.approx(
            [0.0, 0.3, 0.0], abs=1e-9
        )
        assert reparsed.get_link("base_link").inertial_orientation.to_rpy() == (
            pytest.approx(original.get_link("base_link").inertial_orientation.to_rpy())
        )

    def test_document_material_is_emitted_once_and_referenced_by_name(self):
        """A by-name reference must stay a reference, not be inlined per visual."""
        _, exported, _ = _roundtrip(ROS_URDF)
        root = ET.fromstring(exported)

        document_materials = root.findall("material")
        assert [elem.get("name") for elem in document_materials] == ["grey"]
        assert document_materials[0].find("color") is not None

        visual_material = root.find("link[@name='base_link']/visual/material")
        assert visual_material.get("name") == "grey"
        assert visual_material.find("color") is None


class TestURDFConventions:
    def test_fixed_joint_gets_no_axis(self):
        urdf = """<?xml version="1.0"?>
        <robot name="fixed_only">
          <link name="a"/><link name="b"/>
          <joint name="j" type="fixed"><parent link="a"/><child link="b"/></joint>
        </robot>"""
        joint = ET.fromstring(_export(_parse(urdf))).find("joint")
        assert joint.find("axis") is None

    def test_continuous_joint_gets_no_positional_bounds(self):
        """Bounds on a continuous joint silently convert it to a limited hinge."""
        limit = ET.fromstring(_export(_parse(ROS_URDF))).find(
            "joint[@name='joint_2']/limit"
        )
        assert limit.get("lower") is None
        assert limit.get("upper") is None
        assert limit.get("effort") == "5.0"
        assert limit.get("velocity") == "1.0"

    def test_identity_origin_is_omitted(self):
        """URDF defaults a missing <origin> to identity; materializing it is churn."""
        urdf = """<?xml version="1.0"?>
        <robot name="identity">
          <link name="a"/><link name="b"/>
          <joint name="j" type="fixed"><parent link="a"/><child link="b"/></joint>
        </robot>"""
        assert ET.fromstring(_export(_parse(urdf))).find("joint/origin") is None

    def test_bounded_joint_without_bounds_is_reported(self, caplog):
        schema = _parse(ROS_URDF)
        schema.get_joint("joint_1").limits.lower = None
        with caplog.at_level("WARNING"):
            _export(schema)
        assert "no positional bounds" in caplog.text

    def test_bounds_without_effort_or_velocity_fill_both_with_a_warning(self, caplog):
        """URDF requires effort and velocity on every <limit>; urdfdom rejects the
        whole document without them. An MJCF-imported hinge carries only its range."""
        schema = _parse(ROS_URDF)
        limits = schema.get_joint("joint_1").limits
        limits.effort = None
        limits.velocity = None

        with caplog.at_level("WARNING"):
            exported = _export(schema)

        limit = ET.fromstring(exported).find("joint[@name='joint_1']/limit")
        assert limit.attrib == {
            "lower": "-1.0",
            "upper": "1.0",
            "effort": "0.0",
            "velocity": "0.0",
        }
        assert "no effort or velocity" in caplog.text
        reparsed_schema = _parse(exported, filename="reexported.urdf")
        reparsed = reparsed_schema.get_joint("joint_1")
        assert (reparsed.limits.effort, reparsed.limits.velocity) == (0.0, 0.0)
        # The placeholder is spelled the way every other float is, so a second
        # export of the re-parsed schema writes the same <limit> again.
        again = ET.fromstring(_export(reparsed_schema)).find(
            "joint[@name='joint_1']/limit"
        )
        assert again.attrib == limit.attrib

    def test_round_tripped_placeholder_is_not_read_as_a_zero_ceiling(self):
        """The ``0.0`` placeholder re-imports as ``0.0`` (as a CAD exporter's
        ``effort="0"`` does). The scene drive report must read it as "unspecified",
        not as a cap: the Isaac policy binding clamps torque to ``max_force``."""
        from cyberwave_robot_format.schema import Actuator, ActuatorType
        from cyberwave_robot_format.usd.scene import _drive_entry

        schema = _parse(ROS_URDF)
        limits = schema.get_joint("joint_1").limits
        limits.effort = None
        limits.velocity = None
        reparsed = _parse(_export(schema), filename="reexported.urdf").get_joint(
            "joint_1"
        )
        assert (reparsed.limits.effort, reparsed.limits.velocity) == (0.0, 0.0)

        drive = _drive_entry(reparsed, None, 0, None)
        assert (drive["max_force"], drive["max_velocity"]) == (None, None)
        assert (drive["lower"], drive["upper"]) == (-1.0, 1.0)

        # A real ceiling still comes through.
        complete = _drive_entry(_parse(ROS_URDF).get_joint("joint_1"), None, 0, None)
        assert (complete["max_force"], complete["max_velocity"]) == (10.0, 2.0)

        # A declared actuator's zero is a value, not a placeholder.
        actuator = Actuator(
            name="joint_1_motor",
            type=ActuatorType.TORQUE,
            joint="joint_1",
            max_torque=0.0,
        )
        assert _drive_entry(reparsed, actuator, 0, None)["max_force"] == 0.0

    def test_only_the_missing_attribute_is_filled(self, caplog):
        schema = _parse(ROS_URDF)
        schema.get_joint("joint_2").limits.velocity = None

        with caplog.at_level("WARNING"):
            limit = ET.fromstring(_export(schema)).find("joint[@name='joint_2']/limit")

        assert limit.get("effort") == "5.0"
        assert limit.get("velocity") == "0.0"
        assert "no velocity" in caplog.text

    def test_complete_limit_is_written_unchanged_and_silently(self, caplog):
        with caplog.at_level("WARNING"):
            limit = ET.fromstring(_export(_parse(ROS_URDF))).find(
                "joint[@name='joint_1']/limit"
            )
        assert limit.attrib == {
            "lower": "-1.0",
            "upper": "1.0",
            "effort": "10.0",
            "velocity": "2.0",
        }
        assert "URDF requires on every <limit>" not in caplog.text


class TestDocumentMaterialIsTheSourceOfTruth:
    """A named material's document-level copy wins over a visual's copy of it.

    After a save/reload the visual holds its own copy, so the two can disagree when
    only one is edited. urdfdom resolves the same conflict in favour of the global
    material; the exporters do too, and say so.
    """

    @staticmethod
    def _diverged() -> CommonSchema:
        schema = CommonSchema.from_dict(_parse(ROS_URDF).to_dict())
        visual_material = schema.get_link("base_link").visuals[0].material
        assert visual_material is not schema.materials[0]
        visual_material.color = [1.0, 0.0, 0.0, 1.0]
        return schema

    def test_urdf_export_keeps_the_document_copy_and_warns(self, caplog):
        with caplog.at_level("WARNING"):
            exported = _export(self._diverged())

        root = ET.fromstring(exported)
        assert root.find("material[@name='grey']/color").get("rgba") == "0.5 0.5 0.5 1.0"
        visual_material = root.find("link[@name='base_link']/visual/material")
        assert visual_material.attrib == {"name": "grey"}
        assert len(visual_material) == 0
        assert "differs from the document material" in caplog.text

        reparsed = _parse(exported, filename="reexported.urdf")
        assert reparsed.get_link("base_link").visuals[0].material.color == [
            0.5,
            0.5,
            0.5,
            1.0,
        ]

    def test_a_name_only_visual_copy_is_a_reference_not_a_conflict(self, caplog):
        schema = CommonSchema.from_dict(_parse(ROS_URDF).to_dict())
        visual_material = schema.get_link("base_link").visuals[0].material
        visual_material.color = None

        with caplog.at_level("WARNING"):
            _export(schema)

        assert "differs from the document material" not in caplog.text

    def test_mjcf_export_keeps_the_document_copy_too(self):
        from cyberwave_robot_format.mjcf import MJCFExporter

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "robot.xml"
            MJCFExporter().export(self._diverged(), path)
            root = ET.parse(path).getroot()

        material = root.find("asset/material[@name='grey']")
        assert [float(v) for v in material.get("rgba").split()] == pytest.approx(
            [0.5, 0.5, 0.5, 1.0]
        )

    def test_a_composed_scene_keeps_the_document_copy_too(self):
        """``merge_in`` drops ``materials``; it must resolve them onto the visuals
        first, or the simulation scene renders the stale visual copy."""
        from cyberwave_robot_format.mjcf import export_mujoco_scene_xml
        from cyberwave_robot_format.schema import Metadata, Pose

        scene = CommonSchema(metadata=Metadata(name="scene"))
        scene.merge_in(self._diverged(), "arm1", spawn_pose=Pose())

        material = ET.fromstring(export_mujoco_scene_xml(scene)).find(
            "asset/material[@name='arm1__grey']"
        )
        assert [float(v) for v in material.get("rgba").split()] == pytest.approx(
            [0.5, 0.5, 0.5, 1.0]
        )

    def test_usd_export_keeps_the_document_copy_too(self):
        pytest.importorskip("pxr")
        from pxr import Sdf, Usd, UsdShade

        from cyberwave_robot_format.usd import USDExporter

        layer = Sdf.Layer.CreateAnonymous(".usda")
        layer.ImportFromString(USDExporter().export_to_string(self._diverged()))
        stage = Usd.Stage.Open(layer)

        colors = [
            tuple(shader.GetInput("diffuseColor").Get())
            for prim in stage.Traverse()
            if prim.IsA(UsdShade.Shader)
            for shader in [UsdShade.Shader(prim)]
            if shader.GetInput("diffuseColor")
        ]
        assert colors == [pytest.approx((0.5, 0.5, 0.5))]


class TestMeshReferences:
    """The exporter emits a mesh reference verbatim; re-basing needs storage."""

    def test_a_relative_reference_is_emitted_unchanged(self):
        schema = _parse(ROS_URDF)
        schema.get_link("base_link").visuals[0].geometry.filename = "../meshes/base.stl"
        mesh = ET.fromstring(_export(schema)).find(
            "link[@name='base_link']/visual/geometry/mesh"
        )
        assert mesh.get("filename") == "../meshes/base.stl"

    def test_a_project_root_relative_reference_is_also_emitted_unchanged(self):
        """Choosing a frame here would break whichever half of the catalog disagrees.

        The stored schema is not consistent about the frame — measured across a real
        catalog, most references are already URDF-dir-relative while mesh-synthesized
        assets are project-root-relative — so the exporter stays out of it and the
        platform's storage-aware resolver decides. See ``_mesh_filename``.
        """
        schema = _parse(ROS_URDF)
        schema.get_link("base_link").visuals[0].geometry.filename = "meshes/base.stl"
        mesh = ET.fromstring(_export(schema)).find(
            "link[@name='base_link']/visual/geometry/mesh"
        )
        assert mesh.get("filename") == "meshes/base.stl"

    @pytest.mark.parametrize(
        "filename",
        [
            "package://my_robot/meshes/base.stl",
            "https://cdn.example.com/base.stl",
            "/absolute/meshes/base.stl",
        ],
    )
    def test_location_independent_references_are_untouched(self, filename):
        schema = _parse(ROS_URDF)
        schema.get_link("base_link").visuals[0].geometry.filename = filename
        mesh = ET.fromstring(_export(schema)).find(
            "link[@name='base_link']/visual/geometry/mesh"
        )
        assert mesh.get("filename") == filename

    def test_mesh_scale_survives(self):
        _original, exported, reparsed = _roundtrip(ROS_URDF)
        assert 'scale="1.0 1.0 1.0"' in exported
        geometry = reparsed.get_link("base_link").visuals[0].geometry
        assert (geometry.scale.x, geometry.scale.y, geometry.scale.z) == (1.0, 1.0, 1.0)


class TestGazeboSensorBoundary:
    def test_gazebo_sensor_is_both_typed_and_captured(self):
        """Documented boundary: schema.sensors drives Cyberwave, the block drives Gazebo."""
        schema = _parse(ROS_URDF)
        assert [sensor.name for sensor in schema.sensors] == ["wrist_camera"]
        gazebo_xml = schema.extensions[URDF_PASSTHROUGH_KEY]["gazebo"][0]["xml"]
        assert "wrist_camera" in gazebo_xml


class TestDanglingMimicExport:
    """A `<mimic>` naming a joint the schema does not declare must not be emitted.

    The MJCF exporter already skips it (and its fallback motor), so emitting it here
    made one schema produce a coupled gripper in the served URDF and a free, undamped,
    uncommandable hinge in MJCF. A `<mimic joint="...">` pointing at nothing also makes
    the URDF unloadable for most parsers.
    """

    _URDF = """<?xml version="1.0"?>
<robot name="gripper">
  <link name="base"/><link name="l1"/><link name="l2"/>
  <joint name="driver" type="revolute">
    <parent link="base"/><child link="l1"/><axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="10" velocity="2"/>
  </joint>
  <joint name="follower" type="revolute">
    <parent link="base"/><child link="l2"/><axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="10" velocity="2"/>
    <mimic joint="driver" multiplier="-1" offset="0"/>
  </joint>
</robot>"""

    def test_a_resolvable_mimic_is_still_emitted(self):
        assert "<mimic" in _export(_parse(self._URDF))

    def test_a_dangling_mimic_is_omitted(self):
        schema = _parse(self._URDF)
        # Exactly what a patch that renames or removes the driver leaves behind.
        schema.get_joint("follower").mimic.joint = "deleted_by_a_patch"

        exported = _export(schema)

        assert "deleted_by_a_patch" not in exported
        assert "<mimic" not in exported


class TestParseIssuesDoNotGate:
    """A coerced parse must stay exportable and patchable.

    ``validate()`` is a hard gate: ``mujoco_scene_utils``, ``environments_exports``,
    ``universal_schema_patch`` and four robot-format exporters all turn a non-empty
    result into a raise. And ``to_dict`` serializes ``extensions``, so a parse issue
    persists into a stored ``universal_schema`` and survives ``from_dict`` — routing
    parse issues through ``validate()`` therefore makes one bad ``<mimic>`` permanently
    un-exportable *and* un-patchable for that asset, including assets already in the
    database. They belong in ``parse_issues()``, which nothing gates on.
    """

    _URDF = """<?xml version="1.0"?>
<robot name="coerced">
  <link name="base_link"/><link name="a_link"/><link name="b_link"/>
  <joint name="driver" type="revolute">
    <parent link="base_link"/><child link="a_link"/><axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="10" velocity="2"/>
  </joint>
  <joint name="follower" type="revolute">
    <parent link="a_link"/><child link="b_link"/><axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="10" velocity="2"/>
    <mimic joint="driver" multiplier="inf" offset="0"/>
  </joint>
</robot>"""

    def test_a_coerced_parse_still_validates(self):
        schema = _parse(self._URDF)

        assert schema.parse_issues() != []
        assert schema.validate() == []

    def test_the_issue_survives_a_round_trip_but_still_does_not_gate(self):
        """The stored-schema case: this is what makes a gate permanent rather than
        transient."""
        schema = _parse(self._URDF)

        restored = CommonSchema.from_dict(schema.to_dict())

        assert restored.parse_issues() == schema.parse_issues()
        assert restored.validate() == []

    def test_a_coerced_schema_still_exports(self):
        assert "<robot" in _export(_parse(self._URDF))


NAMESPACED_URDF = """<?xml version="1.0"?>
<robot name="ns_robot">
  <link name="ur5:base_link"/>
  <link name="ur5:tool"/>
  <joint name="ur5:shoulder" type="revolute">
    <parent link="ur5:base_link"/>
    <child link="ur5:tool"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="10" velocity="1"/>
  </joint>
  <ros2_control name="GazeboSimSystem" type="system">
    <hardware><plugin>gz_ros2_control/GazeboSimSystem</plugin></hardware>
    <joint name="ur5:shoulder">
      <command_interface name="position"/>
    </joint>
  </ros2_control>
  <transmission name="trans_1">
    <joint name="ur5:shoulder"><hardwareInterface>PositionJointInterface</hardwareInterface></joint>
  </transmission>
  <gazebo reference="ur5:base_link">
    <material>Gazebo/Grey</material>
  </gazebo>
</robot>
"""


class TestPassthroughNameReferencesFollowTheSanitizedNames:
    """Captured blocks must reference the joints/links the exported file declares.

    The parser runs every link and joint name through ``sanitize_name``, which
    rewrites ``:``, ``.``, ``/`` and spaces — spellings xacro namespaces produce
    routinely. Passthrough is stored verbatim, so a ``<transmission>`` kept pointing
    at ``ur5:shoulder`` while the exported ``<joint>`` became ``ur5_shoulder``: the
    replayed block named a joint absent from the very file it ships in, and
    ros2_control fails to load the controller.
    """

    def test_captured_blocks_use_the_sanitized_joint_and_link_names(self):
        schema = _parse(NAMESPACED_URDF)
        passthrough = schema.extensions[URDF_PASSTHROUGH_KEY]

        assert 'name="ur5_shoulder"' in passthrough["ros2_control"][0]["xml"]
        assert "ur5:shoulder" not in passthrough["ros2_control"][0]["xml"]
        assert 'name="ur5_shoulder"' in passthrough["transmission"][0]["xml"]
        assert "ur5:shoulder" not in passthrough["transmission"][0]["xml"]
        assert 'reference="ur5_base_link"' in passthrough["gazebo"][0]["xml"]

    def test_every_reference_in_the_export_resolves_to_a_declared_joint_or_link(self):
        exported = _export(_parse(NAMESPACED_URDF))
        root = ET.fromstring(exported)

        declared_joints = {j.get("name") for j in root.findall("joint")}
        declared_links = {ln.get("name") for ln in root.findall("link")}

        for block in (*root.findall("ros2_control"), *root.findall("transmission")):
            for joint in block.iter("joint"):
                assert joint.get("name") in declared_joints, (
                    f"{joint.get('name')!r} is referenced by a replayed block but the "
                    f"exported robot declares {sorted(declared_joints)}"
                )
        for gazebo in root.findall("gazebo"):
            reference = gazebo.get("reference")
            if reference:
                assert reference in declared_links

    def test_a_robot_needing_no_sanitizing_still_round_trips_unchanged(self):
        """The verbatim contract must hold for the (overwhelmingly common) clean case."""
        original, exported, reparsed = _roundtrip(ROS_URDF)

        assert (
            reparsed.extensions[URDF_PASSTHROUGH_KEY]
            == original.extensions[URDF_PASSTHROUGH_KEY]
        )
