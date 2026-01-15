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

"""Unit tests for URDFExporter."""

import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from cyberwave_robot_format.schema import (
    CommonSchema,
    Joint,
    JointType,
    Link,
    Metadata,
    Pose,
    Vector3,
)
from cyberwave_robot_format.urdf import URDFExporter


class TestURDFExporterLinkValidation:
    """Tests for link validation in URDFExporter."""

    def test_error_when_joint_references_nonexistent_parent_link(self):
        """Test that an error is raised when a joint references a non-existent parent link."""
        schema = CommonSchema(
            metadata=Metadata(name="test_robot"),
            links=[
                Link(name="base_link", mass=1.0),
            ],
            joints=[
                Joint(
                    name="world_to_base",
                    type=JointType.FIXED,
                    parent_link="world",  # Does not exist
                    child_link="base_link",
                    pose=Pose(position=Vector3(x=0.0, y=0.0, z=0.0)),
                ),
            ],
        )

        with tempfile.NamedTemporaryFile(suffix=".urdf", delete=False) as f:
            output_path = f.name

        try:
            exporter = URDFExporter()
            with pytest.raises(ValueError, match="non-existent parent link 'world'"):
                exporter.export(schema, output_path)
        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_error_when_joint_references_nonexistent_child_link(self):
        """Test that an error is raised when a joint references a non-existent child link."""
        schema = CommonSchema(
            metadata=Metadata(name="test_robot"),
            links=[
                Link(name="base_link", mass=1.0),
            ],
            joints=[
                Joint(
                    name="base_to_missing",
                    type=JointType.FIXED,
                    parent_link="base_link",
                    child_link="missing_link",  # Does not exist
                    pose=Pose(position=Vector3(x=0.0, y=0.0, z=0.0)),
                ),
            ],
        )

        with tempfile.NamedTemporaryFile(suffix=".urdf", delete=False) as f:
            output_path = f.name

        try:
            exporter = URDFExporter()
            with pytest.raises(ValueError, match="non-existent child link 'missing_link'"):
                exporter.export(schema, output_path)
        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_valid_export_when_all_links_exist(self):
        """Test that export succeeds when all referenced links exist."""
        schema = CommonSchema(
            metadata=Metadata(name="test_robot"),
            links=[
                Link(name="world", mass=0.0),
                Link(name="base_link", mass=1.0),
            ],
            joints=[
                Joint(
                    name="world_to_base",
                    type=JointType.FIXED,
                    parent_link="world",
                    child_link="base_link",
                    pose=Pose(position=Vector3(x=0.0, y=0.0, z=0.0)),
                ),
            ],
        )

        with tempfile.NamedTemporaryFile(suffix=".urdf", delete=False) as f:
            output_path = f.name

        try:
            exporter = URDFExporter()
            exporter.export(schema, output_path)

            # Parse the generated URDF
            tree = ET.parse(output_path)
            root = tree.getroot()

            # Check both links exist
            links = root.findall("link")
            link_names = {link.get("name") for link in links}
            assert link_names == {"world", "base_link"}

        finally:
            Path(output_path).unlink(missing_ok=True)


class TestURDFExporterJointTypes:
    """Tests for joint type mapping in URDFExporter."""

    @pytest.mark.parametrize(
        "joint_type,expected_urdf_type",
        [
            (JointType.REVOLUTE, "revolute"),
            (JointType.CONTINUOUS, "continuous"),
            (JointType.PRISMATIC, "prismatic"),
            (JointType.FIXED, "fixed"),
            (JointType.FLOATING, "floating"),
            (JointType.PLANAR, "planar"),
            (JointType.SPHERICAL, "continuous"),  # Approximated
            (JointType.UNIVERSAL, "continuous"),  # Approximated
        ],
    )
    def test_joint_type_mapping(self, joint_type, expected_urdf_type):
        """Test that joint types are correctly mapped to URDF types."""
        schema = CommonSchema(
            metadata=Metadata(name="test_robot"),
            links=[
                Link(name="link1", mass=1.0),
                Link(name="link2", mass=0.5),
            ],
            joints=[
                Joint(
                    name="test_joint",
                    type=joint_type,
                    parent_link="link1",
                    child_link="link2",
                    pose=Pose(position=Vector3(x=0.0, y=0.0, z=0.1)),
                ),
            ],
        )

        with tempfile.NamedTemporaryFile(suffix=".urdf", delete=False) as f:
            output_path = f.name

        try:
            exporter = URDFExporter()
            exporter.export(schema, output_path)

            # Parse the generated URDF
            tree = ET.parse(output_path)
            root = tree.getroot()

            # Check joint type
            joints = root.findall("joint")
            assert len(joints) == 1
            assert joints[0].get("type") == expected_urdf_type

        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_floating_joint_for_mobile_base(self):
        """Test that FLOATING joint type creates a proper floating joint in URDF."""
        schema = CommonSchema(
            metadata=Metadata(name="mobile_robot"),
            links=[
                Link(name="world", mass=0.0),
                Link(name="base_link", mass=10.0),
            ],
            joints=[
                Joint(
                    name="world_to_base",
                    type=JointType.FLOATING,
                    parent_link="world",
                    child_link="base_link",
                    pose=Pose(position=Vector3(x=0.0, y=0.0, z=0.0)),
                ),
            ],
        )

        with tempfile.NamedTemporaryFile(suffix=".urdf", delete=False) as f:
            output_path = f.name

        try:
            exporter = URDFExporter()
            exporter.export(schema, output_path)

            # Parse the generated URDF
            tree = ET.parse(output_path)
            root = tree.getroot()

            # Check joint is floating type
            joints = root.findall("joint")
            assert len(joints) == 1
            assert joints[0].get("type") == "floating"
            assert joints[0].get("name") == "world_to_base"

        finally:
            Path(output_path).unlink(missing_ok=True)
