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

import io
import math
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pytest
from cyberwave_robot_format.mjcf import export_mujoco_zip_cloud
from cyberwave_robot_format.schema import (
    Collision,
    CommonSchema,
    Geometry,
    GeometryType,
    Joint,
    JointType,
    Link,
    Material,
    Metadata,
    Pose,
    Quaternion,
    Vector3,
    Visual,
)
from cyberwave_robot_format.urdf import URDFExporter, export_urdf_zip_cloud


@pytest.mark.parametrize("format_name", ["urdf", "mjcf"])
def test_scene_bundle_never_overwrites_a_pre_suffixed_filename(format_name):
    """foot, foot, foot_1 must remain three distinct resources after flattening."""
    files = {
        "asset://robot_a/foot.obj": ("foot.obj", b"robot a"),
        "asset://robot_b/foot.obj": ("foot.obj", b"robot b"),
        "asset://robot_c/foot_1.obj": ("foot_1.obj", b"robot c"),
    }
    schema = CommonSchema(metadata=Metadata(name="multiple_robots"))
    for i, source in enumerate([*files, next(iter(files))]):
        schema.merge_in(
            CommonSchema(
                metadata=Metadata(name="robot"),
                links=[
                    Link(
                        name="foot",
                        visuals=[Visual(geometry=Geometry(type=GeometryType.MESH, filename=source))],
                    )
                ],
            ),
            instance_name=f"robot_{i}",
            spawn_pose=Pose(),
        )
    calls = []

    def resolve(source):
        calls.append(source)
        return files[source]

    exporter = export_urdf_zip_cloud if format_name == "urdf" else export_mujoco_zip_cloud
    with zipfile.ZipFile(io.BytesIO(exporter(schema, resolve))) as bundle:
        if format_name == "urdf":
            root = ET.fromstring(bundle.read("scene.urdf"))
            paths = [root.find(f"link[@name='robot_{i}__foot']/visual/geometry/mesh").get("filename") for i in range(4)]
        else:
            root = ET.fromstring(bundle.read("mujoco_scene.xml"))
            meshes = {m.get("name"): m.get("file") for m in root.findall("asset/mesh")}
            paths = [
                "assets/" + meshes[root.find(f".//body[@name='robot_{i}__foot']/geom").get("mesh")] for i in range(4)
            ]
        assert [bundle.read(path) for path in paths] == [b"robot a", b"robot b", b"robot c", b"robot a"]
        assert len(set(paths)) == 3
    assert calls == list(files)
    assert schema.links[0].visuals[0].geometry.filename == next(iter(files))


@pytest.mark.parametrize("from_mjcf", [False, True])
def test_urdf_bundle_contains_the_declared_texture(from_mjcf):
    source = "asset://robot/meshes/body.png"
    material = Material(name="paint", texture="paint_image" if from_mjcf else source)
    if from_mjcf:
        material.extensions["mjcf_texture"] = {"name": "paint_image", "file": source}
    schema = CommonSchema(
        metadata=Metadata(name="textured"),
        links=[
            Link(
                name="base",
                visuals=[
                    Visual(
                        geometry=Geometry(type=GeometryType.BOX, size=Vector3(1, 1, 1)),
                        material=material,
                    )
                ],
            )
        ],
    )
    archive = export_urdf_zip_cloud(schema, lambda filename: ("body.png", b"image") if filename == source else None)
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        root = ET.fromstring(bundle.read("scene.urdf"))
        texture = root.find("link/visual/material/texture").get("filename")
        assert texture.startswith("assets/")
        assert bundle.read(texture) == b"image"
    assert material.texture == ("paint_image" if from_mjcf else source)


def test_urdf_bundle_reports_a_missing_declared_texture():
    schema = CommonSchema(
        metadata=Metadata(name="missing_texture"),
        links=[
            Link(
                name="base",
                visuals=[
                    Visual(
                        geometry=Geometry(type=GeometryType.BOX, size=Vector3(1, 1, 1)),
                        material=Material(name="paint", texture="asset://robot/missing.png"),
                    )
                ],
            )
        ],
    )
    with pytest.raises(FileNotFoundError, match="texture.*missing.png"):
        export_urdf_zip_cloud(schema, lambda _: None)


@pytest.mark.parametrize("factor", [1.0, 2.0, -3.0, 1e-200, 1e200])
def test_urdf_rotation_is_invariant_to_quaternion_magnitude(tmp_path, factor):
    import numpy as np

    orientation = Quaternion(0, factor, factor, 0)
    schema = CommonSchema(
        metadata=Metadata(name="scaled_rotation"),
        links=[
            Link(
                name="body",
                visuals=[
                    Visual(
                        pose=Pose(orientation=orientation),
                        geometry=Geometry(type=GeometryType.BOX, size=Vector3(1, 2, 3)),
                    )
                ],
            )
        ],
    )
    destination = tmp_path / "robot.urdf"
    URDFExporter().export(schema, destination)
    rpy = [float(value) for value in ET.parse(destination).find(".//visual/origin").get("rpy").split()]
    result = Quaternion.from_rpy(*rpy)
    expected = np.array([0, 1 / math.sqrt(2), 1 / math.sqrt(2), 0])
    assert abs(np.dot(result.to_list(), expected)) == pytest.approx(1)
    assert orientation.to_list() == [0, factor, factor, 0], "Export must not mutate the source"


@pytest.mark.parametrize("values", [[0, 0, 0, 0], [float("nan"), 0, 0, 1], [0, float("inf"), 0, 1]])
def test_invalid_rotation_is_not_exported_as_a_plausible_pose(values):
    with pytest.raises(ValueError, match="finite and nonzero"):
        Quaternion(*values).to_rpy()


class TestURDFExporterWorldLink:
    """Tests for world link injection in URDFExporter."""

    def test_world_link_injected_when_joint_references_world(self):
        """Test that a world link is automatically injected when a joint has parent_link='world'."""
        schema = CommonSchema(
            metadata=Metadata(name="test_robot"),
            links=[
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

            # Check that world link exists
            world_links = [
                link for link in root.findall("link") if link.get("name") == "world"
            ]
            assert len(world_links) == 1, "World link should be injected"

            # Check that base_link also exists
            base_links = [
                link for link in root.findall("link") if link.get("name") == "base_link"
            ]
            assert len(base_links) == 1, "base_link should exist"

            # Check the joint references world correctly
            joints = root.findall("joint")
            assert len(joints) == 1
            joint = joints[0]
            assert joint.find("parent").get("link") == "world"
            assert joint.find("child").get("link") == "base_link"

        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_world_link_not_duplicated_when_already_in_schema(self):
        """Test that world link is not duplicated if it already exists in the schema."""
        schema = CommonSchema(
            metadata=Metadata(name="test_robot"),
            links=[
                Link(name="world", mass=0.0),  # World link already in schema
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

            # Check that only one world link exists (not duplicated)
            world_links = [
                link for link in root.findall("link") if link.get("name") == "world"
            ]
            assert len(world_links) == 1, "World link should not be duplicated"

        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_no_world_link_when_not_needed(self):
        """Test that world link is not injected when no joint references it."""
        schema = CommonSchema(
            metadata=Metadata(name="test_robot"),
            links=[
                Link(name="base_link", mass=1.0),
                Link(name="link1", mass=0.5),
            ],
            joints=[
                Joint(
                    name="base_to_link1",
                    type=JointType.REVOLUTE,
                    parent_link="base_link",
                    child_link="link1",
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

            # Check that no world link exists
            world_links = [
                link for link in root.findall("link") if link.get("name") == "world"
            ]
            assert len(world_links) == 0, "World link should not be injected"

        finally:
            Path(output_path).unlink(missing_ok=True)


class TestQuaternionRpyRoundTrip:
    """Quaternion.to_rpy must invert from_rpy (Fixed XYZ)."""

    @pytest.mark.parametrize(
        "rpy",
        [
            (0.0, 0.0, 0.0),
            (0.3, 0.0, 0.0),
            (0.0, -0.4, 0.0),
            (0.0, 0.0, 1.2),
            (0.3, -0.4, 1.2),
            (-1.0, 0.5, -2.5),
        ],
    )
    def test_roundtrip(self, rpy):
        q = Quaternion.from_rpy(*rpy)
        out = q.to_rpy()
        for got, want in zip(out, rpy):
            assert math.isclose(got, want, abs_tol=1e-9)


class TestURDFExporterOrientation:
    """Origins must carry the pose orientation, not a hardcoded rpy='0 0 0'."""

    def _parse_rpy(self, elem: ET.Element) -> tuple[float, float, float]:
        origin = elem.find("origin")
        assert origin is not None
        return tuple(float(v) for v in origin.get("rpy").split())

    def test_joint_visual_collision_orientation_preserved(self):
        rpy = (0.3, -0.4, 1.2)
        geom = Geometry(type=GeometryType.BOX, size=Vector3(x=0.1, y=0.1, z=0.1))
        schema = CommonSchema(
            metadata=Metadata(name="rotated"),
            links=[
                Link(name="world", mass=0.0),
                Link(
                    name="prop",
                    mass=1.0,
                    visuals=[
                        Visual(
                            pose=Pose.from_xyzrpy([0, 0, 0], list(rpy)), geometry=geom
                        )
                    ],
                    collisions=[
                        Collision(
                            pose=Pose.from_xyzrpy([0, 0, 0], list(rpy)), geometry=geom
                        )
                    ],
                ),
            ],
            joints=[
                Joint(
                    name="world_to_prop",
                    type=JointType.FIXED,
                    parent_link="world",
                    child_link="prop",
                    pose=Pose.from_xyzrpy([1.0, 2.0, 3.0], list(rpy)),
                ),
            ],
        )

        with tempfile.NamedTemporaryFile(suffix=".urdf", delete=False) as f:
            output_path = f.name
        try:
            URDFExporter().export(schema, output_path)
            root = ET.parse(output_path).getroot()

            joint = root.find("joint")
            for got, want in zip(self._parse_rpy(joint), rpy):
                assert math.isclose(got, want, abs_tol=1e-6)

            prop = next(
                link for link in root.findall("link") if link.get("name") == "prop"
            )
            for got, want in zip(self._parse_rpy(prop.find("visual")), rpy):
                assert math.isclose(got, want, abs_tol=1e-6)
            for got, want in zip(self._parse_rpy(prop.find("collision")), rpy):
                assert math.isclose(got, want, abs_tol=1e-6)
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

            # Check world link is injected
            world_links = [
                link for link in root.findall("link") if link.get("name") == "world"
            ]
            assert len(world_links) == 1

            # Check joint is floating type
            joints = root.findall("joint")
            assert len(joints) == 1
            assert joints[0].get("type") == "floating"
            assert joints[0].get("name") == "world_to_base"

        finally:
            Path(output_path).unlink(missing_ok=True)


@pytest.mark.parametrize("pitch", [math.pi / 2, -math.pi / 2, math.pi / 2 - 1e-7, -math.pi / 2 + 1e-7])
@pytest.mark.parametrize("roll,yaw", [(0.0, 0.0), (0.7, -1.2), (-math.pi / 2, math.pi), (math.pi, math.pi / 2)])
@pytest.mark.parametrize("scale", [1.0, -1.0, 7.0])
def test_quaternion_gimbal_lock_preserves_rotation(pitch, roll, yaw, scale):
    source = Quaternion.from_rpy(roll, pitch, yaw)
    scaled = Quaternion(x=source.x * scale, y=source.y * scale, z=source.z * scale, w=source.w * scale)
    restored = Quaternion.from_rpy(*scaled.to_rpy())
    first = (source.x, source.y, source.z, source.w)
    second = (restored.x, restored.y, restored.z, restored.w)
    # Euler angles are not unique at the singularity; compare the rotation.
    error = min(max(abs(a - sign * b) for a, b in zip(first, second)) for sign in (1, -1))
    assert error < 1e-8
