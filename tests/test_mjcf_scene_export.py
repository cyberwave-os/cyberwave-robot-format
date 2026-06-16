import io
import zipfile
import xml.etree.ElementTree as ET

import pytest

from cyberwave_robot_format.mjcf import export_mujoco_zip_cloud
from cyberwave_robot_format.schema import (
    Collision,
    CommonSchema,
    Geometry,
    GeometryType,
    Inertia,
    Joint,
    JointType,
    Link,
    Metadata,
    Pose,
    Vector3,
    Visual,
)


def _mesh_schema() -> CommonSchema:
    mesh_geometry = Geometry(
        type=GeometryType.MESH,
        filename="meshes/missing_body.dae",
        scale=Vector3(1.0, 1.0, 1.0),
    )
    return CommonSchema(
        metadata=Metadata(name="missing_mesh_scene"),
        links=[
            Link(
                name="base",
                mass=1.0,
                inertia=Inertia(ixx=0.1, iyy=0.1, izz=0.1),
                visuals=[Visual(name="body_visual", geometry=mesh_geometry)],
                collisions=[Collision(name="body_collision", geometry=mesh_geometry)],
            )
        ],
    )


def test_export_mujoco_zip_cloud_falls_back_when_mesh_is_missing() -> None:
    zip_bytes = export_mujoco_zip_cloud(_mesh_schema(), mesh_resolver=lambda _: None)

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        xml_content = zf.read("mujoco_scene.xml")
        assert not any(name.startswith("assets/") for name in zf.namelist())

    root = ET.fromstring(xml_content)
    assert root.findall(".//asset/mesh") == []

    geoms = root.findall(".//worldbody//geom")
    assert all(geom.get("mesh") is None for geom in geoms)
    collision_geoms = [
        geom
        for geom in geoms
        if geom.get("class") == "collision" or geom.get("group") == "3"
    ]
    assert len(collision_geoms) == 1
    assert collision_geoms[0].get("type") == "box"
    assert collision_geoms[0].get("size") == "0.025 0.025 0.025"


def test_export_mujoco_zip_cloud_can_require_meshes() -> None:
    with pytest.raises(FileNotFoundError, match="missing_body.dae"):
        export_mujoco_zip_cloud(
            _mesh_schema(),
            mesh_resolver=lambda _: None,
            strict_missing_meshes=True,
        )


def test_export_mujoco_zip_cloud_adds_valid_inertial_for_moving_zero_mass_link() -> None:
    schema = CommonSchema(
        metadata=Metadata(name="zero_mass_joint_scene"),
        links=[
            Link(
                name="base",
                mass=1.0,
                inertia=Inertia(ixx=0.1, iyy=0.1, izz=0.1),
            ),
            Link(name="wheel", mass=0.0),
        ],
        joints=[
            Joint(
                name="wheel_joint",
                type=JointType.CONTINUOUS,
                parent_link="base",
                child_link="wheel",
                pose=Pose(position=Vector3(0.1, 0.0, 0.0)),
            )
        ],
    )

    zip_bytes = export_mujoco_zip_cloud(schema, mesh_resolver=lambda _: None)

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        root = ET.fromstring(zf.read("mujoco_scene.xml"))

    wheel_body = root.find(".//body[@name='wheel']")
    assert wheel_body is not None
    inertial = wheel_body.find("inertial")
    assert inertial is not None
    assert inertial.get("pos") == "0 0 0"
    assert inertial.get("mass") == "1e-4"
    assert inertial.get("diaginertia") == "1e-8 1e-8 1e-8"
