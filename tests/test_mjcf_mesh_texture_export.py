"""A mesh's own texture has to reach MuJoCo, or the model renders in grey.

MuJoCo cannot read DAE, so the exporter converts to OBJ; the converter writes
the mesh's base-colour map out beside it and an `.mtl` that references it. MuJoCo
does not read `.mtl`, so unless the MJCF declares the image itself the texture
sits unused on disk and the geom falls back to the default grey — which is what
made a fully textured inspection asset render flat in simulation.
"""

from pathlib import Path

import pytest

from cyberwave_robot_format.mesh import converted_mesh_texture
from cyberwave_robot_format.mjcf import MJCFExporter
from cyberwave_robot_format.schema import (
    Collision,
    CommonSchema,
    Geometry,
    GeometryType,
    Link,
    Metadata,
    Vector3,
    Visual,
)


def _mesh_schema(filename: str) -> CommonSchema:
    return CommonSchema(
        metadata=Metadata(name="textured"),
        links=[
            Link(
                name="base_link",
                visuals=[
                    Visual(
                        geometry=Geometry(
                            type=GeometryType.MESH,
                            filename=filename,
                            scale=Vector3(x=1.0, y=1.0, z=1.0),
                        )
                    )
                ],
            )
        ],
    )


def test_the_texture_beside_a_converted_mesh_is_found(tmp_path: Path) -> None:
    obj = tmp_path / "Capture.obj"
    obj.write_text("o m\n")
    (tmp_path / "material.mtl").write_text("newmtl m\nmap_Kd material_0.png\n")
    (tmp_path / "material_0.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    found = converted_mesh_texture(str(obj))

    # Absolute, because the MJCF is written somewhere else entirely.
    assert found == str(tmp_path / "material_0.png")
    assert Path(found).is_absolute()


def test_each_conversion_gets_its_own_directory(tmp_path: Path) -> None:
    """trimesh names sidecars after the material, so a shared output directory
    had every textured mesh overwriting the previous one's image. Two meshes
    converted into one output_dir must land in separate directories."""
    import trimesh

    from cyberwave_robot_format.mesh import convert_dae_to_obj

    output_dir = tmp_path / "converted"
    objs = []
    for name in ("first", "second"):
        source = tmp_path / f"{name}.dae"
        trimesh.creation.box().export(str(source))
        objs.append(Path(convert_dae_to_obj(str(source), None, output_dir)))

    # Separate directories -- not merely separate .obj names, which is all the
    # hashed filename ever gave: the sidecars are named by trimesh, not by us.
    assert objs[0].parent != objs[1].parent
    assert objs[0].parent.parent == objs[1].parent.parent == output_dir


def test_two_meshes_do_not_share_one_texture(tmp_path: Path) -> None:
    """A texture is read from its own mesh's directory, never a neighbour's."""
    first = tmp_path / "a_hash1"
    second = tmp_path / "b_hash2"
    for d, colour in ((first, b"one"), (second, b"two")):
        d.mkdir()
        (d / "m.obj").write_text("o m\n")
        (d / "material.mtl").write_text("newmtl m\nmap_Kd material_0.png\n")
        (d / "material_0.png").write_bytes(colour)

    a = converted_mesh_texture(str(first / "m.obj"))
    b = converted_mesh_texture(str(second / "m.obj"))

    assert a != b
    assert Path(a).read_bytes() != Path(b).read_bytes()


def test_a_mesh_with_no_texture_beside_it_resolves_to_nothing(tmp_path: Path) -> None:
    obj = tmp_path / "plain.obj"
    obj.write_text("o m\n")

    assert converted_mesh_texture(str(obj)) is None


def test_a_textured_mesh_reaches_mujoco_as_texture_material_and_geom_ref(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The three parts MuJoCo needs, and the geom that ties them together."""
    converted = tmp_path / "Capture_abc.obj"
    converted.write_text("o m\n")
    texture = tmp_path / "material_0.png"
    texture.write_bytes(b"\x89PNG\r\n\x1a\n")

    monkeypatch.setattr(
        "cyberwave_robot_format.mjcf.exporter.convert_dae_to_obj",
        lambda filename, scale=None: str(converted),
    )
    monkeypatch.setattr(
        "cyberwave_robot_format.mjcf.exporter.converted_mesh_texture",
        lambda obj_path: str(texture),
    )

    out = tmp_path / "model.xml"
    MJCFExporter().export(_mesh_schema("Capture.dae"), out)
    xml = out.read_text()

    assert f'<texture name="Capture_tex" type="2d" file="{texture}"' in xml
    assert '<material name="Capture_mat" texture="Capture_tex"' in xml
    # Without this the assets are declared and nothing uses them.
    assert 'material="Capture_mat"' in xml


def test_an_untextured_mesh_declares_no_texture_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No texture found means the MJCF is byte-for-byte what it always was."""
    converted = tmp_path / "plain_abc.obj"
    converted.write_text("o m\n")
    monkeypatch.setattr(
        "cyberwave_robot_format.mjcf.exporter.convert_dae_to_obj",
        lambda filename, scale=None: str(converted),
    )
    monkeypatch.setattr(
        "cyberwave_robot_format.mjcf.exporter.converted_mesh_texture",
        lambda obj_path: None,
    )

    out = tmp_path / "model.xml"
    MJCFExporter().export(_mesh_schema("plain.dae"), out)
    xml = out.read_text()

    assert "<texture" not in xml
    assert "_mat" not in xml


def test_a_collision_geom_does_not_wear_the_visual_texture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A collision hull carrying the visual's texture is indistinguishable from
    the visual exactly when collision display is on to tell them apart."""
    converted = tmp_path / "Capture_abc.obj"
    converted.write_text("o m\n")
    texture = tmp_path / "material_0.png"
    texture.write_bytes(b"\x89PNG\r\n\x1a\n")
    monkeypatch.setattr(
        "cyberwave_robot_format.mjcf.exporter.convert_dae_to_obj",
        lambda filename, scale=None: str(converted),
    )
    monkeypatch.setattr(
        "cyberwave_robot_format.mjcf.exporter.converted_mesh_texture",
        lambda obj_path: str(texture),
    )

    def geometry() -> Geometry:
        return Geometry(
            type=GeometryType.MESH,
            filename="Capture.dae",
            scale=Vector3(x=1.0, y=1.0, z=1.0),
        )

    schema = CommonSchema(
        metadata=Metadata(name="textured"),
        links=[
            Link(
                name="base_link",
                visuals=[Visual(geometry=geometry())],
                collisions=[Collision(name="c0", geometry=geometry())],
            )
        ],
    )

    out = tmp_path / "model.xml"
    MJCFExporter().export(schema, out)
    geoms = [
        line for line in out.read_text().splitlines() if "type=\"mesh\"" in line
    ]

    visual = next(g for g in geoms if 'class="visual"' in g)
    collision = next(g for g in geoms if 'class="collision"' in g)
    assert 'material="Capture_mat"' in visual
    assert "material=" not in collision
