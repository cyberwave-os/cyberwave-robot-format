"""A mesh's own texture has to reach MuJoCo, or the model renders in grey.

MuJoCo cannot read DAE, so the exporter converts to OBJ; the converter writes
the mesh's base-colour map out beside it and an `.mtl` that references it. MuJoCo
does not read `.mtl`, so unless the MJCF declares the image itself the texture
sits unused on disk and the geom falls back to the default grey — which is what
made a fully textured inspection asset render flat in simulation.
"""

import logging
from pathlib import Path
from xml.etree import ElementTree as ET

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
    Pose,
    Vector3,
    Visual,
)


@pytest.mark.parametrize("directory_key", ["assetdir", "texturedir"])
def test_imported_texture_survives_schema_json_and_mujoco_compile(tmp_path: Path, directory_key: str) -> None:
    from xml.etree import ElementTree as ET

    from cyberwave_robot_format.mjcf import MJCFParser

    mujoco = pytest.importorskip("mujoco")
    image = pytest.importorskip("PIL.Image")
    images = tmp_path / "images"
    images.mkdir()
    image.new("RGB", (2, 2), color=(30, 60, 90)).save(images / "body.png")
    source = tmp_path / "source.xml"
    source.write_text(f"""<mujoco><compiler {directory_key}="images"/>
      <asset><texture type="2d" file="body.png"/>
        <material name="first" texture="body"/><material name="second" texture="body"/>
      </asset><worldbody><body name="robot"><freejoint/>
        <geom size=".1" material="first"/><geom size=".05" pos=".2 0 0" material="second"/>
      </body></worldbody></mujoco>""")
    schema = CommonSchema.from_dict(MJCFParser().parse(source).to_dict())
    destination = tmp_path / "elsewhere" / "model.xml"
    destination.parent.mkdir()
    MJCFExporter().export(schema, destination)
    root = ET.parse(destination).getroot()
    assert len(root.findall("asset/texture")) == 1
    assert root.find("asset/texture").get("file") == str(images / "body.png")
    model = mujoco.MjModel.from_xml_path(str(destination))
    assert model.ntex == 1 and model.nmat == 2
    assert model.tex_data.tolist() == [30, 60, 90] * 4

    # The same declaration must survive the cloud resolver/ZIP path, where
    # importing local files behind the resolver would cross a trust boundary.
    import io
    import zipfile

    from cyberwave_robot_format.mjcf.scene_export import export_mujoco_zip_cloud
    from cyberwave_robot_format.urdf import URDFExporter

    resolved = []

    def resolve(filename):
        resolved.append(filename)
        assert filename == str(images / "body.png")
        return "body.png", (images / "body.png").read_bytes()

    bundle = export_mujoco_zip_cloud(schema, resolve, strict_missing_meshes=True)
    assert resolved == [str(images / "body.png")]
    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        assert "assets/body.png" in archive.namelist()
        packed = ET.fromstring(archive.read("mujoco_scene.xml"))
        assert packed.find("asset/texture").get("file") == "body.png"
    with pytest.raises(FileNotFoundError, match="Missing texture"):
        export_mujoco_zip_cloud(schema, lambda _: None)
    urdf = tmp_path / "robot.urdf"
    URDFExporter().export(schema, urdf)
    assert ET.parse(urdf).find(".//material/texture").get("filename") == str(images / "body.png")

    # Different source assets commonly use the same material/texture names.
    # Composition must namespace consumers and declarations together.
    scene = CommonSchema(metadata=Metadata(name="two_assets"))
    # Remove the imported world root; merge_in owns scene placement.
    schema.links = [item for item in schema.links if item.name != "world"]
    schema.joints = []
    for name in ("one", "two"):
        scene.merge_in(schema, name, Pose(), fixed_base=True)
    merged = tmp_path / "merged.xml"
    MJCFExporter().export(scene, merged)
    combined = ET.parse(merged).getroot()
    assert {item.get("name") for item in combined.findall("asset/texture")} == {"one__body", "two__body"}
    assert {item.get("texture") for item in combined.findall("asset/material")} == {"one__body", "two__body"}
    assert schema.links[0].visuals[0].material.texture == "body"


def _local_dae(directory: Path, stem: str) -> str:
    """A DAE that exists on disk, so the exporter attempts local conversion."""
    source = directory / f"{stem}.dae"
    source.write_text("")
    return str(source)


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
    MJCFExporter().export(_mesh_schema(_local_dae(tmp_path, "Capture")), out)
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
    MJCFExporter().export(_mesh_schema(_local_dae(tmp_path, "plain")), out)
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

    source = _local_dae(tmp_path, "Capture")

    def geometry() -> Geometry:
        return Geometry(
            type=GeometryType.MESH,
            filename=source,
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


@pytest.mark.parametrize(
    "filename",
    [
        "asset://0b1c2d3e-0000-4000-8000-000000000000/../robotiq_description/meshes/visual/robotiq_base.dae",
        "https://example.com/meshes/robotiq_base.dae",
        "package://robotiq_description/meshes/visual/robotiq_base.dae",
    ],
)
def test_a_dae_uri_is_left_to_the_mesh_resolver_without_a_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    filename: str,
) -> None:
    """Cloud exports reference meshes as ``asset://`` URIs that only the scene
    packager's resolver can fetch. Handing one to trimesh failed with "string is
    not a file" and logged a warning for every Robotiq DAE on every export."""
    calls: list[str] = []
    monkeypatch.setattr(
        "cyberwave_robot_format.mjcf.exporter.convert_dae_to_obj",
        lambda name, scale=None: calls.append(name) or name,
    )

    out = tmp_path / "model.xml"
    with caplog.at_level("DEBUG", logger="cyberwave_robot_format.mjcf.exporter"):
        MJCFExporter().export(_mesh_schema(filename), out)

    assert calls == []
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("Deferring DAE conversion" in r.getMessage() for r in caplog.records)
    # The reference is kept verbatim for the resolver to rewrite.
    mesh = ET.parse(out).find("asset/mesh")
    assert mesh.get("file") == filename
    assert mesh.get("name") == "robotiq_base"


def test_a_deferred_dae_keeps_its_scale_for_the_resolver(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The resolver converts a deferred mesh with ``scale=None``, so the MJCF
    ``scale`` attribute is the only surviving carrier of its size. A converted
    mesh is the mirror image: the converter bakes the scale into the vertices,
    so the attribute must read back as unity or the mesh is scaled twice."""
    converted_dir = tmp_path / "converted"
    passed_scales: list[Vector3 | None] = []

    def fake_convert(name: str, scale: Vector3 | None = None) -> str:
        passed_scales.append(scale)
        return str(converted_dir / "base.obj")

    monkeypatch.setattr("cyberwave_robot_format.mjcf.exporter.convert_dae_to_obj", fake_convert)
    scale = Vector3(x=0.001, y=0.002, z=0.003)
    schema = _mesh_schema("asset://0b1c2d3e-0000-4000-8000-000000000000/meshes/base.dae")
    schema.links[0].visuals[0].geometry.scale = scale

    deferred = tmp_path / "deferred.xml"
    MJCFExporter().export(schema, deferred)
    assert passed_scales == []
    assert ET.parse(deferred).find("asset/mesh").get("scale") == "0.001 0.002 0.003"

    schema.links[0].visuals[0].geometry.filename = _local_dae(tmp_path, "base")
    converted = tmp_path / "converted.xml"
    MJCFExporter().export(schema, converted)
    assert passed_scales == [scale]
    assert ET.parse(converted).find("asset/mesh").get("scale") == "1 1 1"


def test_a_missing_local_dae_is_reported_not_silently_deferred(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A plain path that is not on disk still warns: in a direct export nothing
    downstream resolves it, so the exporter is the only place that can name the
    mesh. The message must not call it broken, though, because a cloud export
    whose schema was not resource-scoped reaches here with a relative path that
    its resolver then fetches from storage."""
    calls: list[str] = []

    def convert(name: str, scale: object = None) -> str:
        calls.append(name)
        raise FileNotFoundError(f"DAE mesh is not a file on this host: {name}")

    monkeypatch.setattr("cyberwave_robot_format.mjcf.exporter.convert_dae_to_obj", convert)
    filename = str(tmp_path / "definitely/not/on/disk/robotiq_base.dae")

    out = tmp_path / "model.xml"
    with caplog.at_level("DEBUG", logger="cyberwave_robot_format.mjcf.exporter"):
        MJCFExporter().export(_mesh_schema(filename), out)

    assert calls == [filename]
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1
    assert filename in warnings[0].getMessage()
    assert "for a resolver to rewrite" in warnings[0].getMessage()
    assert not any("Deferring DAE conversion" in r.getMessage() for r in caplog.records)
    # The MJCF still carries the reference unchanged, as it did before.
    assert ET.parse(out).find("asset/mesh").get("file") == filename


def test_a_dae_path_the_os_refuses_to_stat_costs_one_warning_not_the_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``Path.is_file`` raises for some paths (a directory the server may not
    enter, a name too long for the filesystem). Before the on-disk check existed,
    every filesystem touch for a DAE sat under the conversion's catch-all, so one
    bad mesh cost a warning; the check must keep that, not abort the scene."""
    import functools

    from cyberwave_robot_format.mesh import convert_dae_to_obj

    monkeypatch.setattr(
        "cyberwave_robot_format.mjcf.exporter.convert_dae_to_obj",
        functools.partial(convert_dae_to_obj, output_dir=tmp_path / "converted"),
    )
    filename = str(tmp_path / "locked" / "robotiq_base.dae")
    real_is_file = Path.is_file
    refused: list[Path] = []

    def is_file(self: Path) -> bool:
        # Keyed on the name and recorded, so a later normalisation of the path
        # (a resolve, an expanduser) cannot quietly route this test onto the
        # missing-file branch instead.
        if self.name == Path(filename).name:
            refused.append(self)
            raise PermissionError(13, "Permission denied", filename)
        return real_is_file(self)

    monkeypatch.setattr(Path, "is_file", is_file)

    out = tmp_path / "model.xml"
    with caplog.at_level("WARNING", logger="cyberwave_robot_format.mjcf.exporter"):
        MJCFExporter().export(_mesh_schema(filename), out)

    assert refused, "the stat the test refuses was never attempted"
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(filename in m and "Permission denied" in m for m in warnings), warnings
    assert ET.parse(out).find("asset/mesh").get("file") == filename


def test_a_dae_that_is_not_on_disk_leaves_no_empty_directory(tmp_path: Path) -> None:
    from cyberwave_robot_format.mesh import convert_dae_to_obj

    output_dir = tmp_path / "converted"
    with pytest.raises(FileNotFoundError, match="not a file on this host"):
        convert_dae_to_obj(str(tmp_path / "missing.dae"), None, output_dir)

    assert not output_dir.exists()


def test_a_dae_that_cannot_be_loaded_leaves_no_empty_directory(tmp_path: Path) -> None:
    from cyberwave_robot_format.mesh import convert_dae_to_obj

    source = tmp_path / "corrupt.dae"
    source.write_text("not collada")
    output_dir = tmp_path / "converted"
    with pytest.raises(Exception):  # noqa: B017 - whatever trimesh raises
        convert_dae_to_obj(str(source), None, output_dir)

    assert not output_dir.exists()


def test_a_cached_conversion_outlives_its_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Once converted, a mesh whose source has since moved out of reach still
    exports as its cached OBJ, as it did before the on-disk check existed."""
    import functools

    import trimesh

    from cyberwave_robot_format.mesh import convert_dae_to_obj

    output_dir = tmp_path / "converted"
    monkeypatch.setattr(
        "cyberwave_robot_format.mjcf.exporter.convert_dae_to_obj",
        functools.partial(convert_dae_to_obj, output_dir=output_dir),
    )
    source = tmp_path / "box.dae"
    trimesh.creation.box().export(str(source))
    first = tmp_path / "first.xml"
    MJCFExporter().export(_mesh_schema(str(source)), first)
    cached = ET.parse(first).find("asset/mesh").get("file")
    assert cached.endswith("box.obj")

    source.unlink()
    second = tmp_path / "second.xml"
    with caplog.at_level("WARNING", logger="cyberwave_robot_format.mjcf.exporter"):
        MJCFExporter().export(_mesh_schema(str(source)), second)

    assert ET.parse(second).find("asset/mesh").get("file") == cached
    exporter_warnings = [
        r.getMessage()
        for r in caplog.records
        if r.name == "cyberwave_robot_format.mjcf.exporter" and r.levelno >= logging.WARNING
    ]
    assert exporter_warnings == []


def test_a_local_dae_is_still_converted_in_place(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The guard must not stop real on-disk DAEs from being converted."""
    import functools

    import trimesh

    from cyberwave_robot_format.mesh import convert_dae_to_obj

    # The real converter, pointed into the sandbox instead of the shared
    # /tmp/mujoco_converted_meshes default.
    output_dir = tmp_path / "converted"
    monkeypatch.setattr(
        "cyberwave_robot_format.mjcf.exporter.convert_dae_to_obj",
        functools.partial(convert_dae_to_obj, output_dir=output_dir),
    )
    source = tmp_path / "box.dae"
    trimesh.creation.box().export(str(source))
    out = tmp_path / "model.xml"
    MJCFExporter().export(_mesh_schema(str(source)), out)

    mesh_file = ET.parse(out).find("asset/mesh").get("file")
    assert mesh_file.endswith("box.obj")
    assert Path(mesh_file).is_file()
    assert Path(mesh_file).is_relative_to(output_dir)
