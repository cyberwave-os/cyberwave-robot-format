# Copyright [2025] Cyberwave
# Licensed under the Apache License, Version 2.0

"""Tests for ``cyberwave_robot_format.usd.scene_to_glb``.

Stages are authored in code with ``pxr`` so each test states exactly what it feeds
the converter. The pure helpers (triangulation, MDL scraping) need neither ``pxr``
nor Pillow and run everywhere; everything else skips when either is missing.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import trimesh

from cyberwave_robot_format.usd import scene_to_glb as s2g
from cyberwave_robot_format.usd.scene_to_glb import USDSceneToGLB

# --------------------------------------------------------------------------
# Pure helpers (no pxr, no Pillow)
# --------------------------------------------------------------------------


class TestFanTriangulate:
    def test_quads_become_two_triangles_each(self):
        tris = s2g._fan_triangulate(np.array([4, 4]))
        assert tris.tolist() == [[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]]

    def test_mixed_polygon_sizes(self):
        tris = s2g._fan_triangulate(np.array([3, 5]))
        assert tris.tolist() == [[0, 1, 2], [3, 4, 5], [3, 5, 6], [3, 6, 7]]

    def test_degenerate_faces_yield_nothing(self):
        assert s2g._fan_triangulate(np.array([2, 1, 0])).shape == (0, 3)


class TestParseMdl:
    MDL = """
    export material Retaining_Block(*) = ::OmniPBR::OmniPBR(
        diffuse_color_constant: color(0.2f, 0.4f, 0.6f),
        diffuse_texture: texture_2d("./Retaining_Block/base.png" /* tag */, ::tex::gamma_srgb),
        reflection_roughness_constant: 0.5f,
        metallic_constant: 0.f,
        enable_ORM_texture: true,
        ORM_texture: texture_2d("./Retaining_Block/orm.png", ::tex::gamma_linear),
        normalmap_texture: texture_2d("./Retaining_Block/n.png", ::tex::gamma_linear),
        emissive_mask_texture: texture_2d(),
        enable_opacity: false
    );
    """

    def test_reads_constants_and_textures(self):
        v = s2g._parse_mdl(self.MDL)
        assert v["diffuse_color_constant"] == [0.2, 0.4, 0.6]
        assert v["reflection_roughness_constant"] == 0.5
        assert v["diffuse_texture"] == "./Retaining_Block/base.png"
        assert v["ORM_texture"] == "./Retaining_Block/orm.png"
        assert v["enable_ORM_texture"] is True
        assert v["enable_opacity"] is False

    def test_an_empty_texture_slot_is_not_a_texture(self):
        assert "emissive_mask_texture" not in s2g._parse_mdl(self.MDL)


# --------------------------------------------------------------------------
# Stage-backed tests
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pxr():
    pytest.importorskip("pxr")
    pytest.importorskip("PIL")
    from cyberwave_robot_format.usd import common

    return common.require_pxr()


_BOX_POINTS = [
    (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
    (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1),
]  # fmt: skip
# Quads wound counter-clockwise seen from outside, with matching outward normals.
_BOX_FACES = [
    ((0, 3, 2, 1), (0, 0, -1)),
    ((4, 5, 6, 7), (0, 0, 1)),
    ((0, 1, 5, 4), (0, -1, 0)),
    ((2, 3, 7, 6), (0, 1, 0)),
    ((1, 2, 6, 5), (1, 0, 0)),
    ((0, 4, 7, 3), (-1, 0, 0)),
]


def _author_box(stage, pxr, path="/World/Box", half=50.0):
    """A face-varying box mesh with normals and UVs, ``2 * half`` on a side."""
    Gf, UsdGeom, Vt, Sdf = pxr.Gf, pxr.UsdGeom, pxr.Vt, pxr.Sdf
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*(half * np.array(p))) for p in _BOX_POINTS])
    mesh.CreateFaceVertexCountsAttr([4] * 6)
    mesh.CreateFaceVertexIndicesAttr([i for quad, _ in _BOX_FACES for i in quad])
    mesh.CreateNormalsAttr([Gf.Vec3f(*n) for _, n in _BOX_FACES for _ in range(4)])
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
    )
    # v is deliberately asymmetric (0.25 / 1.0) so a flipped V is detectable.
    st.Set(Vt.Vec2fArray([Gf.Vec2f(u, v) for _ in range(6) for u, v in
                          ((0, 0.25), (1, 0.25), (1, 1), (0, 1))]))  # fmt: skip
    return mesh


def _stage(tmp_path, pxr, *, up="Y", mpu=0.01, name="scene.usda"):
    UsdGeom = pxr.UsdGeom
    path = tmp_path / name
    stage = pxr.Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, up)
    UsdGeom.SetStageMetersPerUnit(stage, mpu)
    UsdGeom.Xform.Define(stage, "/World")
    return stage, path


def _load(glb: bytes) -> trimesh.Scene:
    return trimesh.load(io.BytesIO(glb), file_type="glb")


def _signed_volume(geom: trimesh.Trimesh) -> float:
    t = geom.vertices[geom.faces]
    return float(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6)


class TestGeometry:
    def test_units_up_axis_and_triangulation(self, tmp_path, pxr):
        stage, path = _stage(tmp_path, pxr)  # Y-up, centimetres
        box = _author_box(stage, pxr)  # 100 cm cube
        pxr.UsdGeom.XformCommonAPI(box).SetTranslate((0, 50, 0))  # sits on y = 0
        stage.Save()

        z_up = USDSceneToGLB().convert(path)
        scene = _load(z_up.glb)
        assert z_up.report.triangle_count == 12  # 6 quads
        assert scene.bounds.round(3).tolist() == [[-0.5, -0.5, 0.0], [0.5, 0.5, 1.0]]

        y_up = USDSceneToGLB(z_up=False).convert(path)
        assert _load(y_up.glb).bounds.round(3).tolist() == [
            [-0.5, 0.0, -0.5], [0.5, 1.0, 0.5]
        ]  # fmt: skip

    def test_z_up_stage_is_left_alone_when_z_up_is_wanted(self, tmp_path, pxr):
        stage, path = _stage(tmp_path, pxr, up="Z", mpu=1.0)
        _author_box(stage, pxr, half=1.0)
        stage.Save()
        bounds = _load(USDSceneToGLB().convert(path).glb).bounds
        assert bounds.round(3).tolist() == [[-1, -1, -1], [1, 1, 1]]

    def test_uvs_survive(self, tmp_path, pxr):
        stage, path = _stage(tmp_path, pxr)
        _author_box(stage, pxr)
        stage.Save()
        geom = next(iter(_load(USDSceneToGLB().convert(path).glb).geometry.values()))
        assert geom.visual.uv is not None
        # trimesh flips V on export and again on load, so what comes back must be
        # what was authored; a second flip in the converter would give {0.75, 0.0}.
        assert {round(float(v), 2) for v in geom.visual.uv[:, 1]} == {0.25, 1.0}

    def test_mirrored_transform_keeps_faces_outward(self, tmp_path, pxr):
        stage, path = _stage(tmp_path, pxr)
        box = _author_box(stage, pxr)
        pxr.UsdGeom.XformCommonAPI(box).SetScale((-1, 1, 1))
        stage.Save()
        geom = next(iter(_load(USDSceneToGLB().convert(path).glb).geometry.values()))
        assert _signed_volume(geom) > 0  # inside-out would be negative

    def test_primitives_are_tessellated(self, tmp_path, pxr):
        stage, path = _stage(tmp_path, pxr, mpu=1.0)
        pxr.UsdGeom.Cube.Define(stage, "/World/Cube").CreateSizeAttr(2.0)
        sphere = pxr.UsdGeom.Sphere.Define(stage, "/World/Sphere")
        sphere.CreateRadiusAttr(0.5)
        stage.Save()
        report = USDSceneToGLB().convert(path).report
        assert report.mesh_count == 2

    @pytest.mark.parametrize(
        "kind, axis, expected_extents",
        [
            ("Cylinder", "Z", [1.0, 1.0, 2.0]),
            ("Cylinder", "X", [2.0, 1.0, 1.0]),
            ("Capsule", "Y", [1.0, 3.0, 1.0]),  # height 2 + two radius-0.5 caps
            ("Cone", "Y", [1.0, 2.0, 1.0]),
        ],
    )
    def test_axis_aligned_primitives_follow_their_axis(
        self, tmp_path, pxr, kind, axis, expected_extents
    ):
        stage, path = _stage(tmp_path, pxr, up="Z", mpu=1.0)  # no re-orientation
        shape = getattr(pxr.UsdGeom, kind).Define(stage, f"/World/{kind}")
        shape.CreateRadiusAttr(0.5)
        shape.CreateHeightAttr(2.0)
        shape.CreateAxisAttr(axis)
        stage.Save()
        bounds = _load(USDSceneToGLB().convert(path).glb).bounds
        assert (bounds[1] - bounds[0]).round(2).tolist() == expected_extents

    def test_a_scene_with_no_geometry_is_an_error(self, tmp_path, pxr):
        _, path = _stage(tmp_path, pxr)
        pxr.Usd.Stage.Open(str(path)).Save()
        with pytest.raises(ValueError, match="no renderable meshes"):
            USDSceneToGLB().convert(path)


class TestTriangleBudget:
    """Peak memory is ~0.75 GB per million triangles (measured), and a 48 MB scene
    can hold 8M of them. A scene over the budget must be refused, with a message
    the uploader can act on, instead of killing the worker that converts it."""

    def _two_boxes(self, tmp_path, pxr):
        stage, path = _stage(tmp_path, pxr)
        _author_box(stage, pxr, "/World/A")  # 6 quads = 12 triangles each
        _author_box(stage, pxr, "/World/B")
        stage.Save()
        return path

    def test_a_scene_over_the_budget_is_refused_with_the_limit_named(
        self, tmp_path, pxr
    ):
        path = self._two_boxes(tmp_path, pxr)
        with pytest.raises(ValueError, match="more than 10 triangles"):
            USDSceneToGLB(max_triangles=10).convert(path)

    def test_the_budget_is_cumulative_across_meshes(self, tmp_path, pxr):
        """Each box (12) fits alone; together (24) they do not."""
        path = self._two_boxes(tmp_path, pxr)
        with pytest.raises(ValueError, match="more than 20 triangles"):
            USDSceneToGLB(max_triangles=20).convert(path)

    def test_a_scene_exactly_at_the_budget_converts(self, tmp_path, pxr):
        path = self._two_boxes(tmp_path, pxr)
        result = USDSceneToGLB(max_triangles=24).convert(path)
        assert result.report.triangle_count == 24

    def test_the_default_budget_is_generous_for_ordinary_scenes(self, tmp_path, pxr):
        path = self._two_boxes(tmp_path, pxr)
        assert USDSceneToGLB().convert(path).report.mesh_count == 2

    def test_the_budget_must_be_positive(self):
        with pytest.raises(ValueError, match="max_triangles"):
            USDSceneToGLB(max_triangles=0)

    def test_primitives_count_too(self, tmp_path, pxr):
        """A scene of many small cubes (instanced points) must not slip past a
        budget that only watched meshes: six cubes are 72 triangles."""
        stage, path = _stage(tmp_path, pxr, mpu=1.0)
        for i in range(6):
            pxr.UsdGeom.Cube.Define(stage, f"/World/Cube{i}").CreateSizeAttr(1.0)
        stage.Save()
        with pytest.raises(ValueError, match="more than 50 triangles"):
            USDSceneToGLB(max_triangles=50).convert(path)
        assert USDSceneToGLB(max_triangles=72).convert(path).report.triangle_count == 72

    def test_meshes_and_primitives_share_one_budget(self, tmp_path, pxr):
        stage, path = _stage(tmp_path, pxr, mpu=1.0)
        _author_box(stage, pxr, "/World/Mesh")  # 12 triangles
        pxr.UsdGeom.Cube.Define(stage, "/World/Cube").CreateSizeAttr(1.0)  # 12 more
        stage.Save()
        with pytest.raises(ValueError, match="more than 20 triangles"):
            USDSceneToGLB(max_triangles=20).convert(path)


class TestVisibility:
    def _scene(self, tmp_path, pxr):
        stage, path = _stage(tmp_path, pxr)
        box = _author_box(stage, pxr)
        box.CreateVisibilityAttr(pxr.UsdGeom.Tokens.invisible)
        _author_box(stage, pxr, "/World/Visible")
        stage.Save()
        return path

    def test_hidden_prims_are_included_by_default(self, tmp_path, pxr):
        path = self._scene(tmp_path, pxr)
        assert USDSceneToGLB().convert(path).report.mesh_count == 2

    def test_hidden_prims_can_be_excluded(self, tmp_path, pxr):
        path = self._scene(tmp_path, pxr)
        result = USDSceneToGLB(include_invisible=False).convert(path)
        assert result.report.mesh_count == 1

    def test_guide_geometry_is_never_included(self, tmp_path, pxr):
        stage, path = _stage(tmp_path, pxr)
        guide = _author_box(stage, pxr)
        guide.CreatePurposeAttr(pxr.UsdGeom.Tokens.guide)
        _author_box(stage, pxr, "/World/Real")
        stage.Save()
        assert USDSceneToGLB().convert(path).report.mesh_count == 1


def _write_png(path: Path, size: int = 256, rgba=(200, 30, 30, 255)) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (size, size), rgba).save(path)


def _bind_preview_surface(stage, pxr, prim, texture: str | None, **inputs):
    UsdShade, Sdf = pxr.UsdShade, pxr.Sdf
    mat = UsdShade.Material.Define(stage, "/World/Looks/Mat")
    pbr = UsdShade.Shader.Define(stage, "/World/Looks/Mat/PBR")
    pbr.CreateIdAttr("UsdPreviewSurface")
    for name, value in inputs.items():
        pbr.CreateInput(name, Sdf.ValueTypeNames.Float).Set(value)
    if texture is not None:
        tex = UsdShade.Shader.Define(stage, "/World/Looks/Mat/Tex")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(texture))
        tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            tex.ConnectableAPI(), "rgb"
        )
    pbr.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    mat.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(mat)


def _bind_mdl_shader(stage, pxr, prim, name, source, **inputs):
    """Bind an MDL-style material whose shader inputs are ``(type, value)`` pairs."""
    UsdShade, Sdf = pxr.UsdShade, pxr.Sdf
    mat = UsdShade.Material.Define(stage, f"/World/Looks/{name}")
    sh = UsdShade.Shader.Define(stage, f"/World/Looks/{name}/Shader")
    sh.CreateImplementationSourceAttr("sourceAsset")
    sh.GetPrim().CreateAttribute("info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(source)
    )
    for key, (type_name, value) in inputs.items():
        sh.CreateInput(key, getattr(Sdf.ValueTypeNames, type_name)).Set(value)
    sh.CreateOutput("out", Sdf.ValueTypeNames.Token)
    mat.CreateOutput("mdl:surface", Sdf.ValueTypeNames.Token).ConnectToSource(
        sh.ConnectableAPI(), "out"
    )
    UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(mat)


def _material_of(glb: bytes):
    return next(iter(_load(glb).geometry.values())).visual.material


class TestOmniPbr:
    def test_textured_material_keeps_a_white_base_colour_factor(self, tmp_path, pxr):
        """Regression: an integer ``[1, 1, 1, 1]`` factor is read by trimesh as
        1/255 and multiplied the texture to black."""
        Sdf, Gf = pxr.Sdf, pxr.Gf
        stage, path = _stage(tmp_path, pxr)
        box = _author_box(stage, pxr)
        _write_png(tmp_path / "dial.png")
        _bind_mdl_shader(
            stage, pxr, box, "Dial", "OmniPBR.mdl",
            diffuse_color_constant=("Color3f", Gf.Vec3f(0.5, 0.5, 0.5)),
            diffuse_texture=("Asset", Sdf.AssetPath("dial.png")),
        )  # fmt: skip
        stage.Save()

        result = USDSceneToGLB().convert(path)
        mat = _material_of(result.glb)
        assert result.report.materials == {"Dial": "omnipbr"}
        assert mat.baseColorTexture is not None
        assert mat.baseColorFactor.tolist() == [255, 255, 255, 255]

    def test_black_chrome_gets_a_metal_preset(self, tmp_path, pxr):
        Gf = pxr.Gf
        stage, path = _stage(tmp_path, pxr)
        box = _author_box(stage, pxr)
        _bind_mdl_shader(
            stage, pxr, box, "Chrome_1", "OmniPBR.mdl",
            diffuse_color_constant=("Color3f", Gf.Vec3f(0, 0, 0)),
        )  # fmt: skip
        stage.Save()

        mat = _material_of(USDSceneToGLB().convert(path).glb)
        assert mat.metallicFactor == 1.0
        assert mat.baseColorFactor[0] > 200  # not black

        plain = _material_of(USDSceneToGLB(apply_name_presets=False).convert(path).glb)
        assert plain.metallicFactor == 0.0

    def test_translucent_material_is_blended_but_never_invisible(self, tmp_path, pxr):
        Gf = pxr.Gf
        stage, path = _stage(tmp_path, pxr)
        box = _author_box(stage, pxr)
        _bind_mdl_shader(
            stage, pxr, box, "Glass", "OmniGlass.mdl",
            diffuse_color_constant=("Color3f", Gf.Vec3f(1, 1, 1)),
            enable_opacity=("Bool", True),
            opacity_constant=("Float", 0.0),
        )  # fmt: skip
        stage.Save()

        mat = _material_of(USDSceneToGLB().convert(path).glb)
        assert mat.alphaMode == "BLEND"
        assert 0 < mat.baseColorFactor[3] < 255


class TestMdlFile:
    MDL = """mdl 1.4;
export material Block(*) = ::OmniPBR::OmniPBR(
    diffuse_texture: texture_2d("./tex/base.png", ::tex::gamma_srgb),
    enable_ORM_texture: true,
    ORM_texture: texture_2d("./tex/orm.png", ::tex::gamma_linear)
);
"""

    def test_textures_come_from_the_mdl_file(self, tmp_path, pxr):
        stage, path = _stage(tmp_path, pxr)
        box = _author_box(stage, pxr)
        (tmp_path / "Block.mdl").write_text(self.MDL)
        _write_png(tmp_path / "tex" / "base.png")
        _write_png(tmp_path / "tex" / "orm.png")
        _bind_mdl_shader(stage, pxr, box, "Block", "./Block.mdl")
        stage.Save()

        result = USDSceneToGLB().convert(path)
        mat = _material_of(result.glb)
        assert result.report.materials == {"Block": "mdl-file"}
        assert mat.baseColorTexture is not None
        # ORM is glTF's own packing: one image feeds both slots.
        assert mat.metallicRoughnessTexture is not None
        assert mat.occlusionTexture is not None

    def test_an_mdl_outside_the_input_directory_is_not_read(self, tmp_path, pxr):
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "Block.mdl").write_text(self.MDL)
        scene_dir = tmp_path / "scene"
        scene_dir.mkdir()
        stage, path = _stage(scene_dir, pxr)
        box = _author_box(stage, pxr)
        _bind_mdl_shader(stage, pxr, box, "Block", str(outside / "Block.mdl"))
        stage.Save()

        result = USDSceneToGLB().convert(path)
        assert result.report.substituted_materials == ["Block"]


class TestMetalRoughPacking:
    def test_separate_roughness_and_metallic_channels_are_packed(self, tmp_path, pxr):
        from PIL import Image

        UsdShade, Sdf = pxr.UsdShade, pxr.Sdf
        stage, path = _stage(tmp_path, pxr)
        box = _author_box(stage, pxr)
        Image.new("RGB", (8, 8), (0, 64, 0)).save(tmp_path / "rough.png")  # G = 64
        Image.new("RGB", (8, 8), (200, 0, 0)).save(tmp_path / "metal.png")  # R = 200

        mat = UsdShade.Material.Define(stage, "/World/Looks/Mat")
        pbr = UsdShade.Shader.Define(stage, "/World/Looks/Mat/PBR")
        pbr.CreateIdAttr("UsdPreviewSurface")
        for slot, file, channel in (("roughness", "rough.png", "g"),
                                    ("metallic", "metal.png", "r")):  # fmt: skip
            tex = UsdShade.Shader.Define(stage, f"/World/Looks/Mat/{slot}Tex")
            tex.CreateIdAttr("UsdUVTexture")
            tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(file))
            tex.CreateOutput(channel, Sdf.ValueTypeNames.Float)
            pbr.CreateInput(slot, Sdf.ValueTypeNames.Float).ConnectToSource(
                tex.ConnectableAPI(), channel
            )
        pbr.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        mat.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(box.GetPrim()).Bind(mat)
        stage.Save()

        packed = _material_of(
            USDSceneToGLB().convert(path).glb
        ).metallicRoughnessTexture
        r, g, b = packed.convert("RGB").getpixel((0, 0))
        assert (g, b) == (64, 200)  # glTF: roughness in G, metallic in B


class TestMaterials:
    def test_preview_surface_texture_is_embedded_and_downscaled(self, tmp_path, pxr):
        stage, path = _stage(tmp_path, pxr)
        box = _author_box(stage, pxr)
        _write_png(tmp_path / "textures" / "red.png", size=256)
        _bind_preview_surface(
            stage, pxr, box, "textures/red.png", roughness=0.3, metallic=1.0
        )
        stage.Save()

        result = USDSceneToGLB(max_texture_size=64).convert(path)
        mat = next(iter(_load(result.glb).geometry.values())).visual.material
        assert result.report.materials == {"Mat": "UsdPreviewSurface"}
        assert mat.baseColorTexture is not None
        assert max(mat.baseColorTexture.size) == 64
        assert mat.roughnessFactor == pytest.approx(0.3, abs=0.01)
        assert mat.metallicFactor == pytest.approx(1.0, abs=0.01)

    def test_texture_outside_the_input_directory_is_not_embedded(self, tmp_path, pxr):
        scene_dir = tmp_path / "scene"
        scene_dir.mkdir()
        secret = tmp_path / "private" / "secret.png"
        _write_png(secret)
        stage, path = _stage(scene_dir, pxr)
        box = _author_box(stage, pxr)
        _bind_preview_surface(stage, pxr, box, str(secret))  # absolute, elsewhere
        stage.Save()

        result = USDSceneToGLB().convert(path)
        mat = next(iter(_load(result.glb).geometry.values())).visual.material
        assert mat.baseColorTexture is None
        assert str(secret) in result.report.missing_textures

    def test_texture_found_only_by_file_name_is_used_and_reported(self, tmp_path, pxr):
        stage, path = _stage(tmp_path, pxr)
        box = _author_box(stage, pxr)
        _write_png(tmp_path / "textures" / "red.png")
        _bind_preview_surface(stage, pxr, box, "somewhere/else/red.png")
        stage.Save()

        result = USDSceneToGLB().convert(path)
        assert _material_of(result.glb).baseColorTexture is not None
        assert any("matched by file name" in w for w in result.report.warnings)

    def test_an_ambiguous_file_name_match_says_so(self, tmp_path, pxr):
        stage, path = _stage(tmp_path, pxr)
        box = _author_box(stage, pxr)
        _write_png(tmp_path / "a" / "red.png")
        _write_png(tmp_path / "b" / "red.png")
        _bind_preview_surface(stage, pxr, box, "nowhere/red.png")
        stage.Save()

        warnings = USDSceneToGLB().convert(path).report.warnings
        assert any("first of 2 candidates" in w for w in warnings)

    def test_an_exact_path_never_triggers_the_name_fallback(self, tmp_path, pxr):
        stage, path = _stage(tmp_path, pxr)
        box = _author_box(stage, pxr)
        _write_png(tmp_path / "textures" / "red.png")
        _write_png(tmp_path / "other" / "red.png")
        _bind_preview_surface(stage, pxr, box, "textures/red.png")
        stage.Save()
        assert USDSceneToGLB().convert(path).report.warnings == []

    def test_unavailable_mdl_library_is_substituted_and_reported(self, tmp_path, pxr):
        UsdShade, Sdf = pxr.UsdShade, pxr.Sdf
        stage, path = _stage(tmp_path, pxr)
        box = _author_box(stage, pxr)
        mat = UsdShade.Material.Define(stage, "/World/Looks/Carpet")
        sh = UsdShade.Shader.Define(stage, "/World/Looks/Carpet/Shader")
        sh.CreateImplementationSourceAttr("sourceAsset")
        sh.GetPrim().CreateAttribute(
            "info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset
        ).Set(Sdf.AssetPath("../../Library/Carpet.mdl"))
        sh.CreateOutput("out", Sdf.ValueTypeNames.Token)
        mat.CreateOutput("mdl:surface", Sdf.ValueTypeNames.Token).ConnectToSource(
            sh.ConnectableAPI(), "out"
        )
        UsdShade.MaterialBindingAPI.Apply(box.GetPrim()).Bind(mat)
        stage.Save()

        result = USDSceneToGLB().convert(path)
        assert result.report.substituted_materials == ["Carpet"]
        assert any("Carpet.mdl" in w for w in result.report.warnings)

        # ...and an override replaces the placeholder.
        fixed = USDSceneToGLB(
            material_overrides={"Carpet": {"base_color": [0.3, 0.3, 0.2]}}
        ).convert(path)
        assert fixed.report.substituted_materials == []
        assert fixed.report.materials == {"Carpet": "override"}

    def test_a_broken_material_does_not_sink_the_scene(self, tmp_path, pxr):
        stage, path = _stage(tmp_path, pxr)
        box = _author_box(stage, pxr)
        broken = tmp_path / "broken.png"
        broken.write_bytes(b"not an image")
        _bind_preview_surface(stage, pxr, box, "broken.png")
        stage.Save()
        result = USDSceneToGLB().convert(path)
        assert result.report.mesh_count == 1
        assert "broken.png" in result.report.missing_textures


class TestComposition:
    def _stage_with_outside_layer(self, tmp_path, pxr):
        outside = tmp_path / "outside"
        outside.mkdir()
        layer = pxr.Usd.Stage.CreateNew(str(outside / "other.usda"))
        pxr.UsdGeom.Xform.Define(layer, "/World")
        _author_box(layer, pxr, "/World/Leaked")
        layer.Save()

        scene_dir = tmp_path / "scene"
        scene_dir.mkdir()
        stage, path = _stage(scene_dir, pxr)
        stage.GetRootLayer().subLayerPaths.append(str(outside / "other.usda"))
        stage.Save()
        return path

    def test_layers_outside_the_input_directory_are_rejected(self, tmp_path, pxr):
        path = self._stage_with_outside_layer(tmp_path, pxr)
        with pytest.raises(ValueError, match="outside the permitted directory"):
            USDSceneToGLB().convert(path)

    def test_they_can_be_allowed_for_trusted_input(self, tmp_path, pxr):
        path = self._stage_with_outside_layer(tmp_path, pxr)
        result = USDSceneToGLB(allow_external_layers=True).convert(path)
        assert result.report.mesh_count == 1

    def test_textures_inside_a_usdz_package_are_read(self, tmp_path, pxr):
        """A USDZ is a ZIP: its textures are not files on disk, so they are read
        back through USD's resolver. They used to be reported as missing."""
        from pxr import Sdf, UsdUtils

        build = tmp_path / "build"
        build.mkdir()
        stage, scene = _stage(build, pxr)
        box = _author_box(stage, pxr)
        _write_png(build / "red.png")
        _bind_preview_surface(stage, pxr, box, "red.png")
        stage.Save()
        package_dir = tmp_path / "package"
        package_dir.mkdir()
        package = package_dir / "scene.usdz"
        assert UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(str(scene)), str(package))

        result = USDSceneToGLB().convert(package)  # only the package is on disk
        assert result.report.missing_textures == []
        assert _material_of(result.glb).baseColorTexture is not None

    def test_non_usd_input_is_rejected(self, tmp_path, pxr):
        bogus = tmp_path / "scene.txt"
        bogus.write_text("hello")
        with pytest.raises(ValueError, match="not a USD file"):
            USDSceneToGLB().convert(bogus)

    def test_root_dir_allows_layers_in_a_sibling_folder_of_the_scene(
        self, tmp_path, pxr
    ):
        """An extracted archive: ``models/scene.usda`` sublayering ``../shared``."""
        shared = tmp_path / "shared"
        shared.mkdir()
        layer = pxr.Usd.Stage.CreateNew(str(shared / "props.usda"))
        pxr.UsdGeom.Xform.Define(layer, "/World")
        _author_box(layer, pxr, "/World/Prop")
        layer.Save()

        models = tmp_path / "models"
        models.mkdir()
        stage, path = _stage(models, pxr)
        stage.GetRootLayer().subLayerPaths.append("../shared/props.usda")
        stage.Save()

        # Confined to the scene file's own folder, the sibling is out of bounds...
        with pytest.raises(ValueError, match="outside the permitted directory"):
            USDSceneToGLB().convert(path)
        # ...but fine once the archive root is the boundary.
        result = USDSceneToGLB().convert(path, root_dir=tmp_path)
        assert result.report.mesh_count == 1

    def test_the_scene_file_must_be_inside_root_dir(self, tmp_path, pxr):
        stage, path = _stage(tmp_path, pxr)
        _author_box(stage, pxr)
        stage.Save()
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        with pytest.raises(ValueError, match="not inside the permitted directory"):
            USDSceneToGLB().convert(path, root_dir=elsewhere)


def test_constructor_validates_texture_size():
    with pytest.raises(ValueError, match="max_texture_size"):
        USDSceneToGLB(max_texture_size=0)
