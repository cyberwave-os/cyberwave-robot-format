# Copyright [2025] Cyberwave
# Licensed under the Apache License, Version 2.0

"""The deferred-``pxr``-import contract.

``usd-core`` is an optional extra — it publishes no ``linux-aarch64`` wheel, and
the rest of the package has to stay installable and usable without it. So nothing
under ``cyberwave_robot_format/usd/`` may import ``pxr`` at module scope: importing
the package, importing the subpackage, and *constructing* the converters must all
succeed, and only *calling* them may fail — with a message that says how to fix it.

These tests deliberately do **not** use ``importorskip``: they have to run in both
environments, and they are the only coverage of the with-``pxr``-missing path on a
machine that has it installed.
"""

from __future__ import annotations

import builtins
import importlib
from pathlib import Path

import pytest

from cyberwave_robot_format.schema import CommonSchema, Link, Metadata, Vector3
from cyberwave_robot_format.usd import common


@pytest.fixture
def without_pxr(monkeypatch):
    """Make ``import pxr`` fail, with the module-level cache reset around it.

    ``require_pxr()`` memoizes the imported modules in ``common._PXR``. Without
    resetting it the guard would pass spuriously on a machine that *has*
    ``usd-core`` (the cache short-circuits the patched import), and leaving a
    poisoned cache behind would break every later test in the session — so the
    reset happens on the way in and monkeypatch undoes it on the way out.
    """
    monkeypatch.setattr(common, "_PXR", None)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pxr" or name.startswith("pxr."):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    yield
    monkeypatch.setattr(common, "_PXR", None)


def simple_schema() -> CommonSchema:
    return CommonSchema(metadata=Metadata(name="robot"), links=[Link(name="a")])


class TestImportsSucceedWithoutPxr:
    def test_the_package_imports_and_exposes_the_converters(self, without_pxr):
        module = importlib.reload(importlib.import_module("cyberwave_robot_format"))
        assert hasattr(module, "USDExporter")
        assert hasattr(module, "USDParser")

    def test_the_usd_subpackage_imports(self, without_pxr):
        module = importlib.reload(importlib.import_module("cyberwave_robot_format.usd"))
        assert module.USDExporter is not None
        assert module.USDParser is not None

    @pytest.mark.parametrize(
        "module_name",
        [
            "cyberwave_robot_format.usd.tokens",
            "cyberwave_robot_format.usd.frames",
            "cyberwave_robot_format.usd.common",
            "cyberwave_robot_format.usd.exporter",
            "cyberwave_robot_format.usd.parser",
        ],
    )
    def test_each_usd_module_imports(self, without_pxr, module_name):
        assert importlib.reload(importlib.import_module(module_name)) is not None

    def test_constructing_the_converters_succeeds(self, without_pxr):
        from cyberwave_robot_format.usd import USDExporter, USDParser

        assert USDExporter(bake_meshes=True) is not None
        assert USDParser() is not None

    def test_can_parse_works_without_pxr(self, without_pxr, tmp_path):
        # Purely a suffix/magic-byte check, so it must not need USD.
        from cyberwave_robot_format.usd import USDParser

        path = tmp_path / "r.usda"
        path.write_text("#usda 1.0\n")
        assert USDParser().can_parse(path)

    def test_format_converter_constructs(self, without_pxr):
        from cyberwave_robot_format.core import FormatConverter

        converter = FormatConverter()
        # USD still registers — nothing under usd/ imports pxr at module scope — so
        # a caller gets the actionable ImportError instead of a missing format.
        assert "usd" in converter.engine.parsers
        assert "urdf" in converter.engine.parsers


class TestCallsFailWithAnActionableMessage:
    def test_export_raises_import_error_naming_usd_core(self, without_pxr):
        from cyberwave_robot_format.usd import USDExporter

        with pytest.raises(ImportError, match="usd-core"):
            USDExporter().export_to_string(simple_schema())

    def test_export_to_file_raises_import_error(self, without_pxr, tmp_path):
        from cyberwave_robot_format.usd import USDExporter

        with pytest.raises(ImportError, match="usd-core"):
            USDExporter().export(simple_schema(), tmp_path / "r.usda")

    def test_parse_string_raises_import_error_naming_usd_core(self, without_pxr):
        from cyberwave_robot_format.usd import USDParser

        with pytest.raises(ImportError, match="usd-core"):
            USDParser().parse_string("#usda 1.0\n")

    def test_parse_file_raises_import_error(self, without_pxr, tmp_path):
        from cyberwave_robot_format.usd import USDParser

        path = tmp_path / "r.usda"
        path.write_text("#usda 1.0\n")
        with pytest.raises(ImportError, match="usd-core"):
            USDParser().parse(path)

    def test_the_message_names_the_extra(self, without_pxr):
        from cyberwave_robot_format.usd import USDExporter

        with pytest.raises(ImportError) as excinfo:
            USDExporter().export_to_string(simple_schema())
        message = str(excinfo.value)
        assert "cyberwave-robot-format[usd]" in message
        assert "pxr" in message


class TestPxrFreeModules:
    """``frames`` and ``tokens`` must not touch ``pxr`` even lazily.

    Load-bearing: ``test_usd_frames.py`` and ``test_usd_tokens.py``'s pure-table
    assertions run on the strength of this, which is what keeps the joint-frame and
    inertia math covered on a machine with no ``usd-core``.
    """

    @pytest.mark.parametrize("module_name", ["frames", "tokens"])
    def test_module_never_imports_pxr(self, module_name):
        module = importlib.import_module(f"cyberwave_robot_format.usd.{module_name}")
        source = Path(module.__file__).read_text()
        # Prose mentions of pxr in docstrings are fine; import statements are not.
        offenders = [
            line.strip()
            for line in source.splitlines()
            if line.strip().startswith(("import pxr", "from pxr"))
            or "require_pxr" in line
        ]
        assert offenders == []

    def test_frames_math_works_with_pxr_unavailable(self, without_pxr):
        module = importlib.reload(
            importlib.import_module("cyberwave_robot_format.usd.frames")
        )
        quat = module.axis_to_ref_quat(Vector3(y=1.0))
        assert module.ref_quat_to_axis(quat).y == pytest.approx(1.0)


class TestUsdCoreIsActuallyInstalled:
    """Turn a silently-skipped USD suite into a loud failure.

    Every other USD test module opens with ``pytest.importorskip("pxr")``, so a
    missing ``usd-core`` does not fail the run — it removes roughly 500 tests from
    collection and the suite still reports success. That is exactly the shape of a
    CI job that looks green while testing none of the code under review.

    ``usd-core`` publishes **no sdist**, and its Linux wheels are x86_64-only, so
    skipping is legitimate on linux/aarch64 (the ``dev`` extra's environment marker
    excludes it there) and nowhere else. Anywhere a wheel exists, a missing ``pxr``
    means the extra was not installed and the run is not testing what it appears to.
    """

    @staticmethod
    def _wheel_exists_for_this_platform() -> bool:
        import platform
        import sys

        if sys.platform != "linux":
            # macOS ships universal2 wheels, Windows ships win_amd64.
            return True
        return platform.machine() in {"x86_64", "AMD64", "amd64"}

    def test_pxr_is_importable_wherever_usd_core_has_a_wheel(self):
        import platform
        import sys

        if not self._wheel_exists_for_this_platform():
            pytest.skip(
                f"usd-core publishes no wheel for {sys.platform}/{platform.machine()}; "
                "the USD suite is legitimately skipped here."
            )
        try:
            importlib.import_module("pxr")
        except ImportError:  # pragma: no cover - only on a misconfigured environment
            pytest.fail(
                f"usd-core is not installed on {sys.platform}/{platform.machine()}, "
                "where a wheel does exist. Every USD test module would silently skip. "
                'Install the dev extra: pip install -e ".[dev]"'
            )


class TestCacheIsRestored:
    def test_pxr_still_works_after_the_guard_fixture(self):
        # Guards against the fixture poisoning later tests in the same session.
        pytest.importorskip("pxr")
        from cyberwave_robot_format.usd import USDExporter

        assert "#usda" in USDExporter().export_to_string(simple_schema())
