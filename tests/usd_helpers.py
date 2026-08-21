# Copyright [2025] Cyberwave
# Licensed under the Apache License, Version 2.0

"""Comparison helper for the USD round-trip tests."""

from __future__ import annotations

import math
from typing import Any

from cyberwave_robot_format.schema import CommonSchema


def assert_schema_equal(
    first: CommonSchema,
    second: CommonSchema,
    *,
    rel_tol: float = 1e-9,
    abs_tol: float = 1e-12,
) -> None:
    """Assert two schemas are equal, reporting the path of the first mismatch.

    The tolerance is deliberately tight. Almost every native ``UsdPhysics``
    attribute is ``float32``, but the exporter pairs each lossy one with a
    double-precision ``cyberwave:`` sidecar that the parser reads preferentially —
    so a genuine round trip is essentially exact, and a loose tolerance would hide
    a sidecar that stopped being written.

    Note:
        ``CommonSchema.to_dict()`` converts tuples to lists, so this helper cannot
        catch a tuple-vs-list regression in ``Actuator.control_range`` /
        ``force_range``. Those need a direct ``isinstance`` check on the parsed
        object — see ``test_actuator_ranges_stay_tuples``.
    """
    mismatches = _compare(first.to_dict(), second.to_dict(), "$", rel_tol, abs_tol)
    if mismatches:
        detail = "\n".join(f"  {line}" for line in mismatches[:20])
        suffix = (
            f"\n  ... and {len(mismatches) - 20} more" if len(mismatches) > 20 else ""
        )
        raise AssertionError(
            f"{len(mismatches)} schema mismatch(es) after round trip:\n{detail}{suffix}"
        )


def _compare(
    expected: Any, actual: Any, path: str, rel_tol: float, abs_tol: float
) -> list[str]:
    if isinstance(expected, dict) and isinstance(actual, dict):
        out: list[str] = []
        for key in sorted(set(expected) | set(actual)):
            if key not in expected:
                out.append(f"{path}.{key}: unexpected key ({actual[key]!r})")
            elif key not in actual:
                out.append(f"{path}.{key}: missing key ({expected[key]!r})")
            else:
                out += _compare(
                    expected[key], actual[key], f"{path}.{key}", rel_tol, abs_tol
                )
        return out

    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [f"{path}: length {len(expected)} != {len(actual)}"]
        out = []
        for index, (left, right) in enumerate(zip(expected, actual)):
            out += _compare(left, right, f"{path}[{index}]", rel_tol, abs_tol)
        return out

    # bool before int/float: bool is a subclass of int, so True would otherwise
    # compare equal to 1 and silently pass.
    if isinstance(expected, bool) or isinstance(actual, bool):
        if expected is not actual:
            return [f"{path}: {expected!r} != {actual!r}"]
        return []

    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if not math.isclose(
            float(expected), float(actual), rel_tol=rel_tol, abs_tol=abs_tol
        ):
            return [f"{path}: {expected!r} != {actual!r}"]
        return []

    if type(expected) is not type(actual):
        return [
            (
                f"{path}: type {type(expected).__name__} != "
                f"{type(actual).__name__} ({expected!r} vs {actual!r})"
            )
        ]

    if expected != actual:
        return [f"{path}: {expected!r} != {actual!r}"]
    return []


__all__ = ["assert_schema_equal"]
