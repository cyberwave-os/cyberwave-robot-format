import math

import pytest

from cyberwave_robot_format.fov_convention import (
    find_fov_value,
    normalize_fov_key,
)


@pytest.mark.parametrize(
    "name,value,expected_key,expected_rad",
    [
        ("hfov", 1.5707963267948966, "hfov_rad", 1.5707963267948966),
        ("horizontal_fov", 1.5707963267948966, "hfov_rad", 1.5707963267948966),
        ("vfov", 60.0, "vfov_rad", math.radians(60.0)),
        ("fovy", 90.0, "vfov_rad", math.radians(90.0)),
        ("fov_degrees", 60.0, "vfov_rad", math.radians(60.0)),
        ("fov_vertical", 60.0, "vfov_rad", math.radians(60.0)),
        ("vertical_fov", 60.0, "vfov_rad", math.radians(60.0)),
        ("fov_horizontal", 90.0, "hfov_rad", math.radians(90.0)),
        ("horizontal_fov_degrees", 90.0, "hfov_rad", math.radians(90.0)),
        ("hfov_rad", 1.2, "hfov_rad", 1.2),
        ("vfov_rad", 1.2, "vfov_rad", 1.2),
    ],
)
def test_normalize_fov_key_known_spellings(name, value, expected_key, expected_rad):
    result = normalize_fov_key(name, value)
    assert result is not None
    key, rad = result
    assert key == expected_key
    assert rad == pytest.approx(expected_rad)


def test_normalize_fov_key_vfov_is_degrees_matching_the_backend_resolver():
    # unlike hfov/horizontal_fov, vfov is degrees -- backwards here is a silent ~57x corruption
    key, rad = normalize_fov_key("vfov", 60.0)
    assert key == "vfov_rad"
    assert rad == pytest.approx(math.radians(60.0))
    assert rad != pytest.approx(60.0)


def test_normalize_fov_key_is_separator_insensitive():
    result = normalize_fov_key("Horizontal-FOV", 1.0)
    assert result == ("hfov_rad", 1.0)


def test_normalize_fov_key_is_insensitive_to_no_separator_at_all():
    result = normalize_fov_key("HorizontalFOV", 1.0)
    assert result == ("hfov_rad", 1.0)
    assert normalize_fov_key("horizontalfov", 1.0) == ("hfov_rad", 1.0)


def test_normalize_fov_key_never_infers_unit_from_magnitude():
    key, rad = normalize_fov_key("hfov", 0.5)
    assert key == "hfov_rad"
    assert rad == pytest.approx(0.5)


def test_normalize_fov_key_returns_none_for_unrecognized_name():
    assert normalize_fov_key("totally_unknown_spelling", 42.0) is None


def test_find_fov_value_picks_first_present_candidate():
    params = {"fovy": 90.0, "vfov": 45.0}
    result = find_fov_value(params, ("hfov", "fovy", "vfov"))
    assert result == ("vfov_rad", math.radians(90.0))


def test_find_fov_value_returns_none_when_nothing_present():
    assert find_fov_value({"width": 640}, ("hfov", "fovy")) is None
