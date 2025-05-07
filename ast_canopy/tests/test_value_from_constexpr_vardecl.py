import os

import pytest

from ast_canopy import value_from_constexpr_vardecl


@pytest.fixture(scope="module")
def sample_value_from_constexpr_vardecl():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(
        current_directory, "data/", "sample_constexpr_vardecl.cu"
    )


@pytest.fixture(scope="module")
def source(sample_value_from_constexpr_vardecl):
    with open(sample_value_from_constexpr_vardecl) as f:
        src = f.read()
    return src


@pytest.mark.parametrize(
    "var_name, expected",
    [
        ("i4", 42),
        ("u4", 42),
        ("i8", 42),
        ("u8", 42),
        ("i8b", 42),
        ("u8b", 42),
        ("i2", 42),
        ("i4b", 42),
        ("i8c", 42),
        ("u2", 42),
        ("u4b", 42),
        ("u8c", 42),
        ("f4", 3.14),
        ("f8", 3.14),
    ],
)
def test_value_from_constexpr_vardecl(source, var_name, expected):
    # Small code piece, cost ~400ms on threadripper PRO 7975WX
    # If without parsing cstdint, can save another 150ms
    result = value_from_constexpr_vardecl(source, var_name, "sm_70")
    assert result.value == expected
