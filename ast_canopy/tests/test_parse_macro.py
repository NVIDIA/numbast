# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from ast_canopy import parse_declarations_from_source


@pytest.fixture(scope="module")
def sample_macro_source(data_folder):
    return data_folder / "sample_macro.cu"


def test_macro_expansions(sample_macro_source):
    srcstr = str(sample_macro_source)

    decls = parse_declarations_from_source(
        srcstr,
        [srcstr],
        "sm_80",
    )
    functions = decls.functions

    assert len(functions) == 3

    forty_two_int, forty_two_float, forty_two_double = functions

    assert forty_two_int.name == "forty_two_int"
    assert forty_two_float.name == "forty_two_float"
    assert forty_two_double.name == "forty_two_double"

    assert forty_two_int.return_type.name == "int"
    assert forty_two_float.return_type.name == "float"
    assert forty_two_double.return_type.name == "double"


def test_macro_defines(data_folder):
    srcstr = str(data_folder / "sample_macro_defines.cu")

    decls = parse_declarations_from_source(
        srcstr,
        [srcstr],
        "sm_80",
    )

    assert decls.macro_defines["foo"] == "123"
    assert decls.macro_defines["FLAG"] == ""
    assert "MAKE_VALUE" not in decls.macro_defines
    assert "INCLUDED_VALUE" not in decls.macro_defines


def test_macro_defines_retained_include(data_folder):
    srcstr = str(data_folder / "sample_macro_defines.cu")
    include = str(data_folder / "sample_macro_defines_include.cuh")

    decls = parse_declarations_from_source(
        srcstr,
        [srcstr, include],
        "sm_80",
    )

    assert decls.macro_defines["foo"] == "123"
    assert decls.macro_defines["INCLUDED_VALUE"] == "456"
