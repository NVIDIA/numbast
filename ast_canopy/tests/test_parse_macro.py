# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from ast_canopy import parse_declarations_from_source


@pytest.fixture(scope="module")
def sample_macro_source(data_folder):
    return data_folder / "sample_macro.cu"


def test_macro_expansions(sample_macro_source):
    srcstr = str(sample_macro_source)

    # We allow parsing macro expanded functions via an allow list for the prefixes. This is a temporary feature
    # until we have a better solution for macro expansion.
    decls = parse_declarations_from_source(
        srcstr, [srcstr], "sm_80", anon_filename_decl_prefix_allowlist=["forty_two_"]
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
