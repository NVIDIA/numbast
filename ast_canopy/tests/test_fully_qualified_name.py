# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from ast_canopy import parse_declarations_from_source


@pytest.fixture(scope="module")
def sample_qualified_names_source(data_folder):
    return data_folder / "sample_qualified_names.cu"


def test_fully_qualified_names(sample_qualified_names_source):
    srcstr = str(sample_qualified_names_source)
    decls = parse_declarations_from_source(srcstr, [srcstr], "sm_80")

    # Test global function
    global_func = next(f for f in decls.functions if f.name == "global_func")
    assert global_func.fully_qualified_name == "global_func"

    # Test function in outer namespace
    outer_func = next(f for f in decls.functions if f.name == "outer_func")
    assert outer_func.fully_qualified_name == "outer_outer_func"

    # Test function in nested namespace
    inner_func = next(f for f in decls.functions if f.name == "inner_func")
    assert inner_func.fully_qualified_name == "outer_inner_inner_func"

    # Test struct method in global namespace
    global_struct = next(s for s in decls.structs if s.name == "GlobalStruct")
    global_method = next(
        m for m in global_struct.methods if m.name == "struct_method"
    )
    assert global_method.fully_qualified_name == "GlobalStruct_struct_method"

    # Test struct method in nested namespace
    nested_struct = next(s for s in decls.structs if s.name == "NestedStruct")
    nested_method = next(
        m for m in nested_struct.methods if m.name == "struct_method"
    )
    assert (
        nested_method.fully_qualified_name
        == "outer_inner_NestedStruct_struct_method"
    )
