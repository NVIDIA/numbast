# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from ast_canopy import parse_declarations_from_source


@pytest.fixture(scope="module")
def sample_inline_hint_macro(data_folder):
    return data_folder / "inline_hint_macro_fix.cu"


def test_inline_hint_macro(sample_inline_hint_macro):
    srcstr = str(sample_inline_hint_macro)

    decls = parse_declarations_from_source(srcstr, [srcstr], "sm_80")
    functions = {func.name: func for func in decls.functions}

    assert functions["inline_hint_void"].return_type.name == "void"
    assert functions["inline_hint_int"].return_type.name == "int"
