# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pathlib

import pytest

from ast_canopy import parse_declarations_from_source


@pytest.fixture(scope="module")
def vendored_cuda_h():
    current_directory = pathlib.Path(__file__).resolve().parent
    cuda_h = current_directory / "cuda_headers" / "include" / "cuda.h"
    if not cuda_h.exists():
        pytest.skip(f"Vendored cuda.h not found at {cuda_h}")
    return cuda_h


@pytest.fixture(scope="module")
def cuda_h_decls(vendored_cuda_h):
    source_file = str(vendored_cuda_h)
    return source_file, parse_declarations_from_source(
        source_file,
        [source_file],
        "sm_80",
        cuda_header_mode=True,
    )


def test_parse_vendored_cuda_h_functions(cuda_h_decls):
    source_file, decls = cuda_h_decls
    functions_by_name = {func.name: func for func in decls.functions}

    assert functions_by_name
    assert "cuInit" in functions_by_name
    assert functions_by_name["cuInit"].parse_entry_point == source_file


def test_parse_vendored_cuda_h_structs(cuda_h_decls):
    source_file, decls = cuda_h_decls

    structs_by_name = {struct.name: struct for struct in decls.structs}
    expected_struct_names = {
        "CUuuid_st",
        "CUipcEventHandle_st",
        "CUipcMemHandle_st",
        "CUDA_MEMCPY3D_st",
    }

    assert structs_by_name
    assert expected_struct_names.issubset(structs_by_name)
    for struct_name in expected_struct_names:
        assert structs_by_name[struct_name].parse_entry_point == source_file


def test_parse_vendored_cuda_h_enums(cuda_h_decls):
    _, decls = cuda_h_decls

    enum_names = {enum.name for enum in decls.enums}
    expected_enum_names = {
        "CUctx_flags_enum",
        "CUdevice_attribute_enum",
        "CUstream_flags_enum",
        "CUevent_flags_enum",
    }

    assert enum_names
    assert expected_enum_names.issubset(enum_names)
