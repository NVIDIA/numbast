# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pickle
from pathlib import Path

import pytest

from ast_canopy import parse_declarations_from_source
from ast_canopy import pylibastcanopy as bindings


@pytest.fixture(scope="module")
def declarations():
    source = str(
        Path(__file__).with_name("data") / "sample_non_record_typedefs.cu"
    )
    return parse_declarations_from_source(source, [source], "sm_80")


@pytest.mark.parametrize("roundtrip", [False, True])
def test_non_record_typedefs_are_retained(declarations, roundtrip):
    typedefs = declarations.typedefs
    if roundtrip:
        typedefs = pickle.loads(pickle.dumps(typedefs))
    by_name = {item.name: item for item in typedefs}
    assert set(by_name) == {
        "team_t",
        "team_alias_t",
        "region_handle_t",
        "const_int_t",
        "int_ptr_t",
        "pointer_alias_t",
        "int_array_t",
        "callback_t",
        "status_t",
        "record_t",
        "record_alias_t",
        "storage_t",
    }
    assert by_name["team_t"].underlying_name == "int32_t"
    assert by_name["team_alias_t"].underlying_name == "team_t"
    assert by_name["region_handle_t"].underlying_name == "uint64_t"
    assert by_name["const_int_t"].underlying_type.is_const_qualified()
    assert (
        by_name["int_ptr_t"].underlying_type.kind == bindings.type_kind.pointer
    )
    assert (
        by_name["int_array_t"].underlying_type.kind
        == bindings.type_kind.constant_array
    )
    assert by_name["int_array_t"].underlying_type.array_size == 4
    assert (
        by_name["callback_t"].underlying_type.kind == bindings.type_kind.pointer
    )
    enum_type = by_name["status_t"].underlying_type
    while enum_type.kind == bindings.type_kind.sugar:
        enum_type = enum_type.inner_type
    assert enum_type.kind == bindings.type_kind.enum_
    assert by_name["record_t"].underlying_name == "Record"
    assert by_name["record_alias_t"].underlying_name == "Record"


def test_typedefs_from_unretained_headers_are_not_collected(declarations):
    assert not any(item.name == "int32_t" for item in declarations.typedefs)
