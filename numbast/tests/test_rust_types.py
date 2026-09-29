# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from ast_canopy import parse_declarations_from_source

from numbast.cuda_oxide_binding_model import CudaOxideBindingPlan
from numbast.rust_types import (
    CudaOxideArray,
    CudaOxidePointer,
    CudaOxideType,
    cuda_abi_alias_for_arch,
    is_identifier,
    parse_cuda_oxide_type,
    rust_identifier,
    rust_parameter_name,
    rust_struct_storage,
)


@pytest.mark.parametrize("name", ["function", "_function", "function_2"])
def test_c_and_rust_identifiers_are_accepted(name):
    assert is_identifier(name)
    assert rust_identifier(name) == name


@pytest.mark.parametrize("name", ["", "2function", "not-an-identifier"])
def test_invalid_identifiers_are_rejected(name):
    assert not is_identifier(name)
    with pytest.raises(ValueError, match="Not a valid C/Rust identifier"):
        rust_identifier(name)


def test_rust_keywords_use_raw_identifiers_when_possible():
    assert rust_identifier("match") == "r#match"
    assert rust_identifier("union") == "r#union"
    assert rust_identifier("self") == "self_"


def test_rust_parameter_names_are_sanitized():
    assert rust_parameter_name("", 3) == "arg3"
    assert rust_parameter_name("type", 0) == "r#type"
    assert rust_parameter_name("9bad-name", 0) == "arg_9bad_name"


def test_cuda_oxide_type_preserves_pointer_and_array_order():
    plan = CudaOxideBindingPlan()
    pointers = CudaOxideType(
        c_spelling="const int *const *volatile",
        base_name="int",
        layers=(CudaOxidePointer("const"), CudaOxidePointer("const")),
    )
    assert plan.render_rust_type(pointers) == "*const *const i32"

    array = CudaOxideType(
        c_spelling="unsigned int[2][3]",
        base_name="unsigned int",
        layers=(CudaOxideArray(2), CudaOxideArray(3)),
    )
    assert plan.render_rust_type(array) == "[[u32; 3]; 2]"

    pointer_to_array = CudaOxideType(
        c_spelling="const int (*)[2][3]",
        base_name="int",
        layers=(
            CudaOxidePointer("const"),
            CudaOxideArray(2),
            CudaOxideArray(3),
        ),
    )
    assert plan.render_rust_type(pointer_to_array) == ("*const [[i32; 3]; 2]")

    array_of_pointers = CudaOxideType(
        c_spelling="const int *[2][3]",
        base_name="int",
        layers=(
            CudaOxideArray(2),
            CudaOxideArray(3),
            CudaOxidePointer("const"),
        ),
    )
    assert plan.render_rust_type(array_of_pointers) == "[[*const i32; 3]; 2]"


def test_cuda_oxide_type_equality_ignores_source_spelling():
    spaced = CudaOxideType(
        c_spelling="int *",
        base_name="int",
        layers=(CudaOxidePointer("mut"),),
    )
    compact = CudaOxideType(
        c_spelling="int*",
        base_name="int",
        layers=(CudaOxidePointer("mut"),),
    )
    const_pointer = CudaOxideType(
        c_spelling="const int *",
        base_name="int",
        layers=(CudaOxidePointer("const"),),
    )

    assert spaced == compact
    assert hash(spaced) == hash(compact)
    assert spaced != const_pointer


def test_cuda_oxide_type_uses_ast_canopy_structure():
    path = Path(__file__).parent / "data" / "sample_cuda_oxide_types.cuh"
    declarations = parse_declarations_from_source(
        str(path), [str(path)], "sm_80", cuda_header_mode=True
    )
    record = next(
        item for item in declarations.structs if item.name == "CudaOxideTypes"
    )
    fields = {field.name: field.type_ for field in record.fields}
    plan = CudaOxideBindingPlan()

    assert (
        plan.render_rust_type(
            parse_cuda_oxide_type(fields["array_of_pointers"])
        )
        == "[*mut i32; 4]"
    )
    assert (
        plan.render_rust_type(parse_cuda_oxide_type(fields["pointer_to_array"]))
        == "*mut [i32; 4]"
    )
    assert (
        plan.render_rust_type(parse_cuda_oxide_type(fields["pointer_to_const"]))
        == "*const i32"
    )
    assert (
        plan.render_rust_type(parse_cuda_oxide_type(fields["const_pointer"]))
        == "*mut i32"
    )


def test_cuda_aliases_are_architecture_sensitive():
    assert cuda_abi_alias_for_arch("__half", "sm_90") == ("u16", 2, 2)
    assert cuda_abi_alias_for_arch("__half", "sm_90a") == ("u16", 2, 2)
    assert cuda_abi_alias_for_arch("__half", "sm_100") == ("f16", 2, 2)
    assert cuda_abi_alias_for_arch("double2", "sm_100") == (
        "[u128; 1]",
        16,
        16,
    )


def test_opaque_struct_storage_preserves_size_and_alignment():
    assert rust_struct_storage(16, 8) == "[u64; 2]"
    with pytest.raises(ValueError, match="cannot represent"):
        rust_struct_storage(12, 8)
