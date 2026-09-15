# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from numbast.rust_types import (
    CAbiType,
    cuda_abi_alias_for_arch,
    is_identifier,
    rust_identifier,
    rust_parameter_name,
    rust_record_storage,
    rust_type,
)


def empty_model():
    return SimpleNamespace(cuda_aliases={}, enums=[], records=[], typedefs=[])


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


def test_rust_type_preserves_pointer_and_array_order():
    pointers = CAbiType(
        c_spelling="const int *const *volatile",
        base_name="int",
        pointer_kinds=("const", "const"),
    )
    assert rust_type(pointers, empty_model()) == "*const *const i32"

    array = CAbiType(
        c_spelling="unsigned int[2][3]",
        base_name="unsigned int",
        array_dimensions=(2, 3),
    )
    assert rust_type(array, empty_model()) == "[[u32; 3]; 2]"

    pointer_to_array = CAbiType(
        c_spelling="const int (*)[2][3]",
        base_name="int",
        pointer_kinds=("const",),
        array_dimensions=(2, 3),
    )
    assert rust_type(pointer_to_array, empty_model()) == (
        "*const [[i32; 3]; 2]"
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


def test_opaque_record_storage_preserves_size_and_alignment():
    assert rust_record_storage(16, 8) == "[u64; 2]"
    with pytest.raises(ValueError, match="cannot represent"):
        rust_record_storage(12, 8)
