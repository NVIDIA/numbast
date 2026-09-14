# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from numbast.name_policy import (
    apply_prefix_removal,
    is_identifier,
    rust_identifier,
    rust_parameter_name,
)


def test_apply_prefix_removal_uses_first_matching_prefix():
    name = "library_detail_function"

    assert apply_prefix_removal(name, ["library_", "library_detail_"]) == (
        "detail_function"
    )
    assert apply_prefix_removal(name, ["library_detail_", "library_"]) == (
        "function"
    )


def test_apply_prefix_removal_preserves_nonmatching_name():
    assert apply_prefix_removal("function", ["library_"]) == "function"


def test_apply_prefix_removal_accepts_empty_prefix_list():
    assert apply_prefix_removal("function", []) == "function"


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
