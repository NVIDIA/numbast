# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from ast_canopy import parse_declarations_from_source


@pytest.fixture(scope="module")
def sample_namespace_source(data_folder):
    return data_folder / "sample_namespace.cu"


@pytest.fixture(scope="module")
def decls(sample_namespace_source):
    srcstr = str(sample_namespace_source)
    return parse_declarations_from_source(
        srcstr, [srcstr], "sm_80", verbose=True
    )


def test_no_namespace_declarations(decls):
    """Test declarations with no namespace."""
    # Test struct
    no_ns_struct = next(
        s for s in decls.structs if s.name == "NoNamespaceStruct"
    )
    assert no_ns_struct.namespace_stack == []
    assert len(no_ns_struct.methods) == 1
    assert no_ns_struct.methods[0].name == "foo"
    assert no_ns_struct.methods[0].namespace_stack == []

    # Test function template
    no_ns_func = next(
        f
        for f in decls.function_templates
        if f.function.name == "no_namespace_func"
    )
    assert no_ns_func.namespace_stack == []


def test_flat_namespace_declarations(decls):
    """Test declarations in a flat (single-level) namespace."""
    # Test struct
    flat_struct = next(s for s in decls.structs if s.name == "FlatStruct")
    assert flat_struct.namespace_stack == ["flat"]
    assert len(flat_struct.methods) == 1
    assert flat_struct.methods[0].name == "bar"
    assert flat_struct.methods[0].namespace_stack == ["flat"]

    # Test function template
    flat_func = next(
        f for f in decls.function_templates if f.function.name == "flat_func"
    )
    assert flat_func.namespace_stack == ["flat"]

    # Test enum
    flat_enum = next(e for e in decls.enums if e.name == "FlatEnum")
    assert flat_enum.namespace_stack == ["flat"]
    assert len(flat_enum.enumerators) == 2
    assert flat_enum.enumerators[0] == "A"
    assert flat_enum.enumerators[1] == "B"


def test_nested_namespace_declarations(decls):
    """Test declarations in nested namespaces."""
    # Test struct
    nested_struct = next(s for s in decls.structs if s.name == "NestedStruct")
    assert nested_struct.namespace_stack == ["inner", "outer"]
    assert len(nested_struct.methods) == 1
    assert nested_struct.methods[0].name == "baz"
    assert nested_struct.methods[0].namespace_stack == ["inner", "outer"]

    # Test function template
    nested_func = next(
        f for f in decls.function_templates if f.function.name == "nested_func"
    )
    assert nested_func.namespace_stack == ["inner", "outer"]


def test_anonymous_namespace_declarations(decls):
    """Test declarations in anonymous namespaces."""
    # Test global anonymous namespace
    anon_struct = next(s for s in decls.structs if s.name == "AnonymousStruct")
    assert anon_struct.namespace_stack == [
        ""
    ]  # Anonymous namespace is represented as empty string

    # Test anonymous namespace nested in outer namespace
    outer_anon_struct = next(
        s for s in decls.structs if s.name == "OuterAnonymousStruct"
    )
    assert outer_anon_struct.namespace_stack == [
        "",
        "outer",
    ]  # Anonymous namespace in outer namespace
