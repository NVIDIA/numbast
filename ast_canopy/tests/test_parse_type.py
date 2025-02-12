# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from ast_canopy import parse_declarations_from_source


@pytest.fixture(scope="module")
def sample_type_source(data_folder):
    return data_folder / "sample_types.cu"


def test_recursive_remove_const_qualifiers(sample_type_source):
    srcstr = str(sample_type_source)

    decls = parse_declarations_from_source(srcstr, [srcstr], "sm_80")
    structs = decls.structs

    assert len(structs) == 1

    Foo = structs[0]

    assert Foo.name == "Foo"

    expected = [
        ("a", "int", "int"),
        ("b", "const int", "int"),
        ("c", "int *", "int *"),
        ("d", "int **", "int * *"),
        ("e", "int &", "int"),
        ("f", "const int *", "int *"),
        ("g", "int *const", "int *"),
        ("h", "const int *const", "int *"),
        ("i", "const int *const *const", "int * *"),
    ]

    for e, f in zip(expected, Foo.fields):
        name, ty_name, qualifier_removed_name = e

        assert f.name == name
        assert f.type_.name == ty_name
        assert f.type_.unqualified_non_ref_type_name == qualifier_removed_name


def test_preserve_stdint_types(sample_type_source):
    srcstr = str(sample_type_source)

    decls = parse_declarations_from_source(srcstr, [srcstr], "sm_80")
    functions = decls.functions

    assert len(functions) == 1

    expected = [
        ("a", "uint64_t", "uint64_t"),
        ("b", "int_fast32_t", "int_fast32_t"),
        ("c", "int32_t &", "int32_t"),
        ("d", "uint8_t *", "uint8_t *"),
        ("e", "const int64_t *", "int64_t *"),
        ("f", "uint32_t *const", "uint32_t *"),
    ]

    assert functions[0].name == "bar"
    params = functions[0].params

    for (name, tyname, unqual_tyname), p in zip(expected, params):
        assert p.name == name
        assert p.type_.name == tyname
        assert p.type_.unqualified_non_ref_type_name == unqual_tyname
