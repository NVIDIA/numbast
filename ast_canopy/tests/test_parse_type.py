# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from ast_canopy import parse_declarations_from_source


@pytest.fixture(scope="module")
def sample_type_source(data_folder):
    return data_folder / "sample_types.cu"


def test_parse_tests(sample_type_source):
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
