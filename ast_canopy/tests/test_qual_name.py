# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest

from ast_canopy import parse_declarations_from_source


@pytest.fixture(scope="module")
def sample_qual_name_source():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_directory, "data/", "sample_qual_name.cu")


def test_qual_name_is_exposed(sample_qual_name_source):
    decls = parse_declarations_from_source(
        sample_qual_name_source, [sample_qual_name_source], "sm_80"
    )

    # Wrapped decls (ast_canopy.decl.*)
    assert all(
        isinstance(s.qual_name, str) and s.qual_name for s in decls.structs
    )
    assert all(
        isinstance(f.qual_name, str) and f.qual_name for f in decls.functions
    )
    assert all(
        isinstance(ft.qual_name, str) and ft.qual_name
        for ft in decls.function_templates
    )
    assert all(
        isinstance(ct.qual_name, str) and ct.qual_name
        for ct in decls.class_templates
    )

    # Raw binding decls (pylibastcanopy.*)
    assert all(
        isinstance(e.qual_name, str) and e.qual_name for e in decls.enums
    )
    assert all(
        isinstance(td.qual_name, str) and td.qual_name for td in decls.typedefs
    )


def test_qual_name_matches_namespaces(sample_qual_name_source):
    decls = parse_declarations_from_source(
        sample_qual_name_source, [sample_qual_name_source], "sm_80"
    )

    s = next(s for s in decls.structs if s.name == "S")
    assert s.qual_name == "ns1::ns2::S"

    m = next(m for m in s.methods if m.name == "m")
    assert m.qual_name == "ns1::ns2::S::m"

    f = next(f for f in decls.functions if f.name == "f")
    assert f.qual_name == "ns1::ns2::f"

    ft = next(ft for ft in decls.function_templates if ft.function.name == "tf")
    assert ft.qual_name == "ns1::ns2::tf"
    assert ft.function.qual_name == "ns1::ns2::tf"

    ct = next(ct for ct in decls.class_templates if ct.record.name == "Tpl")
    assert ct.qual_name == "ns1::ns2::Tpl"
    assert ct.record.qual_name == "ns1::ns2::Tpl"

    e = next(e for e in decls.enums if e.name == "E")
    assert e.qual_name == "ns1::ns2::E"

    td = next(td for td in decls.typedefs if td.name == "Alias")
    assert td.qual_name == "ns1::ns2::Alias"
