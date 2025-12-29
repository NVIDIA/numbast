# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import os
import re

import pytest

from ast_canopy import parse_declarations_from_source


@pytest.fixture(scope="module")
def sample_qual_name_source():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_directory, "data/", "sample_qual_name.cu")


def test_qual_name_is_exposed(sample_qual_name_source):
    """
    `qual_name` is part of the public surface area for both:
    - wrapped decls (`ast_canopy.decl.*`), and
    - raw binding decls (`pylibastcanopy.*`).

    This test is intentionally broad: it asserts the attribute exists and is a
    non-empty string for declarations we expose.

    Notes on "anonymous" declarations:
    - For anonymous records, Clang can produce empty qualified names. ast_canopy
      synthesizes a stable placeholder name (`unnamed<ID>`) so `qual_name`
      remains usable. That behavior is verified in a dedicated test below.
    """
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
    """
    In named namespaces, `qual_name` should follow C++ scoping rules, e.g.
    `ns1::ns2::S::m`.
    """
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


def test_qual_name_global_scope_is_unqualified(sample_qual_name_source):
    """
    In the global scope (no namespace), Clang's qualified name is typically the
    unqualified identifier. We treat that as the canonical `qual_name`.
    """
    decls = parse_declarations_from_source(
        sample_qual_name_source, [sample_qual_name_source], "sm_80"
    )

    s = next(s for s in decls.structs if s.name == "GlobalS")
    assert s.qual_name == "GlobalS"
    m = next(m for m in s.methods if m.name == "m")
    assert m.qual_name == "GlobalS::m"

    f = next(f for f in decls.functions if f.name == "global_f")
    assert f.qual_name == "global_f"

    ft = next(
        ft for ft in decls.function_templates if ft.function.name == "global_tf"
    )
    assert ft.qual_name == "global_tf"
    assert ft.function.qual_name == "global_tf"

    ct = next(
        ct for ct in decls.class_templates if ct.record.name == "GlobalTpl"
    )
    assert ct.qual_name == "GlobalTpl"
    assert ct.record.qual_name == "GlobalTpl"

    e = next(e for e in decls.enums if e.name == "GlobalE")
    assert e.qual_name == "GlobalE"

    td = next(td for td in decls.typedefs if td.name == "GlobalAlias")
    assert td.qual_name == "GlobalAlias"


def test_qual_name_anonymous_record_is_linked_from_typedef(
    sample_qual_name_source,
):
    """
    C-style anonymous record typedef:
        typedef struct { ... } CStyleAnon;

    The underlying RecordDecl has no tag name and Clang typically reports an
    empty name. ast_canopy then falls back to `unnamed<ID>` using Clang's
    internal Decl ID (not stable across runs).

    The corresponding Typedef points to the underlying record via
    `.underlying_name` (which will commonly look like `unnamed<ID>`).

    However, Clang often treats the typedef name as the *record's* qualified
    name in this pattern. In that case, we expect:
    - Record.name == "unnamed<ID>" (ast_canopy fallback)
    - Record.qual_name == "CStyleAnon" (Clang-derived user-visible name)
    """
    decls = parse_declarations_from_source(
        sample_qual_name_source, [sample_qual_name_source], "sm_80"
    )

    td = next(td for td in decls.typedefs if td.name == "CStyleAnon")
    assert td.qual_name == "CStyleAnon"
    assert re.fullmatch(r"unnamed\d+", td.underlying_name), td.underlying_name

    anon_record = next(s for s in decls.structs if s.name == td.underlying_name)
    assert anon_record.qual_name == td.name
    assert {f.name for f in anon_record.fields} >= {"a", "b"}


def test_qual_name_anonymous_namespace_is_marked(sample_qual_name_source):
    """
    Anonymous namespace:
        namespace { ... }

    Clang typically renders this as `(anonymous namespace)` in qualified names.
    We assert on the marker and the final identifier, rather than requiring an
    exact full string, to keep the test resilient across supported Clang
    versions.
    """
    decls = parse_declarations_from_source(
        sample_qual_name_source, [sample_qual_name_source], "sm_80"
    )

    s = next(s for s in decls.structs if s.name == "AnonNS_S")
    assert "(anonymous namespace)" in s.qual_name
    assert s.qual_name.endswith("::AnonNS_S")

    f = next(f for f in decls.functions if f.name == "anon_ns_f")
    assert "(anonymous namespace)" in f.qual_name
    assert f.qual_name.endswith("::anon_ns_f")
