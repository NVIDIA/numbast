# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone test for the safe record_id lookup in
``ast_canopy/cpp/src/typedef.cpp``.

IDs → names that the record and class-template-specialization matchers
captured. The Typedef constructor looked up the underlying record's ID
with ``map::at``, which throws ``std::out_of_range`` when the ID isn't
present. Class template instantiations are captured by a separate
matcher, but their IDs also need to be registered because typedefs can
refer to them.

The fix registers class template specialization IDs in the shared map
without forcing incomplete typedef-only instantiations to materialize,
uses ``find`` for lookup, and keeps a fallback for unregistered or
non-record underlying types.
"""

import os

import pytest

from ast_canopy import parse_declarations_from_source


@pytest.fixture(scope="module")
def source_path():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "data", "sample_typedef_template_inst.cu")


def test_typedef_over_template_instantiation_does_not_throw(source_path):
    """Parsing a typedef to a class template instantiation must not
    abort."""
    decls = parse_declarations_from_source(source_path, [source_path], "sm_80")
    assert decls is not None


def test_both_typedefs_are_parsed(source_path):
    """Both Vec3fStorage and Vec4dStorage should appear in the parsed
    typedefs, proving neither lookup threw."""
    decls = parse_declarations_from_source(source_path, [source_path], "sm_80")
    names = [td.name for td in decls.typedefs]
    assert "Vec3fStorage" in names, names
    assert "Vec4dStorage" in names, names


def test_underlying_name_comes_from_registered_template_specialization(
    source_path,
):
    """Template instantiation typedefs resolve through the registered
    specialization record name."""
    decls = parse_declarations_from_source(source_path, [source_path], "sm_80")
    by_name = {td.name: td for td in decls.typedefs}
    assert by_name["Vec3fStorage"].underlying_name == "Storage"
    assert by_name["Vec4dStorage"].underlying_name == "Storage"
