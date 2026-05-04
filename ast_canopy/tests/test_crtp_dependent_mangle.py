# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone test for the dependent-signature mangle guard in
``ast_canopy/cpp/src/function.cpp``.

Calling ``clang::ItaniumMangleContext::mangleName`` on a ``FunctionDecl``
whose parameter or return types are still template-dependent segfaults
the host process (the mangler dereferences uninstantiated type nodes).
Because ast_canopy unconditionally mangled every function it visited,
*any* header containing a CRTP base class or a function template whose
return type depended on its parameters (e.g. Eigen's expression
templates) would segfault the Python interpreter.

The fix introduces ``has_dependent_signature(FD)`` and skips Itanium
mangling when the signature is still dependent, using an explicit
dependent-signature fallback instead. It also replaces a raw ``create``
with a ``unique_ptr`` to plug a leak.
"""

import os

import pytest

from ast_canopy import parse_declarations_from_source


@pytest.fixture(scope="module")
def source_path():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "data", "sample_crtp_dependent.cu")


def test_crtp_header_does_not_segfault(source_path):
    """Parsing a CRTP base with dependent-signature methods must not
    crash the process."""
    decls = parse_declarations_from_source(source_path, [source_path], "sm_80")
    assert decls is not None


def test_crtp_templates_are_captured(source_path):
    """Both CRTPBase and Vec3 class templates should be parsed."""
    decls = parse_declarations_from_source(source_path, [source_path], "sm_80")
    names = [ct.qual_name for ct in decls.class_templates]
    assert any("CRTPBase" in n for n in names), names
    assert any("Vec3" in n for n in names), names


def test_dependent_function_template_captured(source_path):
    """extract_value has a dependent return type. It should appear as a
    function template without having segfaulted the mangler."""
    decls = parse_declarations_from_source(source_path, [source_path], "sm_80")
    funcs = [
        ft.function
        for ft in decls.function_templates
        if ft.function.name == "extract_value"
    ]
    assert funcs, [ft.function.name for ft in decls.function_templates]

    mangled = funcs[0].mangled_name
    assert mangled
    assert not mangled.startswith("_Z"), mangled
    assert "dependent_signature" in mangled, mangled


def test_non_dependent_function_still_mangled(source_path):
    """The fix must not regress mangling for concrete device functions:
    vec3_dot should have a non-empty Itanium-style mangled name."""
    decls = parse_declarations_from_source(source_path, [source_path], "sm_80")
    funcs = {f.name: f for f in decls.functions}
    assert "vec3_dot" in funcs
    mangled = funcs["vec3_dot"].mangled_name
    # Itanium mangling prefixes with `_Z`.
    assert mangled.startswith("_Z"), mangled
