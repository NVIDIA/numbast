# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone test for the unsupported-template-argument fix in
``ast_canopy/cpp/src/class_template_specialization.cpp``.

The class template specialization constructor previously threw
``std::runtime_error("Unsupported template argument kind")`` when it saw
any ``TemplateArgument::ArgKind`` other than ``Type`` or ``Integral``.
Parameter packs, for example, produce a ``Pack``-kind argument on the
specialization's argument list, which caused the whole parse to abort.

The fix records whether a declared template parameter is a pack and
prints ``Pack``-kind specialization arguments by joining their elements,
so parsing continues without losing the actual packed argument list.
"""

import os

import pytest

from ast_canopy import parse_declarations_from_source


@pytest.fixture(scope="module")
def source_path():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "data", "sample_unsupported_template_args.cu")


def _find_specialization(decls, qual_name):
    matches = [
        cts
        for cts in decls.class_template_specializations
        if cts.qual_name == qual_name
    ]
    assert len(matches) == 1, [
        cts.qual_name for cts in decls.class_template_specializations
    ]
    return matches[0]


def test_pack_specialization_does_not_throw(source_path):
    """A specialization with a parameter-pack argument must not abort the
    parse."""
    decls = parse_declarations_from_source(source_path, [source_path], "sm_80")
    assert decls is not None


def test_supported_specialization_still_parsed(source_path):
    """The presence of an unsupported-kind specialization must not prevent
    other, fully-supported specializations in the same header from being
    parsed."""
    decls = parse_declarations_from_source(source_path, [source_path], "sm_80")
    cts_names = [cts.qual_name for cts in decls.class_template_specializations]
    assert "Simple<float, 3>" in cts_names, cts_names


def test_variadic_template_parameter_is_marked_as_pack(source_path):
    """The Variadic template's single declared parameter should be
    represented as a type template parameter pack."""
    decls = parse_declarations_from_source(source_path, [source_path], "sm_80")
    variadic_templates = [
        ct for ct in decls.class_templates if ct.qual_name == "Variadic"
    ]
    assert len(variadic_templates) == 1, [
        ct.qual_name for ct in decls.class_templates
    ]
    params = variadic_templates[0].template_parameters
    assert len(params) == 1
    assert params[0].name == "Ts"
    assert params[0].is_pack


def test_pack_argument_is_expanded(source_path):
    """For the Variadic<int, float, double> specialization, the pack
    argument should preserve the packed argument list rather than using
    an unsupported placeholder."""
    decls = parse_declarations_from_source(source_path, [source_path], "sm_80")
    variadic = _find_specialization(decls, "Variadic<int, float, double>")
    args = variadic.actual_template_arguments
    assert args == ["int, float, double"]
