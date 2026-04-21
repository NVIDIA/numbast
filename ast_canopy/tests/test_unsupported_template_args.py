# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone test for the unsupported-template-argument fix in
``ast_canopy/cpp/src/class_template_specialization.cpp``.

The class template specialization constructor previously threw
``std::runtime_error("Unsupported template argument kind")`` when it saw
any ``TemplateArgument::ArgKind`` other than ``Type`` or ``Integral``.
Parameter packs, for example, produce a ``Pack``-kind argument on the
specialization's argument list, which caused the whole parse to abort.

The fix substitutes a ``"<unsupported>"`` placeholder string so parsing
continues.
"""

import os

import pytest

from ast_canopy import parse_declarations_from_source


@pytest.fixture(scope="module")
def source_path():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "data", "sample_unsupported_template_args.cu")


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
    assert any("Simple" in n for n in cts_names), cts_names


def test_unsupported_pack_has_placeholder_argument(source_path):
    """For the Variadic<int, float, double> specialization, the pack
    argument should be represented by a placeholder string rather than
    having thrown."""
    decls = parse_declarations_from_source(source_path, [source_path], "sm_80")
    variadic = [
        cts
        for cts in decls.class_template_specializations
        if "Variadic" in cts.qual_name
    ]
    assert variadic, "Variadic specialization missing from parsed decls"
    args = variadic[0].actual_template_arguments
    assert "<unsupported>" in args, args
