# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone test for the TemplateTemplateParmDecl fix in
``ast_canopy/cpp/src/template_param.cpp``.

Before the fix, the ``TemplateParam`` constructor for a
``TemplateTemplateParmDecl`` threw ``std::runtime_error``. Any class
template that used a template template parameter (e.g.
``template <typename T, template <typename> class C> struct Adapter``)
would abort parsing.
"""

import os

import pytest

from ast_canopy import parse_declarations_from_source


@pytest.fixture(scope="module")
def source_path():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "data", "sample_template_template_param.cu")


def test_template_template_param_does_not_throw(source_path):
    """Parsing a header with a template template parameter must not throw."""
    decls = parse_declarations_from_source(source_path, [source_path], "sm_80")
    assert decls is not None


def test_adapter_class_template_is_parsed(source_path):
    """The Adapter class template (which uses a template template param)
    must appear in the parsed class templates."""
    decls = parse_declarations_from_source(source_path, [source_path], "sm_80")
    names = [ct.qual_name for ct in decls.class_templates]
    assert any("Adapter" in n for n in names), names
