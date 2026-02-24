# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest


@pytest.fixture(scope="function")
def function_template_binding(make_binding):
    return make_binding("function_template_static.cuh", {}, {}, "sm_50")


def test_function_template_symbol_is_available(function_template_binding):
    bindings = function_template_binding["bindings"]

    assert "templated_add" in bindings
    assert "TemplateBox" not in bindings
    assert callable(bindings["templated_add"])


def test_generated_source_includes_function_template_section(
    function_template_binding,
):
    src = function_template_binding["src"]

    assert "# Function Templates:" in src
    assert "bind_static_function_templates(" in src
