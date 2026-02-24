# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest


@pytest.fixture(scope="function")
def class_template_binding(make_binding):
    return make_binding("class_template_static.cuh", {}, {}, "sm_50")


def test_class_template_symbol_is_available(class_template_binding):
    bindings = class_template_binding["bindings"]

    assert "TemplateBox" in bindings
    assert "templated_add" not in bindings
    assert isinstance(bindings["TemplateBox"], type)


def test_generated_source_includes_class_template_section(
    class_template_binding,
):
    src = class_template_binding["src"]

    assert "# Class Templates:" in src
    assert "bind_static_class_templates(" in src
