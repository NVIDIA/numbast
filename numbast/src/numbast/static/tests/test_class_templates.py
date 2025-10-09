# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import pytest


from ast_canopy import parse_declarations_from_source

from numbast.static.renderer import clear_base_renderer_cache, registry_setup
from numbast.static.class_template import (
    StaticClassTemplatesRenderer,
)


@pytest.fixture(autouse=True, scope="module")
def cleanup():
    clear_base_renderer_cache()


@pytest.fixture(scope="module")
def decl(data_folder, cleanup):
    header = data_folder("class_template.cuh")

    decls = parse_declarations_from_source(header, [header], "sm_50")
    class_templates = decls.class_templates

    assert len(class_templates) == 1

    registry_setup(use_separate_registry=False)
    SFR = StaticClassTemplatesRenderer(class_templates)

    bindings = SFR.render_as_str(with_imports=True, with_shim_stream=True)
    globals = {}
    exec(bindings, globals)

    print(bindings)
    public_apis = ["BlockLoad"]
    assert all(public_api in globals for public_api in public_apis)

    return {k: globals[k] for k in public_apis}


def test_class_template(decl):
    print(decl)
    assert decl["BlockLoad"] is not None
