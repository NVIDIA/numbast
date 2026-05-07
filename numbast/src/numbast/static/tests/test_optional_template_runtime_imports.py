# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from ast_canopy.pylibastcanopy import execution_space

from numbast.static.class_template import (
    StaticClassTemplatesRenderer,
    clear_class_template_cache,
)
from numbast.static.function import clear_function_apis_registry
from numbast.static.function_template import (
    StaticFunctionTemplatesRenderer,
    clear_function_template_registry,
)
from numbast.static.renderer import BaseRenderer, clear_base_renderer_cache
from numbast.tools.static_binding_generator import (
    Config,
    _static_binding_generator,
)


_FUNCTION_TEMPLATE_IMPORT = (
    "from numbast.static.function_template import "
    "bind_static_function_templates"
)
_CLASS_TEMPLATE_IMPORT = (
    "from numbast.static.class_template import bind_static_class_templates"
)


def _reset_static_state():
    clear_base_renderer_cache()
    clear_function_apis_registry()
    clear_function_template_registry()
    clear_class_template_cache()


def _assert_no_template_runtime_imports(src: str):
    assert "from numbast" not in src
    assert "import numbast" not in src
    assert "from ast_canopy" not in src
    assert "import ast_canopy" not in src
    assert _FUNCTION_TEMPLATE_IMPORT not in src
    assert _CLASS_TEMPLATE_IMPORT not in src
    assert "bind_static_function_templates(" not in src
    assert "bind_static_class_templates(" not in src


def _generate_static_source(
    *,
    tmpdir,
    data_folder,
    header_name: str,
    exclude_functions: list[str] | None = None,
    exclude_structs: list[str] | None = None,
) -> str:
    _reset_static_state()
    header_path = data_folder(header_name)
    cfg = Config.from_params(
        entry_point=header_path,
        retain_list=[header_path],
        gpu_arch=["sm_50"],
        types={},
        datamodels={},
        exclude_functions=exclude_functions,
        exclude_structs=exclude_structs,
    )
    _static_binding_generator(cfg, tmpdir)

    output_name = header_name.split(".")[0] + ".py"
    with open(tmpdir / output_name) as f:
        return f.read()


def test_non_template_generated_binding_omits_template_runtime_imports(
    make_binding,
):
    src = make_binding("function.cuh", {}, {}, "sm_50")["src"]

    _assert_no_template_runtime_imports(src)


def test_empty_template_renderers_do_not_add_runtime_imports():
    _reset_static_state()

    function_template_src = StaticFunctionTemplatesRenderer(
        [],
        excludes=[],
        skip_prefix=None,
        skip_non_device=True,
    ).render_as_str(with_imports=False, with_shim_stream=False)
    class_template_src = StaticClassTemplatesRenderer(
        [],
        header_path="unused.cuh",
        excludes=[],
    ).render_as_str(with_imports=False, with_shim_stream=False)

    assert function_template_src == ""
    assert class_template_src == ""
    assert _FUNCTION_TEMPLATE_IMPORT not in BaseRenderer.Imports
    assert _CLASS_TEMPLATE_IMPORT not in BaseRenderer.Imports


def test_excluded_template_renderers_do_not_add_runtime_imports():
    _reset_static_state()
    function_template = SimpleNamespace(
        function=SimpleNamespace(
            name="templated_add",
            exec_space=execution_space.device,
            is_operator=False,
            is_overloaded_operator=lambda: False,
        )
    )
    class_template = SimpleNamespace(
        record=SimpleNamespace(name="TemplateBox", qual_name="TemplateBox")
    )

    function_template_src = StaticFunctionTemplatesRenderer(
        [function_template],
        excludes=["templated_add"],
        skip_prefix=None,
        skip_non_device=True,
    ).render_as_str(with_imports=False, with_shim_stream=False)
    class_template_src = StaticClassTemplatesRenderer(
        [class_template],
        header_path="unused.cuh",
        excludes=["TemplateBox"],
    ).render_as_str(with_imports=False, with_shim_stream=False)

    assert function_template_src == ""
    assert class_template_src == ""
    assert _FUNCTION_TEMPLATE_IMPORT not in BaseRenderer.Imports
    assert _CLASS_TEMPLATE_IMPORT not in BaseRenderer.Imports


def test_generated_function_template_binding_omits_import_when_excluded(
    tmpdir,
    data_folder,
):
    src = _generate_static_source(
        tmpdir=tmpdir,
        data_folder=data_folder,
        header_name="function_template_static.cuh",
        exclude_functions=["templated_add"],
    )

    _assert_no_template_runtime_imports(src)


def test_generated_class_template_binding_omits_import_when_excluded(
    tmpdir,
    data_folder,
):
    src = _generate_static_source(
        tmpdir=tmpdir,
        data_folder=data_folder,
        header_name="class_template_static.cuh",
        exclude_structs=["TemplateBox"],
    )

    _assert_no_template_runtime_imports(src)
