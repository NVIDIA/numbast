# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
import textwrap

import pytest


def _run_python_script(script_path):
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"Subprocess failed for {script_path}\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )


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


def test_class_template_runtime_kernel_in_subprocess(
    class_template_binding, tmp_path
):
    binding_path = tmp_path / "generated_class_template_binding.py"
    binding_path.write_text(class_template_binding["src"], encoding="utf-8")

    runner = textwrap.dedent(
        f"""
        import importlib.util
        import numpy as np

        from numba import cuda

        spec = importlib.util.spec_from_file_location(
            "generated_class_template_binding",
            r"{binding_path}",
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        TemplateBox = module.TemplateBox
        shim_obj = module.shim_obj

        @cuda.jit(link=[shim_obj])
        def kernel(inp_i32, inp_f32, out_i32, out_f32):
            box_i32 = TemplateBox(inp_i32[0], T=np.int32)
            box_f32 = TemplateBox(inp_f32[0], T=np.float32)
            out_i32[0] = box_i32.get()
            out_f32[0] = box_f32.get()

        inp_i32 = np.array([42], dtype=np.int32)
        inp_f32 = np.array([3.5], dtype=np.float32)
        out_i32 = np.array([0], dtype=np.int32)
        out_f32 = np.array([0], dtype=np.float32)

        kernel[1, 1](inp_i32, inp_f32, out_i32, out_f32)
        cuda.synchronize()

        assert out_i32[0] == 42
        np.testing.assert_allclose(out_f32, np.array([3.5], dtype=np.float32))
        """
    )

    runner_path = tmp_path / "run_class_template_kernel.py"
    runner_path.write_text(runner, encoding="utf-8")
    _run_python_script(runner_path)
