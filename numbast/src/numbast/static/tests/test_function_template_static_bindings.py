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


def test_function_template_runtime_kernel_in_subprocess(
    function_template_binding, tmp_path
):
    binding_path = tmp_path / "generated_function_template_binding.py"
    binding_path.write_text(function_template_binding["src"], encoding="utf-8")

    runner = textwrap.dedent(
        f"""
        import importlib.util
        import numpy as np

        from numba import cuda

        spec = importlib.util.spec_from_file_location(
            "generated_function_template_binding",
            r"{binding_path}",
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        templated_add = module.templated_add
        shim_obj = module.shim_obj

        @cuda.jit(link=[shim_obj])
        def kernel(int_a, int_b, float_a, float_b, out_i32, out_f32):
            out_i32[0] = templated_add(int_a[0], int_b[0])
            out_f32[0] = templated_add(float_a[0], float_b[0])

        int_a = np.array([3], dtype=np.int32)
        int_b = np.array([4], dtype=np.int32)
        float_a = np.array([1.5], dtype=np.float32)
        float_b = np.array([2.5], dtype=np.float32)
        out_i32 = np.array([0], dtype=np.int32)
        out_f32 = np.array([0], dtype=np.float32)

        kernel[1, 1](int_a, int_b, float_a, float_b, out_i32, out_f32)
        cuda.synchronize()

        assert out_i32[0] == 7
        np.testing.assert_allclose(out_f32, np.array([4.0], dtype=np.float32))
        """
    )

    runner_path = tmp_path / "run_function_template_kernel.py"
    runner_path.write_text(runner, encoding="utf-8")
    _run_python_script(runner_path)
