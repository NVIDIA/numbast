# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys


def test_module_callbacks_empty(run_in_isolated_folder, arch_str):
    """Test that Module Callbacks work when not provided (empty case)."""
    res = run_in_isolated_folder(
        "module_callbacks.yml.j2",
        "module_callbacks.cuh",
        {
            "arch_str": arch_str,
            "setup_callback": "",
            "teardown_callback": "",
        },
        ruff_format=False,
    )

    run_result = res["result"]
    output_folder = res["output_folder"]
    binding = res["binding"]

    assert run_result.exit_code == 0

    # Verify that no callbacks are set when empty
    assert "shim_obj.setup_callback" not in binding, (
        f"Expected no setup_callback when empty, got: {binding}"
    )
    assert "shim_obj.teardown_callback" not in binding, (
        f"Expected no teardown_callback when empty, got: {binding}"
    )

    # Test that the function can be imported and used in a CUDA kernel
    test_kernel_src = """
from numba import cuda
from module_callbacks import test_function

@cuda.jit
def kernel():
    result = test_function(1, 2)  # Verify that test_function works end-to-end

kernel[1, 1]()
"""

    test_kernel = os.path.join(output_folder, "test.py")
    with open(test_kernel, "w") as f:
        f.write(test_kernel_src)

    res = subprocess.run(
        [sys.executable, test_kernel],
        cwd=output_folder,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    stdout_text = res.stdout.decode("utf-8")
    assert res.returncode == 0, stdout_text

    # Since no callbacks are set, there should be no callback output
    assert "Setup only" not in stdout_text, (
        f"Expected no setup callback output, got: {stdout_text}"
    )
    assert "Setup callback called" not in stdout_text, (
        f"Expected no setup callback output, got: {stdout_text}"
    )
    assert "Teardown callback called" not in stdout_text, (
        f"Expected no teardown callback output, got: {stdout_text}"
    )


def test_module_callbacks_partial(run_in_isolated_folder, arch_str):
    """Test that Module Callbacks work when only one callback is provided."""
    res = run_in_isolated_folder(
        "module_callbacks.yml.j2",
        "module_callbacks.cuh",
        {
            "arch_str": arch_str,
            "setup_callback": "lambda x: print('Setup only')",
            "teardown_callback": "",
        },
        ruff_format=False,
    )

    run_result = res["result"]
    output_folder = res["output_folder"]
    binding = res["binding"]

    assert run_result.exit_code == 0

    # Verify that only the setup_callback is set
    assert (
        "shim_obj.setup_callback = lambda x: print('Setup only')" in binding
    ), f"Expected setup_callback to be set in binding, got: {binding}"
    assert "shim_obj.teardown_callback" not in binding, (
        f"Expected no teardown_callback when empty, got: {binding}"
    )

    # Test that the function can be imported and used in a CUDA kernel
    test_kernel_src = """
from numba import cuda
from module_callbacks import test_function

@cuda.jit
def kernel():
    result = test_function(1, 2)  # Verify that test_function works end-to-end

kernel[1, 1]()
"""

    test_kernel = os.path.join(output_folder, "test.py")
    with open(test_kernel, "w") as f:
        f.write(test_kernel_src)

    res = subprocess.run(
        [sys.executable, test_kernel],
        cwd=output_folder,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    stdout_text = res.stdout.decode("utf-8")
    assert res.returncode == 0, stdout_text

    # Verify that setup callback was called and printed its message
    assert "Setup only" in stdout_text, (
        f"Expected 'Setup only' in stdout, got: {stdout_text}"
    )
    # Verify that teardown callback was not called since it's empty
    assert "Teardown callback called" not in stdout_text, (
        f"Expected no teardown callback output, got: {stdout_text}"
    )


def test_module_callbacks_both(run_in_isolated_folder, arch_str):
    """Test that Module Callbacks work when both setup and teardown are provided."""
    res = run_in_isolated_folder(
        "module_callbacks.yml.j2",
        "module_callbacks.cuh",
        {
            "arch_str": arch_str,
            "setup_callback": "lambda x: print('Setup callback called')",
            "teardown_callback": "lambda x: print('Teardown callback called')",
        },
        ruff_format=False,
    )

    run_result = res["result"]
    output_folder = res["output_folder"]
    binding = res["binding"]

    assert run_result.exit_code == 0

    # Verify that both callbacks are set in the binding
    assert (
        "shim_obj.setup_callback = lambda x: print('Setup callback called')"
        in binding
    ), f"Expected setup_callback to be set in binding, got: {binding}"
    assert (
        "shim_obj.teardown_callback = lambda x: print('Teardown callback called')"
        in binding
    ), f"Expected teardown_callback to be set in binding, got: {binding}"

    # Test that the function can be imported and used in a CUDA kernel
    test_kernel_src = """
from numba import cuda
from module_callbacks import test_function

@cuda.jit
def kernel():
    result = test_function(1, 2)  # Verify that test_function works end-to-end

kernel[1, 1]()
"""

    test_kernel = os.path.join(output_folder, "test.py")
    with open(test_kernel, "w") as f:
        f.write(test_kernel_src)

    res = subprocess.run(
        [sys.executable, test_kernel],
        cwd=output_folder,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    stdout_text = res.stdout.decode("utf-8")
    assert res.returncode == 0, stdout_text

    # Verify that both callbacks were called and printed their messages
    assert "Setup callback called" in stdout_text, (
        f"Expected 'Setup callback called' in stdout, got: {stdout_text}"
    )
    assert "Teardown callback called" in stdout_text, (
        f"Expected 'Teardown callback called' in stdout, got: {stdout_text}"
    )
