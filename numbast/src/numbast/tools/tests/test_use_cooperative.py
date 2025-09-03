# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys

import pytest


def test_use_cooperative(run_in_isolated_folder, arch_str):
    """Test that only a limited set of symbols are exposed via __all__ imports."""
    res = run_in_isolated_folder(
        "cooperative_launch.yml.j2",
        "use_cooperative.cuh",
        {"arch_str": arch_str},
        load_symbols=True,
        ruff_format=False,
    )

    run_result = res["result"]
    output_folder = res["output_folder"]

    assert run_result.exit_code == 0

    test_kernel_src = """
from numba import cuda
from use_cooperative import cta_barrier
@cuda.jit
def kernel():
    cta_barrier()

kernel[1, 1]()
assert kernel.overloads[()].cooperative, f"{{kernel.overloads[()].cooperative=}}"
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

    assert res.returncode == 0, res.stdout.decode("utf-8")


def test_use_cooperative_regex_patterns(run_in_isolated_folder, arch_str):
    """Test that regex patterns correctly match function names for cooperative launch."""
    res = run_in_isolated_folder(
        "cooperative_launch_regex.yml.j2",
        "use_cooperative.cuh",
        {"arch_str": arch_str},
        load_symbols=True,
        ruff_format=False,
    )

    run_result = res["result"]
    output_folder = res["output_folder"]

    assert run_result.exit_code == 0

    # Test functions that should match .*_barrier.* pattern
    test_barrier_kernel_src = """
from numba import cuda
from use_cooperative import global_barrier_sync, thread_barrier_wait

@cuda.jit
def barrier_kernel():
    global_barrier_sync()

@cuda.jit
def thread_barrier_kernel():
    thread_barrier_wait()

barrier_kernel[1, 1]()
thread_barrier_kernel[1, 1]()

assert barrier_kernel.overloads[()].cooperative, f"global_barrier_sync should be cooperative: {{barrier_kernel.overloads[()].cooperative=}}"
assert thread_barrier_kernel.overloads[()].cooperative, f"thread_barrier_wait should be cooperative: {{thread_barrier_kernel.overloads[()].cooperative=}}"
"""

    # Test functions that should match .*_sync.* pattern
    test_sync_kernel_src = """
from numba import cuda
from use_cooperative import grid_sync_all, block_sync_threads

@cuda.jit
def grid_sync_kernel():
    grid_sync_all()

@cuda.jit
def block_sync_kernel():
    block_sync_threads()

grid_sync_kernel[1, 1]()
block_sync_kernel[1, 1]()

assert grid_sync_kernel.overloads[()].cooperative, f"grid_sync_all should be cooperative: {{grid_sync_kernel.overloads[()].cooperative=}}"
assert block_sync_kernel.overloads[()].cooperative, f"block_sync_threads should be cooperative: {{block_sync_kernel.overloads[()].cooperative=}}"
"""

    # Test barrier functions
    test_barrier_kernel = os.path.join(output_folder, "test_barrier.py")
    with open(test_barrier_kernel, "w") as f:
        f.write(test_barrier_kernel_src)

    res = subprocess.run(
        [sys.executable, test_barrier_kernel],
        cwd=output_folder,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    assert res.returncode == 0, res.stdout.decode("utf-8")

    # Test sync functions
    test_sync_kernel = os.path.join(output_folder, "test_sync.py")
    with open(test_sync_kernel, "w") as f:
        f.write(test_sync_kernel_src)

    res = subprocess.run(
        [sys.executable, test_sync_kernel],
        cwd=output_folder,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    assert res.returncode == 0, res.stdout.decode("utf-8")


def test_use_cooperative_non_matching_functions(
    run_in_isolated_folder, arch_str
):
    """Test that functions not matching regex patterns are not marked as cooperative."""
    res = run_in_isolated_folder(
        "cooperative_launch_regex.yml.j2",
        "use_cooperative.cuh",
        {"arch_str": arch_str},
        load_symbols=True,
        ruff_format=False,
    )

    run_result = res["result"]
    output_folder = res["output_folder"]

    assert run_result.exit_code == 0

    # Test functions that should NOT match any patterns
    test_non_coop_kernel_src = """
from numba import cuda
from use_cooperative import regular_function

@cuda.jit
def regular_kernel():
    regular_function()

regular_kernel[1, 1]()

# These functions should NOT be marked as cooperative
assert not regular_kernel.overloads[()].cooperative, f"regular_function should not be cooperative: {{regular_kernel.overloads[()].cooperative=}}"
"""

    test_non_coop_kernel = os.path.join(output_folder, "test_non_coop.py")
    with open(test_non_coop_kernel, "w") as f:
        f.write(test_non_coop_kernel_src)

    res = subprocess.run(
        [sys.executable, test_non_coop_kernel],
        cwd=output_folder,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    assert res.returncode == 0, res.stdout.decode("utf-8")


def test_use_cooperative_invalid_regex(run_in_isolated_folder, arch_str):
    """Test that functions not matching regex patterns are not marked as cooperative."""

    with pytest.raises(ValueError, match="Invalid regex pattern"):
        run_in_isolated_folder(
            "cooperative_launch_invalid_regex.yml.j2",
            "use_cooperative.cuh",
            {"arch_str": arch_str},
            load_symbols=True,
            ruff_format=False,
        )
