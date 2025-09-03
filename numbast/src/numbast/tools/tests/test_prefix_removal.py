# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys


def test_prefix_removal(run_in_isolated_folder, arch_str):
    """Test that API prefix removal works correctly for function names."""
    res = run_in_isolated_folder(
        "prefix_removal.yml.j2",
        "prefix_removal.cuh",
        {"arch_str": arch_str},
        load_symbols=True,
        ruff_format=False,
    )

    run_result = res["result"]
    output_folder = res["output_folder"]
    symbols = res["symbols"]
    alls = symbols["__all__"]

    assert run_result.exit_code == 0

    # Verify that the function is exposed as "foo" (without the "prefix_" prefix)
    assert "foo" in alls, f"Expected 'foo' in __all__, got: {alls}"

    # Verify that the original name "prefix_foo" is NOT exposed
    assert "prefix_foo" not in alls, (
        f"Expected 'prefix_foo' NOT in __all__, got: {alls}"
    )

    # Test that the function can be imported and used as "foo"
    test_kernel_src = """
from numba import cuda
from prefix_removal import foo

@cuda.jit
def kernel():
    result = foo(1, 2)  # Verify that prefix_foo is accessible as foo

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

    assert res.returncode == 0, res.stdout.decode("utf-8")
