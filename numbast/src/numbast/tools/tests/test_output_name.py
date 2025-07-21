# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys


def test_output_name_override(run_in_isolated_folder, arch_str):
    """Tests:
    1. Name of output binding can be overridden via `Output Name` entry.
    """

    module_name = "bindings"
    output_name = f"{module_name}.py"
    res = run_in_isolated_folder(
        "output_name.yml.j2",
        "data.cuh",
        {"arch_str": arch_str},
        output_name=output_name,
        ruff_format=False,
    )

    run_result = res["result"]
    output_folder = res["output_folder"]
    binding_path = res["binding_path"]

    assert run_result.exit_code == 0
    assert output_name in binding_path

    test_kernel_src = f"""
from numba import cuda
from {module_name} import Foo, add
@cuda.jit
def kernel():
    foo = Foo()
    one = add(foo.x, 1)

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
