# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys


def test_shim_include_override_additional_import(
    run_in_isolated_folder, arch_str
):
    """Tests:
    1. Additional Import field actually adds custom import libs in binding
    2. Shim Include Override overrides the shim include line
    """
    res = run_in_isolated_folder(
        "shim_include_override.yml.j2",
        "data.cuh",
        {"arch_str": arch_str},
        ruff_format=False,
    )

    run_result = res["result"]
    binding_path = res["binding_path"]
    output_folder = res["output_folder"]

    assert run_result.exit_code == 0

    with open(binding_path) as f:
        bindings = f.readlines()

    os_is_imported = False
    for line in bindings:
        # check import
        if line.startswith("import"):
            os_is_imported = True
        # check shim include override
        if line.startswith("shim_include"):
            assert "os" in line
    assert os_is_imported

    test_kernel_src = """
from numba import cuda
from data import Foo, add
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
