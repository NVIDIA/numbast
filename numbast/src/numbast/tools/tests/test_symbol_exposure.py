# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys


def test_symbol_exposure(run_in_isolated_folder):
    """Test that only a limited set of symbols are exposed via __all__ imports."""
    res = run_in_isolated_folder(
        "cfg.yml.j2", "data.cuh", {}, ruff_format=False
    )

    run_result = res["result"]
    output_folder = res["output_folder"]

    assert run_result.exit_code == 0

    test_kernel_src = """
from numba import cuda
from data import *
@cuda.jit
def kernel():
    foo = Foo()         # Verify record symbol
    one = add(foo.x, 1) # Verify function synbol

kernel[1, 1]()

t = _type_Foo           # Verify type object symbol
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
