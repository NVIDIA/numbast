# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys


def test_use_cooperative(run_in_isolated_folder):
    """Test that only a limited set of symbols are exposed via __all__ imports."""
    res = run_in_isolated_folder(
        "cooperative_launch.yml.j2",
        "use_cooperative.cuh",
        {},
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
