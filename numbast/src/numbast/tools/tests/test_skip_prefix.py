# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys

def test_skip_prefix(run_in_isolated_folder, arch_str):
    """Tests:
    1. Output binding can be skipped via `Skip Prefix` entry.
    """

    res = run_in_isolated_folder(
        "skip_prefix.yml.j2",
        "data.cuh",
        {"arch_str": arch_str, "skip_prefix": "m"}, # this will skip `mul`
        ruff_format=False,
    )

    run_result = res["result"]
    output_folder = res["output_folder"]
    binding_path = res["binding_path"]

    assert run_result.exit_code == 0

    test_kernel_src = f"""
from numba import cuda
import data
for sym in data.__all__:
    print(sym)
"""

    test_kernel = os.path.join(output_folder, "test.py")
    with open(test_kernel, "w") as f:
        f.write(test_kernel_src)

    res = subprocess.run(
        [sys.executable, test_kernel],
        cwd=output_folder,
        capture_output=True,
        text=True,
    )

    assert res.returncode == 0, res.stdout
    assert "add" in res.stdout
    assert "mul" not in res.stdout
