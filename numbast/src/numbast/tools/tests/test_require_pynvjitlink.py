# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import subprocess
import os

import pytest


@pytest.mark.parametrize("require_pynvjitlink", [True, False])
def test_require_pynvjitlink(
    run_in_isolated_folder, require_pynvjitlink, arch_str
):
    """Tests if `require_pynvjitlink` field conditionally adds pynvjitlink
    guard.
    """
    res = run_in_isolated_folder(
        "require_pynvjitlink.yml.j2",
        "data.cuh",
        {"require_pynvjitlink": require_pynvjitlink, "arch_str": arch_str},
        ruff_format=False,
    )

    binding = res["binding"]
    output_folder = res["output_folder"]

    assert (
        'if not importlib.util.find_spec("pynvjitlink"):' in binding
    ) is require_pynvjitlink, binding

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
