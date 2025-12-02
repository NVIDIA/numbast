# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys

from numbast.static.renderer import clear_base_renderer_cache
from numbast.static.function import clear_function_apis_registry

from cuda.core.experimental import Device

dev = Device(0)
cc = dev.compute_capability


def test_symbol_exposure(run_in_isolated_folder, arch_str):
    """Test that only a limited set of symbols are exposed via __all__ imports."""
    clear_base_renderer_cache()
    clear_function_apis_registry()

    res = run_in_isolated_folder(
        "cfg.yml.j2",
        "data.cuh",
        {"arch_str": arch_str},
        load_symbols=True,
        ruff_format=False,
    )

    run_result = res["result"]
    output_folder = res["output_folder"]
    symbols = res["symbols"]
    alls = symbols["__all__"]

    assert run_result.exit_code == 0

    if cc >= (8, 6):
        assert len(alls) == 4, (
            len(alls) != 4,
            alls,
        )  # Foo, add, mul, _type_Foo
    else:
        assert len(alls) == 3, (len(alls) != 3, alls)  # Foo, add, _type_Foo

    test_kernel_src = f"""
from numba import cuda
from data import *

@cuda.jit
def kernel():
    foo = Foo()         # Verify record symbol
    one = add(foo.x, 1) # Verify function symbol
    {"two = mul(one, 2)" if cc >= (8, 6) else ""}

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
