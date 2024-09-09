# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from functools import partial

from numba import cuda
from numba.cuda import device_array

from ast_canopy import parse_declarations_from_source
from numbast.static.function import StaticFunctionsRenderer


@pytest.fixture(scope="module")
def cuda_function(data_folder):
    header = data_folder("function.cuh")

    _, functions, *_ = parse_declarations_from_source(header, [header], "sm_50")

    assert len(functions) == 2

    SFR = StaticFunctionsRenderer(functions, header)

    bindings = SFR.render_as_str()
    globals = {}
    exec(bindings, globals)

    public_apis = ["add", "c_ext_shim_source"]
    assert all(public_api in globals for public_api in public_apis)

    return {k: globals[k] for k in public_apis}


@pytest.fixture(scope="module")
def numbast_jit(cuda_function):
    c_ext_shim_source = cuda_function["c_ext_shim_source"]
    return partial(cuda.jit, link=[c_ext_shim_source])


@pytest.mark.parametrize("dtype", ["int32", "float32"])
def test_add(cuda_function, numbast_jit, dtype):
    add = cuda_function["add"]

    @numbast_jit
    def kernel(arr):
        arr[0] = add(1, 2)

    arr = device_array((1,), dtype)
    kernel[1, 1](arr)
    assert arr.copy_to_host()[0] == 3