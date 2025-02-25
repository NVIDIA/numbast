# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import pytest
import numpy as np
import cffi

from numba.types import int32, float32
from numba import cuda
from numba.cuda import device_array

from ast_canopy import parse_declarations_from_source

from numbast.static.renderer import clear_base_renderer_cache
from numbast.static.function import StaticFunctionsRenderer


@pytest.fixture(scope="module")
def cuda_function(data_folder):
    clear_base_renderer_cache()

    header = data_folder("function.cuh")

    decls = parse_declarations_from_source(header, [header], "sm_50")
    functions = decls.functions

    assert len(functions) == 4

    SFR = StaticFunctionsRenderer(functions, header)

    bindings = SFR.render_as_str(
        with_prefix=True, with_imports=True, with_shim_functions=True
    )
    globals = {}
    exec(bindings, globals)

    public_apis = ["add", "minus_i32_f32", "set_42"]
    assert all(public_api in globals for public_api in public_apis)

    return {k: globals[k] for k in public_apis}


@pytest.mark.parametrize("dtype", ["int32", "float32"])
def test_same_argument_types_and_overload(cuda_function, dtype):
    add = cuda_function["add"]

    @cuda.jit
    def kernel(arr):
        arr[0] = add(1, 2)

    arr = device_array((1,), dtype)
    kernel[1, 1](arr)
    assert arr.copy_to_host()[0] == 3


def test_different_argument_types(cuda_function):
    minus_i32_f32 = cuda_function["minus_i32_f32"]

    @cuda.jit
    def kernel(arr):
        arr[0] = minus_i32_f32(int32(3), float32(1.4))

    arr = device_array((1,), "int32")
    kernel[1, 1](arr)
    assert arr.copy_to_host()[0] == 2


def test_void_return_type(cuda_function):
    ffi = cffi.FFI()
    set_42 = cuda_function["set_42"]

    @cuda.jit
    def kernel(arr):
        ptr = ffi.from_buffer(arr)
        set_42(ptr)

    arr = np.zeros(1, dtype=np.int32)
    kernel[1, 1](arr)
    assert arr[0] == 42
