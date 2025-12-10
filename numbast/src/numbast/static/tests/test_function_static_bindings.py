# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import pytest
import numpy as np
import cffi

from numba.types import int32, float32
from numba import cuda
from numba.cuda import device_array


@pytest.fixture(scope="function")
def decl(make_binding):
    res = make_binding("function.cuh", {}, {}, "sm_50")
    bindings = res["bindings"]

    public_apis = ["add", "minus_i32_f32", "set_42"]
    assert all(public_api in bindings for public_api in public_apis)

    return bindings


@pytest.fixture(scope="module")
def impl(data_folder):
    return data_folder("src", "function.cu")


@pytest.mark.parametrize("dtype", ["int32", "float32"])
def test_same_argument_types_and_overload(decl, impl, dtype):
    add = decl["add"]

    @cuda.jit(link=[impl])
    def kernel(arr):
        arr[0] = add(1, 2)

    arr = device_array((1,), dtype)
    kernel[1, 1](arr)
    assert arr.copy_to_host()[0] == 3


def test_different_argument_types(decl, impl):
    minus_i32_f32 = decl["minus_i32_f32"]

    @cuda.jit(link=[impl])
    def kernel(arr):
        arr[0] = minus_i32_f32(int32(3), float32(1.4))

    arr = device_array((1,), "int32")
    kernel[1, 1](arr)
    assert arr.copy_to_host()[0] == 2


def test_void_return_type(decl, impl):
    ffi = cffi.FFI()
    set_42 = decl["set_42"]

    @cuda.jit(link=[impl])
    def kernel(arr):
        ptr = ffi.from_buffer(arr)
        set_42(ptr)

    arr = np.zeros(1, dtype=np.int32)
    kernel[1, 1](arr)
    assert arr[0] == 42


def test_float32x2_operator_add_overload(decl, impl):
    @cuda.jit(link=[impl])
    def kernel(arr):
        x = cuda.float32x2(1.0, 1.0) + cuda.float32x2(2.0, 2.0)
        arr[0] = x.x + x.y

    arr = device_array((1,), "float32")
    kernel[1, 1](arr)
    assert arr.copy_to_host()[0] == 6.0
