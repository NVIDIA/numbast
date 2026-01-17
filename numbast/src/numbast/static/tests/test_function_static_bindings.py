# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import pytest
import numpy as np
import cffi

from numba.cuda.types import int32, float32
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


@pytest.fixture(scope="function")
def decl_out(make_binding):
    intents = {
        "add_out": {"out": "out_return"},
        "add_out_ret": {"out": "out_return"},
    }
    res = make_binding("function_out.cuh", {}, {}, "sm_50", intents)
    bindings = res["bindings"]

    public_apis = ["add_out", "add_out_ret"]
    assert all(public_api in bindings for public_api in public_apis)

    return bindings


@pytest.fixture(scope="function")
def decl_out_ptr(make_binding):
    intents = {
        "add_out": {"out": "out_ptr"},
        "add_in_ref": {"x": "in"},
        "add_inout_ref": {"x": "inout_ptr"},
    }
    res = make_binding("function_out.cuh", {}, {}, "sm_50", intents)
    bindings = res["bindings"]

    public_apis = ["add_out", "add_in_ref", "add_inout_ref"]
    assert all(public_api in bindings for public_api in public_apis)

    return bindings


@pytest.fixture(scope="module")
def impl_out(data_folder):
    return data_folder("src", "function_out.cu")


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


def test_out_return_function_bindings(decl_out, impl_out):
    add_out = decl_out["add_out"]
    add_out_ret = decl_out["add_out_ret"]

    @cuda.jit(link=[impl_out])
    def kernel(out_single, out_pair):
        out_single[0] = add_out(10)
        ret, out = add_out_ret(7)
        out_pair[0] = ret
        out_pair[1] = out

    out_single = device_array((1,), "int32")
    out_pair = device_array((2,), "int32")
    kernel[1, 1](out_single, out_pair)

    assert out_single.copy_to_host()[0] == 11
    host_pair = out_pair.copy_to_host()
    assert host_pair[0] == 10
    assert host_pair[1] == 9


def test_out_ptr_in_inout_function_bindings(decl_out_ptr, impl_out):
    ffi = cffi.FFI()
    add_out = decl_out_ptr["add_out"]
    add_in_ref = decl_out_ptr["add_in_ref"]
    add_inout_ref = decl_out_ptr["add_inout_ref"]

    @cuda.jit(link=[impl_out])
    def kernel(out_ptr_buf, in_val, inout_buf, out_val):
        out_ptr = ffi.from_buffer(out_ptr_buf)
        add_out(out_ptr, in_val)
        inout_ptr = ffi.from_buffer(inout_buf)
        add_inout_ref(inout_ptr, 6)
        out_val[0] = add_in_ref(in_val)

    out_ptr_buf = np.zeros(1, dtype=np.int32)
    inout_buf = np.array([4], dtype=np.int32)
    out_val = device_array((1,), "int32")
    kernel[1, 1](out_ptr_buf, int32(8), inout_buf, out_val)

    assert out_ptr_buf[0] == 9
    assert inout_buf[0] == 10
    assert out_val.copy_to_host()[0] == 13
