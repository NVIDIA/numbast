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
    """
    Return the path to the CUDA source implementation file for the `function` tests.
    
    Parameters:
        data_folder (callable): Function that accepts path components and returns a filesystem path (e.g., data_folder("src", "file")).
    
    Returns:
        str: Filesystem path to "src/function.cu".
    """
    return data_folder("src", "function.cu")


@pytest.fixture(scope="function")
def decl_out(make_binding):
    """
    Create bindings for functions in `function_out.cuh` that use out-parameter return semantics and return the resulting bindings mapping.
    
    Parameters:
        make_binding (callable): Factory that builds bindings from a header and options; expected signature accepts
            (header_path, signatures, usages, target, intents) and returns a dict containing a "bindings" mapping.
    
    Returns:
        dict: Mapping of public API names to their binding objects. Contains at least the "add_out" and "add_out_ret" bindings.
    """
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
    """
    Builds bindings for functions that use pointer-style out and inout parameters and returns the generated bindings.
    
    Parameters:
        make_binding (callable): Factory function that generates bindings from a header and options. Expected signature like
            make_binding(header_path, signatures, usages, target, intents) and returns a mapping containing a "bindings" key.
    
    Returns:
        dict: The generated bindings mapping containing the entries "add_out", "add_in_ref", and "add_inout_ref".
    
    Raises:
        AssertionError: If any of the expected public APIs ("add_out", "add_in_ref", "add_inout_ref") are missing from the bindings.
    """
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
    """
    Return the path to the CUDA source implementation file for function_out.
    
    Parameters:
        data_folder (callable): Callable that takes path components and returns the resolved file path.
    
    Returns:
        str: Path to "src/function_out.cu".
    """
    return data_folder("src", "function_out.cu")


@pytest.mark.parametrize("dtype", ["int32", "float32"])
def test_same_argument_types_and_overload(decl, impl, dtype):
    """
    Verify that invoking the bound `add` function with two identical argument types yields the expected numeric result for the specified dtype.
    
    Asserts that a CUDA kernel calling `add(1, 2)` stores the value `3` into a device array.
    
    Parameters:
        dtype (str): Name of the device array data type to test (e.g., "int32" or "float32").
    """
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
        """
        CUDA kernel that invokes `add_out` and `add_out_ret` to populate provided output buffers.
        
        out_single is a 1-element output buffer that will be assigned the result of `add_out(10)`.
        out_pair is a 2-element output buffer where index 0 receives the returned value and index 1 receives the out-parameter value from `add_out_ret(7)`.
        """
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
        """
        CUDA kernel that exercises out, in, and inout bindings by invoking FFI-backed functions to write and mutate buffers and store a computed output.
        
        Parameters:
            out_ptr_buf: Host buffer (e.g., a NumPy array) whose pointer is passed to `add_out` and written by that function.
            in_val: Integer input value passed to the bound functions.
            inout_buf: Host buffer whose pointer is passed to `add_inout_ref` and mutated in place.
            out_val: Device array where the result of `add_in_ref(in_val)` is stored at index 0.
        """
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