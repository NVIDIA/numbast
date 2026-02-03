# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np

from numba import cuda

import cffi

from ast_canopy import parse_declarations_from_source
from numbast import bind_cxx_functions, MemoryShimWriter

import pytest


@pytest.fixture
def _sample_functions():
    """
    Prepare function bindings and a MemoryShimWriter for the mutative sample C++ functions.

    Parses the sample_function_mutative.cuh header from the package data directory, constructs a MemoryShimWriter that includes that header, and binds the parsed C++ functions with argument intent mappings for mutative behavior: both `add_one_inplace` and `set_42` have their `x` parameter treated as an inout pointer.

    Returns:
        tuple: A pair `(func_bindings, shim_writer)` where `func_bindings` is the list of bound function objects and `shim_writer` is the configured MemoryShimWriter.
    """
    DATA_FOLDER = os.path.join(os.path.dirname(__file__), "data")
    p = os.path.join(DATA_FOLDER, "sample_function_mutative.cuh")
    decls = parse_declarations_from_source(p, [p], "sm_80", verbose=True)
    funcs = decls.functions
    shim_writer = MemoryShimWriter(f'#include "{p}"')

    func_bindings = bind_cxx_functions(
        shim_writer,
        funcs,
        arg_intent={
            "add_one_inplace": {"x": "inout_ptr"},
            "set_42": {"x": "inout_ptr"},
        },
    )

    return func_bindings, shim_writer


@pytest.fixture
def func_bindings(_sample_functions):
    """
    Retrieve the function bindings from the sample functions fixture.

    Parameters:
        _sample_functions (tuple): A two-element tuple where the first element is the collection of bound function objects and the second is a MemoryShimWriter.

    Returns:
        func_bindings: The first element of `_sample_functions`, i.e., the bound function objects.
    """
    return _sample_functions[0]


@pytest.fixture
def shim_writer(_sample_functions):
    """
    Provide the MemoryShimWriter produced by the _sample_functions fixture.

    Parameters:
        _sample_functions (tuple): Pair (func_bindings, shim_writer) returned by the `_sample_functions` fixture.

    Returns:
        MemoryShimWriter: The shim writer instance extracted from the fixture.
    """
    return _sample_functions[1]


def find_binding(bindings, name):
    """
    Find a binding object in `bindings` whose `__name__` attribute matches `name`.

    Parameters:
        bindings (Iterable): Iterable of binding objects to search; each may have a `__name__` attribute.
        name (str): The target name to match against each binding's `__name__`.

    Returns:
        object: The first binding whose `__name__` equals `name`.

    Raises:
        AssertionError: If no binding with the given `name` is found; the error message lists the `__name__` values inspected.
    """
    for f in bindings:
        if getattr(f, "__name__", None) == name:
            return f
    raise AssertionError(
        f"Binding '{name}' not found in {[getattr(f, '__name__', None) for f in bindings]}"
    )


def test_mutative_device_function_persists_values(func_bindings, shim_writer):
    """
    Verify that device functions which mutate an integer pointer persist their changes when called from a Numba CUDA kernel.

    This test binds the C++ functions `set_42` and `add_one_inplace`, launches a CUDA kernel that obtains a C pointer to a one-element int32 buffer, calls `set_42` once and `add_one_inplace` twice, and asserts the final buffer value is 44.
    """
    add_one_inplace = find_binding(func_bindings, "add_one_inplace")
    set_42 = find_binding(func_bindings, "set_42")

    ffi = cffi.FFI()

    @cuda.jit(link=shim_writer.links())
    def kernel(out):
        # C++ int& is exposed as a pointer (CPointer(int32)) in Numba typing.
        # Passing a 1-element array provides a stable addressable location.
        """
        CUDA kernel that stores the value 44 into a one-element output buffer by invoking C++ shim functions.

        Parameters:
            out (numpy.ndarray): A single-element int32 buffer whose element will be written with the result (44).
        """
        p = ffi.from_buffer(out)
        set_42(p)
        add_one_inplace(p)
        add_one_inplace(p)

    out = np.array([0], dtype=np.int32)
    kernel[1, 1](out)
    assert out[0] == 44


@pytest.fixture
def _sample_out_functions():
    """
    Create C++ function bindings and a MemoryShimWriter for the sample_function_out test header.

    Parses declarations from the sample_function_out.cuh test header, constructs a MemoryShimWriter that includes that header, and binds the parsed C++ functions with argument intent mappings that treat each function's `out` parameter as an out-return.

    Returns:
        tuple: A pair (func_bindings, shim_writer) where `func_bindings` is the list of bound functions and `shim_writer` is the MemoryShimWriter configured for the sample header.
    """
    DATA_FOLDER = os.path.join(os.path.dirname(__file__), "data")
    p = os.path.join(DATA_FOLDER, "sample_function_out.cuh")
    decls = parse_declarations_from_source(p, [p], "sm_80", verbose=True)
    funcs = decls.functions
    shim_writer = MemoryShimWriter(f'#include "{p}"')

    func_bindings = bind_cxx_functions(
        shim_writer,
        funcs,
        arg_intent={
            "add_out": {"out": "out_return"},
            "add_out_ret": {"out": "out_return"},
        },
    )

    return func_bindings, shim_writer


def test_out_return_device_function_results(_sample_out_functions):
    """
    End-to-end test verifying out-return and pointer/out semantics for C++ functions bound into Numba CUDA kernels.

    First, binds `add_out` and `add_out_ret` and launches a CUDA kernel that:
    - stores `add_out(10)` into `out_single[0]`
    - calls `add_out_ret(7)` and stores the returned pair into `out_pair`

    Asserts that:
    - `out_single[0] == 11`
    - `out_pair[0] == 10`
    - `out_pair[1] == 9`

    Then re-parses and re-binds the same header with different argument intents (`add_out` using an output pointer, `add_in_ref` as an input),
    creates a kernel that uses `ffi.from_buffer` to obtain a pointer for the out parameter, and asserts that:
    - `out_ptr_buf[0] == 9`
    - `out_in_ref[0] == 13`
    """
    func_bindings, shim_writer = _sample_out_functions
    add_out = find_binding(func_bindings, "add_out")
    add_out_ret = find_binding(func_bindings, "add_out_ret")

    @cuda.jit(link=shim_writer.links())
    def kernel(out_single, out_pair):
        """
        Populate provided output buffers with results produced by bound device functions.

        Parameters:
            out_single: Single-element output buffer; receives the result of calling `add_out(10)`.
            out_pair: Two-element output buffer; receives the `(ret, out)` pair returned by `add_out_ret(7)`.
        """
        out_single[0] = add_out(10)
        ret, out = add_out_ret(7)
        out_pair[0] = ret
        out_pair[1] = out

    out_single = np.array([0], dtype=np.int32)
    out_pair = np.array([0, 0], dtype=np.int32)
    kernel[1, 1](out_single, out_pair)
    assert out_single[0] == 11
    assert out_pair[0] == 10
    assert out_pair[1] == 9

    DATA_FOLDER = os.path.join(os.path.dirname(__file__), "data")
    p = os.path.join(DATA_FOLDER, "sample_function_out.cuh")
    decls = parse_declarations_from_source(p, [p], "sm_80", verbose=True)
    funcs = decls.functions
    shim_writer_ptr = MemoryShimWriter(f'#include "{p}"')

    func_bindings_ptr = bind_cxx_functions(
        shim_writer_ptr,
        funcs,
        arg_intent={
            "add_out": {"out": "out_ptr"},
            "add_in_ref": {"x": "in"},
        },
    )

    add_out_ptr = find_binding(func_bindings_ptr, "add_out")
    add_in_ref = find_binding(func_bindings_ptr, "add_in_ref")

    ffi = cffi.FFI()

    @cuda.jit(link=shim_writer_ptr.links())
    def kernel_ptr(out_ptr_buf, in_val, out_in_ref):
        """
        Call bound C++ device functions to store a computed value via a pointer and to write a secondary result into a provided output buffer.

        Parameters:
            out_ptr_buf: A buffer object exposing writable memory for a single integer element; converted to a C pointer and passed to the bound `add_out_ptr` function.
            in_val: Integer input value provided to the bound functions.
            out_in_ref: A writable sequence (e.g., one-element array) whose first element will be set to the value returned by the bound `add_in_ref` function.
        """
        out_ptr = ffi.from_buffer(out_ptr_buf)
        add_out_ptr(out_ptr, in_val)
        out_in_ref[0] = add_in_ref(in_val)

    out_ptr_buf = np.array([0], dtype=np.int32)
    out_in_ref = np.array([0], dtype=np.int32)
    kernel_ptr[1, 1](out_ptr_buf, np.int32(8), out_in_ref)
    assert out_ptr_buf[0] == 9
    assert out_in_ref[0] == 13
