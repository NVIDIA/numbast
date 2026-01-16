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
    return _sample_functions[0]


@pytest.fixture
def shim_writer(_sample_functions):
    return _sample_functions[1]


def test_mutative_device_function_persists_values(func_bindings, shim_writer):
    add_one_inplace = next(
        f
        for f in func_bindings
        if getattr(f, "__name__", None) == "add_one_inplace"
    )
    set_42 = next(
        f for f in func_bindings if getattr(f, "__name__", None) == "set_42"
    )

    ffi = cffi.FFI()

    @cuda.jit(link=shim_writer.links())
    def kernel(out):
        # C++ int& is exposed as a pointer (CPointer(int32)) in Numba typing.
        # Passing a 1-element array provides a stable addressable location.
        p = ffi.from_buffer(out)
        set_42(p)
        add_one_inplace(p)
        add_one_inplace(p)

    out = np.array([0], dtype=np.int32)
    kernel[1, 1](out)
    assert out[0] == 44


@pytest.fixture
def _sample_out_functions():
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
    func_bindings, shim_writer = _sample_out_functions
    add_out = next(
        f for f in func_bindings if getattr(f, "__name__", None) == "add_out"
    )
    add_out_ret = next(
        f
        for f in func_bindings
        if getattr(f, "__name__", None) == "add_out_ret"
    )

    @cuda.jit(link=shim_writer.links())
    def kernel(out_single, out_pair):
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
