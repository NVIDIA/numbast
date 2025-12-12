# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np

from numba import cuda

import cffi

from ast_canopy import parse_declarations_from_source
from numbast import bind_cxx_enums, bind_cxx_functions, MemoryShimWriter

from cuda.bindings.runtime import cudaRoundMode

import pytest


@pytest.fixture
def _sample_enums():
    DATA_FOLDER = os.path.join(os.path.dirname(__file__), "data")
    p = os.path.join(DATA_FOLDER, "sample_enum.cuh")
    decls = parse_declarations_from_source(p, [p], "sm_80", verbose=True)
    funcs = decls.functions
    enums = decls.enums
    shim_writer = MemoryShimWriter(f'#include "{p}"')

    enum_bindings = bind_cxx_enums(enums)
    func_bindings = bind_cxx_functions(shim_writer, funcs)

    return enum_bindings, func_bindings, shim_writer


@pytest.fixture
def enum_bindings(_sample_enums):
    return _sample_enums[0]


@pytest.fixture
def func_bindings(_sample_enums):
    return _sample_enums[1]


@pytest.fixture
def shim_writer(_sample_enums):
    return _sample_enums[2]


def test_enum_arg(enum_bindings, func_bindings, shim_writer):
    ffi = cffi.FFI()
    eat = func_bindings[0]
    Fruit = enum_bindings[0]

    @cuda.jit(link=shim_writer.links())
    def kernel(out):
        first_slot = ffi.from_buffer(out[0:1])
        second_slot = ffi.from_buffer(out[1:2])
        third_slot = ffi.from_buffer(out[2:3])

        eat(Fruit.Apple, first_slot)
        eat(Fruit.Banana, second_slot)
        eat(Fruit.Orange, third_slot)

    out = np.zeros(3, dtype=np.int32)
    kernel[1, 1](out)
    assert np.array_equal(out, [1, 2, 3])


def test_cudaRoundMode_arg(enum_bindings, func_bindings, shim_writer):
    ffi = cffi.FFI()
    test_cudaRoundMode = func_bindings[1]

    @cuda.jit(link=shim_writer.links())
    def kernel(out):
        first_slot = ffi.from_buffer(out[0:1])
        second_slot = ffi.from_buffer(out[1:2])
        third_slot = ffi.from_buffer(out[2:3])
        fourth_slot = ffi.from_buffer(out[3:4])

        test_cudaRoundMode(cudaRoundMode.cudaRoundNearest, first_slot)
        test_cudaRoundMode(cudaRoundMode.cudaRoundZero, second_slot)
        test_cudaRoundMode(cudaRoundMode.cudaRoundPosInf, third_slot)
        test_cudaRoundMode(cudaRoundMode.cudaRoundMinInf, fourth_slot)

    out = np.zeros(4, dtype=np.int32)
    kernel[1, 1](out)
    assert np.array_equal(out, [1, 2, 3, 4])
