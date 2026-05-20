# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import ctypes

import numpy as np

from numba_cuda_mlir import cuda
from numba_cuda_mlir import types

from cuda.pathfinder import find_nvidia_header_directory

from ast_canopy import parse_declarations_from_source
from numbast.experimental.mlir import (
    bind_cxx_enums,
    bind_cxx_functions,
    MemoryShimWriter,
)

import pytest


@pytest.fixture
def _sample_enums():
    DATA_FOLDER = os.path.join(os.path.dirname(__file__), "data")
    p = os.path.join(DATA_FOLDER, "sample_enum.cuh")

    cudart_folder = find_nvidia_header_directory("cudart")
    device_types_h = os.path.join(cudart_folder, "device_types.h")
    decls = parse_declarations_from_source(
        p, [p, device_types_h], "sm_80", verbose=True
    )
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
    eat = [f for f in func_bindings if f.__name__ == "eat"][0]
    Fruit = [e for e in enum_bindings if e.__name__ == "Fruit"][0]

    @cuda.jit(link=shim_writer.links())
    def kernel(out):
        out_ptr = ctypes.cast(types.ptr(out), ctypes.POINTER(ctypes.c_int32))
        first_slot = out_ptr
        second_slot = out_ptr + 1
        third_slot = out_ptr + 2

        eat(Fruit.Apple, first_slot)
        eat(Fruit.Banana, second_slot)
        eat(Fruit.Orange, third_slot)

    out = np.zeros(3, dtype=np.int32)
    kernel[1, 1](out)
    assert np.array_equal(out, [1, 2, 3])


def test_cudaRoundMode_arg(enum_bindings, func_bindings, shim_writer):
    test_cudaRoundMode = [
        f for f in func_bindings if f.__name__ == "test_cudaRoundMode"
    ][0]
    cudaRoundMode = [e for e in enum_bindings if e.__name__ == "cudaRoundMode"][
        0
    ]

    @cuda.jit(link=shim_writer.links())
    def kernel(out):
        out_ptr = ctypes.cast(types.ptr(out), ctypes.POINTER(ctypes.c_int32))
        first_slot = out_ptr
        second_slot = out_ptr + 1
        third_slot = out_ptr + 2
        fourth_slot = out_ptr + 3

        test_cudaRoundMode(cudaRoundMode.cudaRoundNearest, first_slot)
        test_cudaRoundMode(cudaRoundMode.cudaRoundZero, second_slot)
        test_cudaRoundMode(cudaRoundMode.cudaRoundPosInf, third_slot)
        test_cudaRoundMode(cudaRoundMode.cudaRoundMinInf, fourth_slot)

    out = np.zeros(4, dtype=np.int32)
    kernel[1, 1](out)
    assert np.array_equal(out, [1, 2, 3, 4])
