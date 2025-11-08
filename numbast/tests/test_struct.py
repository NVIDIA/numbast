# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np

from numba import types, cuda, float32
from numba.core.datamodel import StructModel

from llvmlite import ir

from ast_canopy import parse_declarations_from_source
from numbast import bind_cxx_structs, MemoryShimWriter

from numba.cuda.descriptor import cuda_target

import pytest


@pytest.fixture
def _sample_structs():
    DATA_FOLDER = os.path.join(os.path.dirname(__file__), "data")
    p = os.path.join(DATA_FOLDER, "sample_struct.cuh")
    decls = parse_declarations_from_source(p, [p], "sm_80", verbose=True)
    structs = decls.structs
    shim_writer = MemoryShimWriter(f'#include "{p}"')

    parent_types = {"Foo": types.Type}
    datamodels = {"Foo": StructModel}

    return bind_cxx_structs(
        shim_writer, structs, parent_types, datamodels
    ), shim_writer


@pytest.fixture
def sample_structs(_sample_structs):
    return _sample_structs[0]


@pytest.fixture
def shim_writer(_sample_structs):
    return _sample_structs[1]


def test_struct_binding_has_correct_LLVM_type(sample_structs):
    # This test checks if the bindings of type Foo correctly lowered into
    # LLVM type { i32, i32, i32 }.
    s = sample_structs[0]

    nbty = s._nbtype
    # Get the LLVM type of this front-end type
    target_ctx = cuda_target.target_context
    llvm_ty = target_ctx.get_value_type(nbty)

    assert len(llvm_ty.elements) == 3
    assert all(ty == ir.IntType(32) for ty in llvm_ty.elements)
    assert not llvm_ty._packed


def test_struct_methods_simple(sample_structs, shim_writer):
    Foo = sample_structs[0]

    @cuda.jit(link=shim_writer.links())
    def kernel(arr):
        foo = Foo()
        arr[0] = foo.get_x()

    arr = np.zeros(1, dtype="int32")
    kernel[1, 1](arr)

    assert arr == [42]


def test_struct_methods_argument(sample_structs, shim_writer):
    Foo = sample_structs[0]

    @cuda.jit(link=shim_writer.links())
    def kernel(arr):
        foo = Foo()
        arr[0] = foo.add_one(float32(42.0))

    arr = np.zeros(1, dtype="float32")
    kernel[1, 1](arr)

    assert arr == [43]
