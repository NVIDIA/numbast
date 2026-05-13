# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np

from numba import types, cuda, float32
from numba.cuda.datamodel import StructModel

from llvmlite import ir

from ast_canopy import parse_declarations_from_source
from numbast import bind_cxx_structs, MemoryShimWriter
from numbast.static.types import to_numba_type_str
from numbast.types import to_numba_type

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


def test_fixed_size_array_type_preserves_multidimensional_shape():
    assert to_numba_type("unsigned int[20]") == types.UniTuple(types.uint32, 20)
    assert to_numba_type("float[2][12]") == types.UniTuple(
        types.UniTuple(types.float32, 12), 2
    )
    assert to_numba_type_str("float[2][12]") == (
        "UniTuple(UniTuple(float32, 12), 2)"
    )


def _bind_pod_structs_from_source(tmp_path, source):
    p = tmp_path / "pod_structs.cuh"
    p.write_text(source)
    decls = parse_declarations_from_source(str(p), [str(p)], "sm_80")
    structs = decls.structs
    shim_writer = MemoryShimWriter(f'#include "{p}"')
    return bind_cxx_structs(
        shim_writer,
        structs,
        {s.name: types.Type for s in structs},
        {s.name: StructModel for s in structs},
    )


def _get_bound_struct(bindings, name):
    return next(s for s in bindings if s.__name__ == name)


def test_pod_struct_array_fields_have_correct_llvm_shape(tmp_path):
    bindings = _bind_pod_structs_from_source(
        tmp_path,
        """
        struct PodArrayFieldsForNumbast {
          unsigned int data[20];
          float transform[2][12];
          float flat_transform[12];
        };
        """,
    )
    PodArrayFields = _get_bound_struct(bindings, "PodArrayFieldsForNumbast")

    llvm_ty = cuda_target.target_context.get_value_type(PodArrayFields._nbtype)

    assert llvm_ty.elements[0] == ir.ArrayType(ir.IntType(32), 20)
    assert llvm_ty.elements[1] == ir.ArrayType(
        ir.ArrayType(ir.FloatType(), 12), 2
    )
    assert llvm_ty.elements[2] == ir.ArrayType(ir.FloatType(), 12)


def test_pod_struct_nested_fields_and_arrays_have_struct_layout(tmp_path):
    bindings = _bind_pod_structs_from_source(
        tmp_path,
        """
        struct PodNestedOuterForNumbast {
          struct Inner {
            int tag;
            float value;
          };

          Inner inner;
          Inner inners[3];
        };
        """,
    )
    PodNestedOuter = _get_bound_struct(bindings, "PodNestedOuterForNumbast")

    llvm_ty = cuda_target.target_context.get_value_type(PodNestedOuter._nbtype)
    inner_llvm_ty = llvm_ty.elements[0]

    assert len(inner_llvm_ty.elements) == 2
    assert inner_llvm_ty.elements[0] == ir.IntType(32)
    assert inner_llvm_ty.elements[1] == ir.FloatType()
    assert llvm_ty.elements[1] == ir.ArrayType(inner_llvm_ty, 3)
    assert not llvm_ty._packed


def test_pod_struct_nested_duplicate_short_names_use_qualified_names(tmp_path):
    bindings = _bind_pod_structs_from_source(
        tmp_path,
        """
        struct PodNestedDuplicateNamesForNumbast {
          struct Left {
            struct Leaf {
              int value;
            };

            Leaf leaf;
          };

          struct Right {
            struct Leaf {
              float value;
            };

            Leaf leaf;
          };

          Left left;
          Right right;
        };
        """,
    )
    PodNestedDuplicateNames = _get_bound_struct(
        bindings, "PodNestedDuplicateNamesForNumbast"
    )

    llvm_ty = cuda_target.target_context.get_value_type(
        PodNestedDuplicateNames._nbtype
    )
    left_leaf_llvm_ty = llvm_ty.elements[0].elements[0]
    right_leaf_llvm_ty = llvm_ty.elements[1].elements[0]

    assert len(left_leaf_llvm_ty.elements) == 1
    assert left_leaf_llvm_ty.elements[0] == ir.IntType(32)
    assert len(right_leaf_llvm_ty.elements) == 1
    assert right_leaf_llvm_ty.elements[0] == ir.FloatType()


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


def test_struct_methods_void_return(sample_structs, shim_writer):
    Foo = sample_structs[0]

    @cuda.jit(link=shim_writer.links())
    def kernel():
        foo = Foo()
        foo.print()

    kernel[1, 1]()
