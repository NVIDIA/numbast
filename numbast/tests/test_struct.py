# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

from numba import types
from numba.core.datamodel import StructModel

from llvmlite import ir

from ast_canopy import parse_declarations_from_source
from numbast import bind_cxx_struct, MemoryShimWriter

from numba.cuda.descriptor import cuda_target

DATA_FOLDER = os.path.join(os.path.dirname(__file__), "data")


def test_struct_binding_has_correct_LLVM_type():
    # This test checks if the bindings of type Foo correctly lowered into
    # LLVM type { i32, i32, i32 }.
    p = os.path.join(DATA_FOLDER, "sample_struct.cuh")
    decls = parse_declarations_from_source(p, [p], "sm_80")
    structs = decls.structs
    shim_writer = MemoryShimWriter(f"#include {p}")
    s = bind_cxx_struct(shim_writer, structs[0], types.Type, StructModel)

    nbty = s._nbtype
    # Get the LLVM type of this front-end type
    target_ctx = cuda_target.target_context
    llvm_ty = target_ctx.get_value_type(nbty)

    assert len(llvm_ty.elements) == 3
    assert all(ty == ir.IntType(32) for ty in llvm_ty.elements)
    assert not llvm_ty._packed
