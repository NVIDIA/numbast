# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from llvmlite import ir
from numba.cuda.vector_types import vector_types

import numbast.types  # noqa: F401 - registers CUDA vector alignments
from numbast.callconv import _alloca_alignment, _set_alloca_alignment


class _TargetData:
    def __init__(self, abi_alignment):
        self._abi_alignment = abi_alignment

    def abi_alignment(self, _llvm_ty):
        return self._abi_alignment


class _Context:
    def __init__(self, abi_alignment):
        self.target_data = _TargetData(abi_alignment)


def _builder():
    module = ir.Module()
    func_ty = ir.FunctionType(ir.VoidType(), ())
    func = ir.Function(module, func_ty, "test_func")
    block = func.append_basic_block("entry")
    return ir.IRBuilder(block)


def test_cuda_vector_type_alignments_are_registered():
    assert vector_types["float32x2"].alignof_ == 8
    assert vector_types["float32x4"].alignof_ == 16
    assert vector_types["uint32x2"].alignof_ == 8
    assert vector_types["uint32x4"].alignof_ == 16


def test_alloca_alignment_uses_explicit_numba_type_alignment():
    llvm_ty = ir.LiteralStructType([ir.FloatType(), ir.FloatType()])
    align = _alloca_alignment(
        _Context(abi_alignment=4), llvm_ty, vector_types["float32x2"]
    )
    assert align == 8


def test_alloca_alignment_falls_back_to_llvm_abi_alignment():
    llvm_ty = ir.IntType(32)
    align = _alloca_alignment(_Context(abi_alignment=4), llvm_ty)
    assert align == 4


def test_set_alloca_alignment_writes_alignment_to_alloca():
    builder = _builder()
    llvm_ty = ir.LiteralStructType([ir.FloatType(), ir.FloatType()])
    ptr = builder.alloca(llvm_ty, name="retval")

    _set_alloca_alignment(
        ptr, _Context(abi_alignment=4), llvm_ty, vector_types["float32x2"]
    )

    assert ptr.align == 8
