# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from types import SimpleNamespace

from llvmlite import ir
from numba.cuda import types as cuda_types
from numba.cuda.descriptor import cuda_target

from numbast.callconv import FunctionCallConv, _get_alloca_alignment
from numbast.intent_defs import ArgIntent, IntentPlan, OutArrayReturnSpec
from numbast.types import CTYPE_MAPS, get_numba_type_alignof


class _ShimWriter:
    def write_to_shim(self, shim_code, shim_function_name):
        pass


def _lower_callconv_to_ir(
    *,
    return_type,
    args=(),
    intent_plan=None,
    out_return_types=None,
    cxx_return_type=None,
):
    context = cuda_target.target_context
    sig = SimpleNamespace(return_type=return_type, args=tuple(args))
    arg_value_types = [context.get_value_type(argty) for argty in sig.args]

    module = ir.Module()
    fn = ir.Function(
        module, ir.FunctionType(ir.VoidType(), arg_value_types), "caller"
    )
    builder = ir.IRBuilder(fn.append_basic_block("entry"))

    callconv = FunctionCallConv(
        "_Z4testv",
        _ShimWriter(),
        "",
        intent_plan=intent_plan,
        out_return_types=out_return_types,
        cxx_return_type=cxx_return_type,
    )
    callconv._lower_impl(builder, context, sig, tuple(fn.args))
    builder.ret_void()
    return str(module)


def test_cuda_vector_alloca_alignment_uses_registered_alignof():
    context = cuda_target.target_context
    float2 = CTYPE_MAPS["float2"]
    float4 = CTYPE_MAPS["float4"]

    assert get_numba_type_alignof(float2) == 8
    assert get_numba_type_alignof(float4) == 16

    assert (
        _get_alloca_alignment(context, context.get_value_type(float2), float2)
        == 8
    )
    assert (
        _get_alloca_alignment(context, context.get_value_type(float4), float4)
        == 16
    )


def test_alloca_alignment_falls_back_to_abi_alignment():
    context = cuda_target.target_context
    no_explicit_align = SimpleNamespace()

    for numba_ty in (CTYPE_MAPS["float2"], CTYPE_MAPS["float4"]):
        value_ty = context.get_value_type(numba_ty)
        assert _get_alloca_alignment(
            context, value_ty, no_explicit_align
        ) == context.get_abi_alignment(value_ty)


def test_alloca_alignment_honors_explicit_alignof_larger_than_16():
    context = cuda_target.target_context
    value_ty = context.get_value_type(cuda_types.int32)
    explicit_aligned_type = SimpleNamespace(alignof_=32)

    assert _get_alloca_alignment(context, value_ty, explicit_aligned_type) == 32


def test_return_and_value_arg_allocas_are_aligned_in_lowered_ir():
    float2 = CTYPE_MAPS["float2"]

    llvm_ir = _lower_callconv_to_ir(return_type=float2, args=(float2,))

    assert llvm_ir.count("alloca {float, float}, align 8") == 2
    assert re.search(r"store \{float, float\} .*, align 8", llvm_ir)
    assert re.search(r"load \{float, float\}, .* align 8", llvm_ir)


def test_intent_plan_allocas_are_aligned_in_lowered_ir():
    float2 = CTYPE_MAPS["float2"]
    float4 = CTYPE_MAPS["float4"]
    plan = IntentPlan(
        intents=(ArgIntent.in_, ArgIntent.out_return),
        visible_param_indices=(0,),
        out_return_indices=(1,),
        pass_ptr_mask=(False,),
    )

    llvm_ir = _lower_callconv_to_ir(
        return_type=float4,
        args=(float2,),
        intent_plan=plan,
        out_return_types=(float4,),
        cxx_return_type=cuda_types.void,
    )

    assert "alloca {float, float}, align 8" in llvm_ir
    assert "alloca {float, float, float, float}, align 16" in llvm_ir
    assert re.search(
        r"load \{float, float, float, float\}, .* align 16", llvm_ir
    )


def test_out_array_return_allocates_stack_storage_and_returns_unituple():
    plan = IntentPlan(
        intents=(ArgIntent.out_array_return,),
        visible_param_indices=(),
        out_return_indices=(0,),
        pass_ptr_mask=(),
        out_array_return_specs=(
            OutArrayReturnSpec(
                dtype=cuda_types.float32,
                length=12,
                shim_arg_indirect=True,
            ),
        ),
    )
    return_type = cuda_types.UniTuple(cuda_types.float32, 12)

    llvm_ir = _lower_callconv_to_ir(
        return_type=return_type,
        args=(),
        intent_plan=plan,
        out_return_types=(return_type,),
        cxx_return_type=cuda_types.void,
    )

    assert re.search(r"alloca float, i64 12, align 4", llvm_ir)
    assert "float**" in llvm_ir
    assert llvm_ir.count("load float, float*") >= 12


def test_out_array_return_honors_vector_element_alignment():
    float4 = CTYPE_MAPS["float4"]
    plan = IntentPlan(
        intents=(ArgIntent.out_array_return,),
        visible_param_indices=(),
        out_return_indices=(0,),
        pass_ptr_mask=(),
        out_array_return_specs=(
            OutArrayReturnSpec(
                dtype=float4,
                length=3,
                shim_arg_indirect=True,
            ),
        ),
    )
    return_type = cuda_types.UniTuple(float4, 3)

    llvm_ir = _lower_callconv_to_ir(
        return_type=return_type,
        args=(),
        intent_plan=plan,
        out_return_types=(return_type,),
        cxx_return_type=cuda_types.void,
    )

    assert "alloca {float, float, float, float}, i64 3, align 16" in llvm_ir
    assert re.search(
        r"load \{float, float, float, float\}, .* align 16", llvm_ir
    )
