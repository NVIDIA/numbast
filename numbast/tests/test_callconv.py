# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from types import SimpleNamespace

from llvmlite import ir
from numba.cuda import types as cuda_types
from numba.cuda.descriptor import cuda_target

from numbast.callconv import FunctionCallConv, _get_alloca_alignment
from numbast.intent_defs import ArgIntent, IntentPlan
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
    assert getattr(float2, "alignof_", None) is None
    assert getattr(float4, "alignof_", None) is None

    assert (
        _get_alloca_alignment(context, context.get_value_type(float2), float2)
        == 8
    )
    assert (
        _get_alloca_alignment(context, context.get_value_type(float4), float4)
        == 16
    )


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
