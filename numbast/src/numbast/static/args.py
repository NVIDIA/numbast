# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

prepare_args_template = """
def prepare_args(target_context, llvm_builder, sig, args):
    processed_sigs = []
    for argty in sig.args:
        if isinstance(argty, types.IntEnumMember):
            pyenum = argty.instance_class
            pyenum_qualname = pyenum.__qualname__
            argty = ENUM_TYPE_UNDERLYING_INTEGER_TYPE_MAP[pyenum_qualname]

        processed_sigs.append(target_context.get_value_type(argty))

    ptrs = []
    for arg_irty in processed_sigs:
        ptrs.append(llvm_builder.alloca(arg_irty))

    ptrs = [
        llvm_builder.alloca(target_context.get_value_type(arg))
        for arg in sig.args
    ]
    for ptr, ty, arg in zip(ptrs, processed_sigs, args):
        if isinstance(ty, nbtypes.IntEnumMember):
            arg = llvm_builder.trunc(arg, ty)

        llvm_builder.store(arg, ptr, align=getattr(ty, "alignof_", None))

    return ptrs
"""
