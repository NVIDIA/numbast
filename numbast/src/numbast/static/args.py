# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

prepare_args_template = """
def prepare_args(target_context, llvm_builder, sig, args, ignore_first=False):

    argtys = sig.args[1:] if ignore_first else sig.args
    args = args[1:] if ignore_first else args

    processed_argtys = []
    for argty in argtys:
        if isinstance(argty, types.IntEnumMember):
            pyenum = argty.instance_class
            pyenum_qualname = pyenum.__qualname__
            underlying_integer_type = EUITSR[pyenum_qualname]
            int_enum_type = types.IntEnumMember(argty.instance_class, underlying_integer_type)
            processed_argtys.append(int_enum_type)
        else:
            processed_argtys.append(argty)

    processed_args = []
    for argty,arg in zip(processed_argtys, args):
        if isinstance(argty, types.IntEnumMember):
            processed_args.append(llvm_builder.trunc(arg, target_context.get_value_type(argty.dtype)))
        else:
            processed_args.append(arg)

    ptrs = [
        llvm_builder.alloca(target_context.get_value_type(argty))
        for argty in processed_argtys
    ]
    for ptr, argty, arg in zip(ptrs, processed_argtys, processed_args):
        llvm_builder.store(arg, ptr, align=getattr(argty, "alignof_", None))

    return ptrs
"""
