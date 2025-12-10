# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numba.cuda.types as nbtypes
from numbast.registry import ENUM_TYPE_UNDERLYING_INTEGER_TYPE_MAP


def prepare_args(target_context, llvm_builder, sig, args):
    """
    Prepare arguments to be passed across FFI calling convention.

    Currently the calling convention is:
    All arguments are copied onto the stack and passed as pointers across FFI.
    If the argument is an enum, it is converted to the underlying integer type.

    Parameters
    ----------
    target_context : numba.cuda.context.CUDATargetContext
        The target context to prepare arguments for.
    llvm_builder : llvmlite.IRBuilder
        The llvmlite IR builder to use.
    sig : numba.cuda.typing.templates.Signature
        The signature of the function to prepare arguments for.
    args : list
        The actual arguments to prepare.

    Returns
    -------
    ptrs : list
        The pointers to the arguments.
    """

    processed_sigs = []
    for argty in sig.args:
        if isinstance(argty, nbtypes.IntEnumMember):
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
