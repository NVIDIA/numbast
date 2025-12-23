# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numba.cuda.types as nbtypes
from numba.cuda.target import CUDATargetContext

from llvmlite import ir

from numbast.registry import enum_underlying_integer_type_registry


def prepare_args(target_context, llvm_builder, sig, args, ignore_first=False):
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
    ignore_first : bool
        Whether to ignore the first argument. Used by class template lowering when
        the first argument is a numba typeof reference that's unused by actual
        lowering.

    Returns
    -------
    ptrs : list
        The pointers to the arguments.
    """

    argtys = sig.args[1:] if ignore_first else sig.args
    args = args[1:] if ignore_first else args

    processed_argtys: list[nbtypes.Type] = []
    for argty in argtys:
        if isinstance(argty, nbtypes.IntEnumMember):
            pyenum = argty.instance_class
            pyenum_qualname = pyenum.__qualname__
            underlying_integer_type = enum_underlying_integer_type_registry[
                pyenum_qualname
            ]
            int_enum_type = nbtypes.IntEnumMember(
                pyenum, underlying_integer_type
            )
            processed_argtys.append(int_enum_type)
        else:
            processed_argtys.append(argty)

    processed_args = []
    for argty, arg in zip(processed_argtys, args):
        if isinstance(argty, nbtypes.IntEnumMember):
            processed_args.append(
                llvm_builder.trunc(
                    arg, target_context.get_value_type(argty.dtype)
                )
            )
        else:
            processed_args.append(arg)

    ptrs = [
        llvm_builder.alloca(target_context.get_value_type(argty))
        for argty in processed_argtys
    ]
    for ptr, argty, arg in zip(ptrs, processed_argtys, processed_args):
        llvm_builder.store(arg, ptr, align=getattr(argty, "alignof_", None))

    return ptrs


def prepare_ir_types(
    context: CUDATargetContext, argtys: list[ir.Type]
) -> list[ir.Type]:
    """
    Prepare IR types for passing arguments via pointers in function calls.

    This utility wraps each argument type in a PointerType to enable
    the call convention used by FunctionCallConv, where arguments are
    passed by reference.

    Parameters
    ----------
    context : context object
        The compilation context providing the get_value_type method.
    argtys : list[ir.Type]
        List of LLVM IR types representing function arguments.

    Returns
    -------
    list[ir.Type]
        List of pointer types wrapping the value types of each argument.
    """
    return [ir.PointerType(context.get_value_type(argty)) for argty in argtys]
