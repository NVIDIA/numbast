# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from numba.cuda.target import CUDATargetContext
from llvmlite import ir


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
