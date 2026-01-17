# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from numba.cuda.target import CUDATargetContext

from llvmlite import ir


def prepare_ir_types(
    context: CUDATargetContext,
    argtys: list[ir.Type],
    *,
    pass_ptr_mask: list[bool] | None = None,
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
    if pass_ptr_mask is None:
        pass_ptr_mask = [False] * len(argtys)

    if len(pass_ptr_mask) != len(argtys):
        raise ValueError(
            f"pass_ptr_mask length ({len(pass_ptr_mask)}) must match argtys length ({len(argtys)})"
        )

    ir_types: list[ir.Type] = []
    for argty, passthrough in zip(argtys, pass_ptr_mask):
        vty = context.get_value_type(argty)
        if passthrough and isinstance(vty, ir.PointerType):
            # Pass pointer-typed values directly (e.g. C++ T& mapped to CPointer(T))
            ir_types.append(vty)
        else:
            # Default ABI: pass pointer-to-value
            ir_types.append(ir.PointerType(vty))

    return ir_types
