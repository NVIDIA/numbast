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
    Prepare LLVM IR types for passing function arguments by reference.
    
    Given a list of argument IR types, return a parallel list of IR types suitable for an ABI that passes arguments by pointer. For each argument, the context's get_value_type() is used to obtain the value type; if the corresponding entry in pass_ptr_mask is True and that value type is already an ir.PointerType, that pointer type is preserved, otherwise the value type is wrapped in an ir.PointerType.
    
    Parameters:
        context (CUDATargetContext): Compilation context used to obtain the value type via get_value_type().
        argtys (list[ir.Type]): Argument IR types to prepare.
        pass_ptr_mask (list[bool] | None): Optional mask the same length as argtys indicating per-argument behavior.
            If None, all entries are treated as False. When True for an argument and the value type is an ir.PointerType,
            the pointer type is passed through unchanged.
    
    Returns:
        list[ir.Type]: Prepared IR types where each entry is either a pointer-to-value or an existing pointer type preserved per pass_ptr_mask.
    
    Raises:
        ValueError: If pass_ptr_mask is provided and its length does not match len(argtys).
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