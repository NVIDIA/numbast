# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from numba_cuda_mlir.mlir_lowering import MLIRLower

from numba_cuda_mlir._mlir import ir
from numba_cuda_mlir._mlir.dialects import llvm


def prepare_ir_types(
    builder: MLIRLower,
    argtys: list[ir.Type],
    *,
    pass_ptr_mask: list[bool] | None = None,
) -> list[ir.Type]:
    """
    Prepare LLVM IR types for passing function arguments by reference.

    Given a list of argument IR types, return a parallel list of IR types suitable for an ABI that passes arguments by pointer. For each argument, the builder's get_mlir_type() is used to obtain the value type; if the corresponding entry in pass_ptr_mask is True and that value type is already an llvm.PointerType, that pointer type is preserved, otherwise the value type is wrapped in an opaque LLVM pointer type.

    Parameters:
        builder (MLIRLower): Lowering helper used to obtain the value type via get_mlir_type().
        argtys (list[ir.Type]): Argument IR types to prepare.
        pass_ptr_mask (list[bool] | None): Optional mask the same length as argtys indicating per-argument behavior.
            If None, all entries are treated as False. When True for an argument and the value type is an llvm.PointerType,
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
        vty = builder.get_mlir_type(argty)
        if passthrough and isinstance(vty, llvm.PointerType):
            # Pass pointer-typed values directly (e.g. C++ T& mapped to CPointer(T))
            ir_types.append(vty)
        else:
            # Default ABI: pass pointer-to-value
            ir_types.append(ir.Type.parse("!llvm.ptr", context=vty.context))

    return ir_types
