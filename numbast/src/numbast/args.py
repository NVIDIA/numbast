# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from llvmlite import ir


def prepare_ir_types(context, argtys: list[ir.Type]) -> list[ir.Type]:
    """
    Prepare the IR types for the signature.
    """
    return [ir.PointerType(context.get_value_type(argty)) for argty in argtys]
