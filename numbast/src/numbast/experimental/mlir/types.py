# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
import re

from numba_cuda_mlir import types as nbtypes
from numba_cuda_mlir.numba_cuda import types as cuda_nbtypes
from numba_cuda_mlir.numba_cuda.types import bfloat16

from cuda.bindings import runtime
from numba_cuda_mlir.type_defs.vector_types import VectorType
from numba_cuda_mlir.types import float32, float64, uint32
from numbast.experimental.mlir.type_naming import make_unique_type_name


class FunctorType(nbtypes.Type):
    def __init__(self, name):
        super().__init__(name=make_unique_type_name(self, name + "FunctorType"))


INTEGER_TYPE_MAPS = {
    "char": nbtypes.int8,
    "signed char": nbtypes.int8,
    "unsigned char": nbtypes.uint8,
    "short": nbtypes.int16,
    "unsigned short": nbtypes.uint16,
    "int": nbtypes.int32,
    "unsigned int": nbtypes.uint32,
    "long": nbtypes.int64,
    "unsigned long": nbtypes.uint64,
    "long long": nbtypes.int64,
    "unsigned long long": nbtypes.uint64,
    # Begin of stdint types
    "int8_t": nbtypes.int8,
    "uint8_t": nbtypes.uint8,
    "int16_t": nbtypes.int16,
    "uint16_t": nbtypes.uint16,
    "int32_t": nbtypes.int32,
    "uint32_t": nbtypes.uint32,
    "int64_t": nbtypes.int64,
    "uint64_t": nbtypes.uint64,
}

FLOATING_TYPE_MAPS = {
    "__half": nbtypes.float16,
    "__nv_bfloat16": bfloat16,
    "float": nbtypes.float32,
    "double": nbtypes.float64,
}

CCCL_Types = {
    "cub::NullType": nbtypes.void,
}

ENUM_TYPE_MAPS = {
    "cudaRoundMode": nbtypes.IntEnumMember(
        runtime.cudaRoundMode, nbtypes.int64
    ),
}

CTYPE_MAPS = {
    **INTEGER_TYPE_MAPS,
    **FLOATING_TYPE_MAPS,
    **CCCL_Types,
    **ENUM_TYPE_MAPS,
    "void": nbtypes.void,
    "bool": nbtypes.bool,
    "uint4": VectorType(uint32, 4),
    "uint2": VectorType(uint32, 2),
    "float2": VectorType(float32, 2),
    "float4": VectorType(float32, 4),
    "double2": VectorType(float64, 2),
    "double4": VectorType(float64, 4),
}


NUMBA_TO_CTYPE_MAPS = {
    nbtypes.int8: "char",
    nbtypes.uint8: "unsigned char",
    nbtypes.int16: "short",
    nbtypes.uint16: "unsigned short",
    nbtypes.int32: "int",
    nbtypes.uint32: "unsigned int",
    nbtypes.int64: "long long",
    nbtypes.uint64: "unsigned long long",
    nbtypes.float16: "__half",
    nbtypes.float32: "float",
    nbtypes.float64: "double",
    nbtypes.bool: "bool",
    nbtypes.void: "void",
}


def register_enum_type(
    cxx_name: str,
    e: type[Enum],
):
    """
    Register a mapping from a C++ enum type name to its corresponding Numba type.

    Parameters:
        cxx_name (str): The C++ enum type name to register (as it appears in C/C++ headers).
        e (type[Enum]): The Python enum type to register.

    Returns:
        None
    """
    CTYPE_MAPS[cxx_name] = nbtypes.IntEnumMember(e, nbtypes.int64)


def to_numba_type(ty: str):
    """
    Map a C/C++ type string to the corresponding Numba type.

    Parameters:
        ty (str): C/C++ type name as it appears in headers (may include pointers '*', fixed-size array syntax '[N]', functor suffix 'FunctorType', or collapsed array-pointer forms).

    Returns:
        nbty (nbtypes.Type): The mapped Numba type (e.g., scalar types from CTYPE_MAPS, `nbtypes.CPointer(...)` for pointer forms, or `nbtypes.UniTuple(..., N)` for fixed-size arrays).
    """
    if "FunctorType" in ty:
        return FunctorType(ty[:-11])
    if ty.endswith("*"):
        base_ty = ty.rstrip("*").rstrip(" ")
        return nbtypes.CPointer(to_numba_type(base_ty))

    # A pointer to an array type, collapsed as a simple array pointer.
    if "(*)[" in ty:
        base_ty = ty.split(" (")[0]
        return nbtypes.CPointer(to_numba_type(base_ty))

    # Support for array type is still incomplete in ast_canopy,
    # doing manual parsing for array type here.
    arr_type_pat = r"(.*)\[(\d+)\]"
    is_array_type = re.match(arr_type_pat, ty)
    if is_array_type:
        base_ty, size = is_array_type.groups()
        return nbtypes.UniTuple(to_numba_type(base_ty), int(size))

    # For any type that's unknown / not yet supported, return an opaque type.
    # Avoid dict.get(..., Opaque(...)) because the default is evaluated even
    # when `ty` is present, and recent numba-cuda-mlir top-level types no
    # longer export Opaque.
    if ty in CTYPE_MAPS:
        return CTYPE_MAPS[ty]

    opaque_type = getattr(nbtypes, "Opaque", cuda_nbtypes.Opaque)
    return opaque_type(ty)


def to_numba_arg_type(ast_type) -> nbtypes.Type:
    """
    Map an ast_canopy Type to the corresponding Numba type for use as a function argument.

    This conversion uses the type's unqualified, non-reference form. It does not map C++ reference parameters (T& / T&&) to pointer types; reference exposure is controlled by higher-level binding configuration (numbast.intent.ArgIntent).

    Parameters:
        ast_type: The AST type to convert; its unqualified non-reference name will be used.

    Returns:
        nbtypes.Type: The Numba type appropriate for a function argument.
    """
    return to_numba_type(ast_type.unqualified_non_ref_type_name)


def to_c_type_str(nbty: nbtypes.Type) -> str:
    """
    Convert a Numba type to its corresponding C/C++ type string.

    Parameters:
        nbty (nbtypes.Type): The Numba type to convert.

    Returns:
        str: The C/C++ type name that corresponds to `nbty`.

    Raises:
        ValueError: If `nbty` has no known mapping to a C type.
    """
    if isinstance(nbty, nbtypes.CPointer):
        return f"{to_c_type_str(nbty.dtype)}*"
    if nbty not in NUMBA_TO_CTYPE_MAPS:
        raise ValueError(
            f"Unknown numba type attempted to converted into ctype: {nbty}"
        )

    return NUMBA_TO_CTYPE_MAPS[nbty]


def is_c_integral_type(typ_str: str) -> bool:
    return typ_str in INTEGER_TYPE_MAPS


def is_c_floating_type(typ_str: str) -> bool:
    return typ_str in FLOATING_TYPE_MAPS


# Register CUDA Python Types
register_enum_type("cudaRoundMode", runtime.cudaRoundMode)
