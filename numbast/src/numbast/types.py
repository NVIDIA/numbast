# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum
import re

from numba import types as nbtypes
from numba.cuda.vector_types import vector_types
from numba.misc.special import typeof


class FunctorType(nbtypes.Type):
    def __init__(self, name):
        super().__init__(name=name + "FunctorType")


CTYPE_MAPS = {
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
    # End of stdint types
    "void": nbtypes.void,
    "__half": nbtypes.float16,
    "float": nbtypes.float32,
    "double": nbtypes.float64,
    "bool": nbtypes.bool_,
    "uint4": vector_types["uint32x4"],
    "uint2": vector_types["uint32x2"],
    "float2": vector_types["float32x2"],
    "float4": vector_types["float32x4"],
    "double2": vector_types["float64x2"],
    "double4": vector_types["float64x4"],
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
    nbtypes.bool_: "bool",
    nbtypes.void: "void",
}


def register_enum_type(cxx_name: str, e: IntEnum):
    global CTYPE_MAPS

    CTYPE_MAPS[cxx_name] = typeof(e)


def to_numba_type(ty: str):
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

    return CTYPE_MAPS[ty]
