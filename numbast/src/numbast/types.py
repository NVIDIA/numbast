# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum
import re

from numba import types as nbtypes
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
    "void": nbtypes.void,
    "__half": nbtypes.float16,
    "float": nbtypes.float32,
    "double": nbtypes.float64,
    "bool": nbtypes.bool_,
    "uint4": nbtypes.Type("uintx4"),
    "uint2": nbtypes.Type("uintx2"),
    "float2": nbtypes.Type("float32x2"),
    "float4": nbtypes.Type("float32x4"),
    "double2": nbtypes.Type("float64x2"),
    "double4": nbtypes.Type("float64x4"),
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
    pat = r"(.*)\[(\d+)\]"
    match = re.match(pat, ty)
    if match:
        base_ty, size = match.groups()
        return nbtypes.UniTuple(to_numba_type(base_ty), int(size))

    return CTYPE_MAPS[ty]
