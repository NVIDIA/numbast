# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
import re

from numba import types as nbtypes
from numba.cuda.types import bfloat16
from numba.cuda.vector_types import vector_types

from numba.cuda._internal.cuda_bf16 import _type_unnamed1405307

from cuda.bindings import runtime


_FIXED_SIZE_ARRAY_RE = re.compile(r"^(?P<base>.+?)(?P<dims>(?:\[\d+\])+)$")
_ARRAY_DIM_RE = re.compile(r"\[(\d+)\]")


class FunctorType(nbtypes.Type):
    def __init__(self, name):
        super().__init__(name=name + "FunctorType")


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
    "unsigned long long": nbtypes.uint,
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

CUDA_VECTOR_TYPE_SPECS = (
    ("char", "int8", {1: 1, 2: 2, 3: 1, 4: 4}),
    ("uchar", "uint8", {1: 1, 2: 2, 3: 1, 4: 4}),
    ("short", "int16", {1: 2, 2: 4, 3: 2, 4: 8}),
    ("ushort", "uint16", {1: 2, 2: 4, 3: 2, 4: 8}),
    ("int", "int32", {1: 4, 2: 8, 3: 4, 4: 16}),
    ("uint", "uint32", {1: 4, 2: 8, 3: 4, 4: 16}),
    ("longlong", "int64", {1: 8, 2: 16, 3: 8, 4: 16}),
    ("ulonglong", "uint64", {1: 8, 2: 16, 3: 8, 4: 16}),
    ("float", "float32", {1: 4, 2: 8, 3: 4, 4: 16}),
    ("double", "float64", {1: 8, 2: 16, 3: 8, 4: 16}),
)

CUDA_VECTOR_TYPE_MAPS = {
    f"{cxx_name}{lanes}": (
        vector_types[f"{numba_name}x{lanes}"],
        alignments[lanes],
    )
    for cxx_name, numba_name, alignments in CUDA_VECTOR_TYPE_SPECS
    for lanes in (1, 2, 3, 4)
}

CTYPE_MAPS = {}
CTYPE_ALIGNOF_MAPS = {}
NUMBA_TYPE_ALIGNOF_MAPS = {}


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
    **{
        numba_type: cxx_name
        for cxx_name, (numba_type, _alignof) in CUDA_VECTOR_TYPE_MAPS.items()
    },
}


def _normalize_alignof(alignof: int | None) -> int | None:
    if alignof is None:
        return None
    alignof = int(alignof)
    if alignof <= 0 or alignof & (alignof - 1):
        raise ValueError("alignof must be a positive power of two")
    return alignof


def register_cxx_type(
    cxx_name: str,
    numba_type: nbtypes.Type,
    *,
    alignof: int | None = None,
):
    """
    Register an external C++ type in numbast's type registry.

    After registration, ``to_numba_type(cxx_name)`` returns *numba_type*
    instead of ``Opaque``. This is useful for binding types that were not
    parsed by ast_canopy (e.g. third-party library types whose layout is
    known but whose headers are too complex to parse).

    Parameters:
        cxx_name (str): The C++ type name as it appears in function signatures.
        numba_type (nbtypes.Type): The Numba type to map it to.
        alignof (int | None): Optional explicit C++ alignment requirement for
            the type. When provided, it must be a positive power of two.
    """
    alignof = _normalize_alignof(alignof)
    existing_type = CTYPE_MAPS.get(cxx_name)
    existing_alignof = CTYPE_ALIGNOF_MAPS.get(cxx_name)
    if alignof is not None:
        existing_type_alignof = NUMBA_TYPE_ALIGNOF_MAPS.get(numba_type)
        if existing_type_alignof is None:
            existing_owner = next(
                (
                    existing_name
                    for existing_name, registered_type in CTYPE_MAPS.items()
                    if (
                        existing_name != cxx_name
                        and registered_type == numba_type
                    )
                ),
                None,
            )
            if existing_owner is not None:
                raise ValueError(
                    "aligned aliases require a distinct Numba type object; "
                    f"{numba_type} is already registered as {existing_owner!r}"
                )
        elif existing_type_alignof != alignof:
            raise ValueError(
                f"{numba_type} already has alignof_={existing_type_alignof}, "
                f"cannot register {cxx_name!r} with alignof={alignof}"
            )
        NUMBA_TYPE_ALIGNOF_MAPS[numba_type] = alignof

    CTYPE_MAPS[cxx_name] = numba_type
    if alignof is None:
        CTYPE_ALIGNOF_MAPS.pop(cxx_name, None)
    else:
        CTYPE_ALIGNOF_MAPS[cxx_name] = alignof

    if (
        existing_type is not None
        and existing_alignof is not None
        and not any(
            registered_type == existing_type
            for existing_name, registered_type in CTYPE_MAPS.items()
            if CTYPE_ALIGNOF_MAPS.get(existing_name) is not None
        )
    ):
        NUMBA_TYPE_ALIGNOF_MAPS.pop(existing_type, None)


def get_numba_type_alignof(numba_type: nbtypes.Type) -> int | None:
    try:
        return NUMBA_TYPE_ALIGNOF_MAPS.get(numba_type)
    except TypeError:
        return None


def _register_builtin_cxx_types():
    for cxx_name, numba_type in {
        **INTEGER_TYPE_MAPS,
        **FLOATING_TYPE_MAPS,
        **CCCL_Types,
        **ENUM_TYPE_MAPS,
        "void": nbtypes.void,
        "bool": nbtypes.bool_,
    }.items():
        register_cxx_type(cxx_name, numba_type)

    for cxx_name, (numba_type, alignof) in CUDA_VECTOR_TYPE_MAPS.items():
        register_cxx_type(cxx_name, numba_type, alignof=alignof)


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
    register_cxx_type(cxx_name, nbtypes.IntEnumMember(e, nbtypes.int64))


def parse_fixed_size_array_type(ty: str) -> tuple[str, list[int]] | None:
    """
    Split a fixed-size C/C++ array type string into its base type and extents.

    Examples:
        ``float[12]`` -> ``("float", [12])``
        ``float[2][12]`` -> ``("float", [2, 12])``
    """
    match = _FIXED_SIZE_ARRAY_RE.match(ty.strip())
    if match is None:
        return None

    base_ty = match.group("base").strip()
    dimensions = [
        int(dim) for dim in _ARRAY_DIM_RE.findall(match.group("dims"))
    ]
    return base_ty, dimensions


def to_numba_type(ty: str):
    """
    Map a C/C++ type string to the corresponding Numba type.

    Parameters:
        ty (str): C/C++ type name as it appears in headers (may include pointers '*', fixed-size array syntax '[N]', functor suffix 'FunctorType', or collapsed array-pointer forms).

    Returns:
        nbty (nbtypes.Type): The mapped Numba type (e.g., scalar types from CTYPE_MAPS, `nbtypes.CPointer(...)` for pointer forms, or `nbtypes.UniTuple(..., N)` for fixed-size arrays).
    """
    if ty == "__nv_bfloat16_raw":
        return _type_unnamed1405307

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
    # doing manual parsing for array type here. C arrays are arrays of arrays,
    # so `T[M][N]` is modeled as M rows of N elements.
    array_type = parse_fixed_size_array_type(ty)
    if array_type is not None:
        base_ty, dimensions = array_type
        nbty = to_numba_type(base_ty)
        for size in reversed(dimensions):
            nbty = nbtypes.UniTuple(nbty, size)
        return nbty

    # For any type that's unknown / not yet supported, return an opaque type.
    return CTYPE_MAPS.get(ty, nbtypes.Opaque(ty))


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


# Register built-in C/CUDA type mappings through the same public path used by
# downstream registrations.
_register_builtin_cxx_types()
