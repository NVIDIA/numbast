# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
import copy

from numbast.types import CTYPE_MAPS
from numbast.static.renderer import BaseRenderer
from numbast.errors import TypeNotFoundError


_DEFAULT_CTYPE_TO_NBTYPE_STR_MAP = {
    k: str(v) for k, v in CTYPE_MAPS.items()
} | {
    "bool": "bool_",
    "void": "void",
    "cudaRoundMode": "IntEnumMember(cudaRoundMode, int64)",
}

CTYPE_TO_NBTYPE_STR = copy.deepcopy(_DEFAULT_CTYPE_TO_NBTYPE_STR_MAP)

_VALID_ENUM_UNDERLYING_INTEGER_TYPES = {
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
}


def register_enum_type_str(
    ctype_enum_name: str,
    enum_name: str,
    underlying_numba_int_type: str = "int64",
):
    """
    Register a mapping from a C++ enum type name to its corresponding Numba type string.

    Parameters:
        ctype_enum_name (str): The C++ enum type name to register (as it appears in C/C++ headers).
        enum_name (str): The enum identifier to use inside the generated Numba type string (becomes the first argument to `IntEnumMember`).
        underlying_numba_int_type (str): The underlying Numba integer type to use for the enum.
    """
    global CTYPE_TO_NBTYPE_STR

    if underlying_numba_int_type not in _VALID_ENUM_UNDERLYING_INTEGER_TYPES:
        raise ValueError(
            "Invalid enum underlying integer type: "
            f"{underlying_numba_int_type!r}. Expected one of "
            f"{sorted(_VALID_ENUM_UNDERLYING_INTEGER_TYPES)}."
        )

    CTYPE_TO_NBTYPE_STR[ctype_enum_name] = (
        f"IntEnumMember({enum_name}, {underlying_numba_int_type})"
    )


def reset_types():
    global CTYPE_TO_NBTYPE_STR

    CTYPE_TO_NBTYPE_STR.clear()
    CTYPE_TO_NBTYPE_STR.update(_DEFAULT_CTYPE_TO_NBTYPE_STR_MAP)


def to_numba_type_str(ty: str):
    """
    Map a C/C++ type string to its corresponding Numba type string.

    This also records any required Numba/CUDA type imports in BaseRenderer so generated code can import the mapped types.

    Parameters:
        ty (str): C/C++ type name (may include pointers '*' or fixed-size array syntax like 'T[4]').

    Returns:
        str: The corresponding Numba type expression (e.g., "int64", "CPointer(int64)", "UniTuple(float32, 4)").

    Raises:
        TypeNotFoundError: If `ty` has no known mapping to a Numba type.
    """

    if ty == "cudaRoundMode":
        BaseRenderer.Imports.add(
            "from cuda.bindings.runtime import cudaRoundMode"
        )
        BaseRenderer._try_import_numba_type("IntEnumMember")
        return CTYPE_TO_NBTYPE_STR[ty]

    if ty == "__nv_bfloat16":
        BaseRenderer._try_import_numba_type("__nv_bfloat16")
        return "bfloat16"

    if ty == "__nv_bfloat16_raw":
        BaseRenderer._try_import_numba_type("__nv_bfloat16_raw")
        return "bfloat16_raw_type"

    if ty.endswith("*"):
        base_ty = ty.rstrip("*").rstrip(" ")
        ptr_ty_str = f"CPointer({to_numba_type_str(base_ty)})"
        BaseRenderer._try_import_numba_type("CPointer")
        return ptr_ty_str

    # A pointer to an array type, collapsed as a simple array pointer.
    if "(*)[" in ty:
        base_ty = ty.split(" (")[0]
        ptr_ty_str = f"CPointer({to_numba_type_str(base_ty)})"
        BaseRenderer._try_import_numba_type("CPointer")
        return ptr_ty_str

    # Support for array type is still incomplete in ast_canopy,
    # doing manual parsing for array type here.
    arr_type_pat = r"(.*)\[(\d+)\]"
    is_array_type = re.match(arr_type_pat, ty)
    if is_array_type:
        base_ty, size = is_array_type.groups()
        base_ty_str = to_numba_type_str(base_ty)

        arr_type_str = f"UniTuple({base_ty_str}, {int(size)})"
        BaseRenderer._try_import_numba_type("UniTuple")
        return arr_type_str

    try:
        nb_type_str = CTYPE_TO_NBTYPE_STR[ty]
    except KeyError:
        raise TypeNotFoundError(ty)

    BaseRenderer._try_import_numba_type(nb_type_str)
    return nb_type_str


def to_numba_arg_type_str(ast_type) -> str:
    """
    Convert an AST Canopy Type to the corresponding Numba type string for use in function argument typing.

    Parameters:
        ast_type: An AST Canopy Type object; its unqualified, non-reference type name is used to determine the mapped Numba type.

    Returns:
        A string representing the Numba type suitable for argument annotations.

    Note:
        This function does not map C++ reference parameters (T& / T&&) to pointer types. Reference exposure is handled by higher-level binding configuration (e.g., numbast.intent.ArgIntent).
    """
    return to_numba_type_str(ast_type.unqualified_non_ref_type_name)
