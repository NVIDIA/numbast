# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
import copy

from numbast.types import CTYPE_MAPS
from numbast.static.renderer import BaseRenderer
from numbast.errors import TypeNotFoundError


_DEFAULT_CTYPE_TO_NBTYPE_STR_MAP = {
    k: str(v) for k, v in CTYPE_MAPS.items()
} | {"bool": "bool_", "void": "void"}

CTYPE_TO_NBTYPE_STR = copy.deepcopy(_DEFAULT_CTYPE_TO_NBTYPE_STR_MAP)


def register_enum_type_str(ctype_enum_name: str, enum_name: str):
    """
    Register a mapping from a C++ enum type name to its corresponding Numba type string.

    Parameters:
        ctype_enum_name (str): The C++ enum type name to register (as it appears in C/C++ headers).
        enum_name (str): The enum identifier to use inside the generated Numba type string (becomes the first argument to `IntEnumMember`).
    """
    global CTYPE_TO_NBTYPE_STR

    CTYPE_TO_NBTYPE_STR[ctype_enum_name] = f"IntEnumMember({enum_name}, int64)"


def reset_types():
    global CTYPE_TO_NBTYPE_STR

    CTYPE_TO_NBTYPE_STR.clear()
    CTYPE_TO_NBTYPE_STR.update(_DEFAULT_CTYPE_TO_NBTYPE_STR_MAP)


def to_numba_type_str(ty: str):
    """Converts C type string into numba type string.

    This function closely mirrors that in `numbast.types.to_numba_type`.
    In addition to conversion, this function also adds the corresponding
    type import lines to Numba for the converted types to the renderer's
    cache for import statements.

    Parameter
    ---------
    ty: str
        A string representing a C type

    Return
    ------
    numba_ty: str
        The corresponding string representing a Numba type
    """

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
