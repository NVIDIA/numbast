# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re

from numbast.types import CTYPE_MAPS
from numbast.static.renderer import BaseRenderer


CTYPE_TO_NBTYPE_STR = {k: str(v) for k, v in CTYPE_MAPS.items()}


def register_enum_type_str(enum_name: str):
    global CTYPE_TO_NBTYPE_STR

    CTYPE_TO_NBTYPE_STR[enum_name] = enum_name


def to_numba_type_str(ty: str):
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

    BaseRenderer._try_import_numba_type(CTYPE_TO_NBTYPE_STR[ty])

    return CTYPE_TO_NBTYPE_STR[ty]