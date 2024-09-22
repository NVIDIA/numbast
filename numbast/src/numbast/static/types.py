# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from numba import types as nbtypes

from numbast.types import CTYPE_MAPS

CTYPE_TO_NBTYPE_STR = {k: str(v) for k, v in CTYPE_MAPS.items()}


def register_enum_type(cxx_name: str, e_str: str):
    global CTYPE_TO_NBTYPE_STR

    CTYPE_TO_NBTYPE_STR[cxx_name] = e_str


def to_numba_type_str(ty: str):
    if ty.endswith("*"):
        base_ty = ty.rstrip("*").rstrip(" ")
        return nbtypes.CPointer(to_numba_type_str(base_ty))

    return CTYPE_TO_NBTYPE_STR[ty]
