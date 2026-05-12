# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

# NBST:BEGIN_INTENT_DEFS
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any


class ArgIntent(str, Enum):
    """
    Per-parameter intent controlling how a C++ parameter is exposed to Numba.

    Notes
    -----
    - Default is `in` (opt-in): argument is treated as input-only and is exposed
      as a value type on the Numba side.
    - `inout_ptr` / `out_ptr`: C++ reference parameter is exposed as a pointer
      (CPointer(T)) on the Numba side and passed through to the shim.
    - `out_return`: C++ reference parameter is *not* exposed as an argument; a
      temporary is allocated, passed to the shim, and then returned to the caller.
    - `out_array_return`: C++ pointer/array output parameter is *not* exposed as
      an argument; fixed-size stack storage is allocated, passed to the shim, and
      returned to the caller as a UniTuple.
    """

    in_ = "in"
    inout_ptr = "inout_ptr"
    out_ptr = "out_ptr"
    out_return = "out_return"
    out_array_return = "out_array_return"


@dataclass(frozen=True)
class OutArrayReturnSpec:
    """
    Metadata for a fixed-size output array returned as a Numba UniTuple.
    """

    dtype: Any
    length: int
    shim_arg_indirect: bool | None = None

    def with_shim_arg_indirect(self, value: bool) -> "OutArrayReturnSpec":
        return replace(self, shim_arg_indirect=bool(value))


def out_array_return(*, dtype: Any, length: int) -> OutArrayReturnSpec:
    """
    Create an argument-intent spec for fixed-size native output arrays.
    """
    length = int(length)
    if length <= 0:
        raise ValueError("out_array_return length must be positive")
    if dtype is None:
        raise ValueError("out_array_return dtype must be provided")
    return OutArrayReturnSpec(dtype=dtype, length=length)


@dataclass(frozen=True)
class IntentPlan:
    """
    Normalized intent plan for a callable with N original parameters.
    """

    intents: tuple[ArgIntent, ...]  # length N
    visible_param_indices: tuple[int, ...]  # subset of [0..N)
    out_return_indices: tuple[int, ...]  # subset of [0..N)
    pass_ptr_mask: tuple[bool, ...]  # aligned with visible params only
    out_array_return_specs: tuple[OutArrayReturnSpec | None, ...] = ()


# NBST:END_INTENT_DEFS
