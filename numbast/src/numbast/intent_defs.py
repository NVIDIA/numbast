# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


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
    """

    in_ = "in"
    inout_ptr = "inout_ptr"
    out_ptr = "out_ptr"
    out_return = "out_return"


@dataclass(frozen=True)
class IntentPlan:
    """
    Normalized intent plan for a callable with N original parameters.
    """

    intents: tuple[ArgIntent, ...]  # length N
    visible_param_indices: tuple[int, ...]  # subset of [0..N)
    out_return_indices: tuple[int, ...]  # subset of [0..N)
    pass_ptr_mask: tuple[bool, ...]  # aligned with visible params only
