# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

# NBST:BEGIN_RETURN_MATERIALIZATION_DEFS
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class PointerReturnMaterialization:
    """
    Describes a borrowed pointer return copied into a Numba tuple.
    """

    length: int

    def __post_init__(self):
        if type(self.length) is not int:
            raise TypeError(
                "pointer return materialization length must be an int"
            )
        if self.length <= 0:
            raise ValueError(
                "pointer return materialization length must be positive"
            )


def parse_return_materialization(
    raw: Any,
) -> PointerReturnMaterialization | None:
    """
    Normalize user-facing return materialization config.
    """
    if raw is None:
        return None
    if isinstance(raw, PointerReturnMaterialization):
        return raw
    if type(raw) is int:
        return PointerReturnMaterialization(raw)
    if not isinstance(raw, Mapping):
        raise TypeError(
            "return materialization must be an int, mapping, "
            f"PointerReturnMaterialization, or None; got {type(raw)}"
        )

    kind = raw.get("kind", raw.get("intent", "pointer"))
    allowed_kinds = {
        "pointer",
        "ptr",
        "pointer_return",
        "ptr_return",
        "borrowed_ptr",
        "borrowed_pointer",
        "fixed_size_pointer",
        "borrowed_fixed_size_ptr",
        "borrowed_fixed_size_pointer",
    }
    if kind not in allowed_kinds:
        raise ValueError(
            f"unsupported return materialization kind {kind!r}; "
            f"expected one of {sorted(allowed_kinds)}"
        )

    length = raw.get("length", raw.get("size", raw.get("count", None)))
    if length is None:
        raise ValueError(
            "pointer return materialization requires a length, size, or count"
        )
    return PointerReturnMaterialization(length)


# NBST:END_RETURN_MATERIALIZATION_DEFS
