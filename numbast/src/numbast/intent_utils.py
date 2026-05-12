# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from numba import types as nbtypes

from numbast.intent import get_out_return_ptr_mask, pointee_type_name
from numbast.intent_defs import ArgIntent, IntentPlan
from numbast.types import to_numba_type


def out_return_type_for_param(param_type, *, pointer_out: bool):
    """
    Return the Numba-visible out-return type for one C++ parameter.
    """
    type_name = param_type.unqualified_non_ref_type_name
    if pointer_out:
        type_name = pointee_type_name(type_name)
    return to_numba_type(type_name)


def out_return_types_for_plan(param_types, plan: IntentPlan):
    ptr_mask = get_out_return_ptr_mask(plan)
    return [
        out_return_type_for_param(param_types[i], pointer_out=ptr_mask[i])
        for i in plan.out_return_indices
    ]


def compose_return_type(cxx_return_type, out_return_types):
    if not out_return_types:
        return cxx_return_type
    if cxx_return_type == nbtypes.void:
        if len(out_return_types) == 1:
            return out_return_types[0]
        return nbtypes.Tuple(tuple(out_return_types))
    return nbtypes.Tuple(tuple([cxx_return_type, *out_return_types]))


def prepend_receiver_to_intent_plan(method_plan: IntentPlan) -> IntentPlan:
    return IntentPlan(
        intents=(ArgIntent.in_,) + method_plan.intents,
        visible_param_indices=(0,)
        + tuple(i + 1 for i in method_plan.visible_param_indices),
        out_return_indices=tuple(i + 1 for i in method_plan.out_return_indices),
        pass_ptr_mask=(False,) + method_plan.pass_ptr_mask,
        out_return_ptr_mask=(False,) + get_out_return_ptr_mask(method_plan),
    )


def shim_arg_type_for_out_return(out_return_type, *, pointer_out: bool):
    if pointer_out:
        return nbtypes.CPointer(out_return_type)
    return out_return_type
