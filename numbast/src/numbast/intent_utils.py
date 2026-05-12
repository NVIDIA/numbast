# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace

from numba import types as nbtypes

from numbast.intent_defs import ArgIntent, IntentPlan, OutArrayReturnSpec
from numbast.types import to_numba_type


def get_out_array_return_specs(
    plan: IntentPlan,
) -> tuple[OutArrayReturnSpec | None, ...]:
    specs = getattr(plan, "out_array_return_specs", ())
    if not specs:
        return (None,) * len(plan.intents)
    if len(specs) != len(plan.intents):
        raise ValueError(
            "IntentPlan out_array_return_specs length does not match intents: "
            f"{len(specs)} != {len(plan.intents)}"
        )
    return tuple(specs)


def resolve_out_array_dtype(dtype) -> nbtypes.Type:
    if isinstance(dtype, nbtypes.Type):
        return dtype
    if isinstance(dtype, str):
        resolved = to_numba_type(dtype)
        if not isinstance(resolved, nbtypes.Opaque):
            return resolved
    raise ValueError(f"Unknown out_array_return dtype: {dtype!r}")


def out_return_type_for_param(param_type, spec: OutArrayReturnSpec | None):
    if spec is None:
        return to_numba_type(param_type.unqualified_non_ref_type_name)
    return nbtypes.UniTuple(resolve_out_array_dtype(spec.dtype), spec.length)


def out_return_types_for_plan(param_types, plan: IntentPlan):
    specs = get_out_array_return_specs(plan)
    return [
        out_return_type_for_param(param_types[i], specs[i])
        for i in plan.out_return_indices
    ]


def normalize_out_array_return_specs(plan: IntentPlan) -> IntentPlan:
    specs = get_out_array_return_specs(plan)
    if not any(spec is not None for spec in specs):
        return plan
    normalized_specs = tuple(
        replace(spec, dtype=resolve_out_array_dtype(spec.dtype))
        if spec is not None
        else None
        for spec in specs
    )
    return IntentPlan(
        intents=plan.intents,
        visible_param_indices=plan.visible_param_indices,
        out_return_indices=plan.out_return_indices,
        pass_ptr_mask=plan.pass_ptr_mask,
        out_array_return_specs=normalized_specs,
    )


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
        out_array_return_specs=(None,) + get_out_array_return_specs(method_plan),
    )


def shim_arg_type_for_out_return(out_return_type, spec: OutArrayReturnSpec | None):
    if spec is None:
        return out_return_type
    return nbtypes.CPointer(resolve_out_array_dtype(spec.dtype))
