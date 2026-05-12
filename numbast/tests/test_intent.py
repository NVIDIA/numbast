# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from ast_canopy import pylibastcanopy
from numba import types as nbtypes

from numbast.intent import compute_intent_plan, get_out_return_ptr_mask
from numbast.intent_utils import out_return_types_for_plan


def _type(name: str, *, lref: bool = False, rref: bool = False):
    return pylibastcanopy.Type(name, name, rref, lref)


def _param(name: str, type_name: str):
    return pylibastcanopy.ParamVar(name, _type(type_name))


def test_out_return_accepts_scalar_pointer_outputs():
    params = [_param("x", "int"), _param("singleMipLevel", "unsigned int *")]
    param_types = [p.type_ for p in params]

    plan = compute_intent_plan(
        params=params,
        param_types=param_types,
        overrides={"singleMipLevel": "out_return"},
    )

    assert plan.visible_param_indices == (0,)
    assert plan.out_return_indices == (1,)
    assert get_out_return_ptr_mask(plan) == (False, True)
    assert out_return_types_for_plan(param_types, plan) == [nbtypes.uint32]


def test_out_return_reference_outputs_do_not_use_pointer_slot():
    params = [_param("out", "int"), _param("x", "int")]
    param_types = [_type("int", lref=True), params[1].type_]

    plan = compute_intent_plan(
        params=params,
        param_types=param_types,
        overrides={"out": "out_return"},
    )

    assert plan.visible_param_indices == (1,)
    assert plan.out_return_indices == (0,)
    assert get_out_return_ptr_mask(plan) == (False, False)
    assert out_return_types_for_plan(param_types, plan) == [nbtypes.int32]
