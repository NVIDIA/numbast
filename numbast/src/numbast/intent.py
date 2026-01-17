# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Mapping

from numbast.intent_defs import ArgIntent, IntentPlan


def _parse_arg_intent(cls, v: Any) -> ArgIntent:
    if isinstance(v, ArgIntent):
        return v
    if isinstance(v, str):
        v2 = v.strip().lower()
        if v2 == "in":
            return ArgIntent.in_
        if v2 == "inout_ptr":
            return ArgIntent.inout_ptr
        if v2 == "out_ptr":
            return ArgIntent.out_ptr
        if v2 == "out_return":
            return ArgIntent.out_return
    raise ValueError(f"Unknown arg intent: {v!r}")


setattr(ArgIntent, "parse", classmethod(_parse_arg_intent))


def _is_ref_type(ast_type: Any) -> bool:
    is_ref = False
    if hasattr(ast_type, "is_left_reference"):
        is_ref = is_ref or bool(ast_type.is_left_reference())
    if hasattr(ast_type, "is_right_reference"):
        is_ref = is_ref or bool(ast_type.is_right_reference())
    return bool(is_ref)


def compute_intent_plan(
    *,
    params: list[Any],
    param_types: list[Any],
    overrides: Mapping[str | int, Any] | None,
    allow_out_return: bool = True,
) -> IntentPlan:
    """
    Compute a per-parameter intent plan.

    Parameters
    ----------
    params
        List of ast_canopy ParamVar-like objects (must have `.name`).
    param_types
        List of ast_canopy Type-like objects (must support ref predicates).
        This is used for validation (e.g. `out_return` only allowed for refs).
    overrides
        Mapping from parameter name (str) or 0-based index (int) to intent.
        Values may be strings, enums, or dicts containing `intent`.
    allow_out_return
        Some call sites may not (yet) support `out_return` (e.g. certain special
        methods). When False, specifying out_return raises.
    """
    if len(params) != len(param_types):
        raise ValueError(
            f"params length ({len(params)}) must match param_types length ({len(param_types)})"
        )

    normalized: list[ArgIntent] = [ArgIntent.in_] * len(params)
    if overrides:
        # First apply index-based overrides, then name-based overrides so names win.
        for key, raw in overrides.items():
            if isinstance(raw, dict):
                raw = raw.get("intent", raw.get("Intent", raw.get("INTENT")))
            intent = _parse_arg_intent(ArgIntent, raw)

            if isinstance(key, int):
                if key < 0 or key >= len(params):
                    raise ValueError(
                        f"arg_intent index {key} out of range for {len(params)} params"
                    )
                normalized[key] = intent
            elif isinstance(key, str):
                # Defer name lookup until after we process all keys.
                continue
            else:
                raise TypeError(
                    f"arg_intent keys must be str (param name) or int (0-based index), got {type(key)}"
                )

        name_to_idx = {
            p.name: i for i, p in enumerate(params) if hasattr(p, "name")
        }
        for key, raw in overrides.items():
            if not isinstance(key, str):
                continue
            if isinstance(raw, dict):
                raw = raw.get("intent", raw.get("Intent", raw.get("INTENT")))
            intent = _parse_arg_intent(ArgIntent, raw)
            if key not in name_to_idx:
                raise ValueError(
                    f"arg_intent specified unknown param name {key!r}; known params: {list(name_to_idx.keys())}"
                )
            normalized[name_to_idx[key]] = intent

    # Validation + derived plan
    visible_param_indices: list[int] = []
    out_return_indices: list[int] = []
    pass_ptr_mask: list[bool] = []

    for i, (intent, ty) in enumerate(zip(normalized, param_types)):
        is_ref = _is_ref_type(ty)
        if intent != ArgIntent.in_:
            if not is_ref:
                raise ValueError(
                    f"arg_intent[{i}]={intent.value!r} is only supported for reference parameters (T&/T&&)"
                )
        if intent == ArgIntent.out_return:
            if not allow_out_return:
                raise ValueError(
                    "out_return intent is not supported in this context"
                )
            out_return_indices.append(i)
        else:
            visible_param_indices.append(i)
            pass_ptr_mask.append(
                intent in (ArgIntent.inout_ptr, ArgIntent.out_ptr)
            )

    return IntentPlan(
        intents=tuple(normalized),
        visible_param_indices=tuple(visible_param_indices),
        out_return_indices=tuple(out_return_indices),
        pass_ptr_mask=tuple(pass_ptr_mask),
    )
