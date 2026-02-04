# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Mapping

from numbast.intent_defs import ArgIntent, IntentPlan


def _parse_arg_intent(cls, v: Any) -> ArgIntent:
    """
    Parse an argument intent representation into an ArgIntent member.

    Parameters:
        v (Any): An ArgIntent instance or a string alias (case-insensitive, whitespace ignored). Accepted string aliases: "in", "inout_ptr", "out_ptr", "out_return".

    Returns:
        ArgIntent: The corresponding ArgIntent enum member.

    Raises:
        ValueError: If `v` is neither an ArgIntent nor a recognized string alias.
    """
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
    """
    Determine whether an AST type-like object represents a reference type.

    If `ast_type` exposes `is_left_reference()` or `is_right_reference()`, those methods are consulted;
    the function returns `True` when either indicates a reference, `False` otherwise.

    Parameters:
        ast_type (Any): Type-like object that may implement `is_left_reference()` and/or `is_right_reference()`.

    Returns:
        bool: `True` if the object represents a reference type, `False` otherwise.
    """
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
    Compute a per-parameter intent plan for a function call.

    This determines an ArgIntent for each parameter (defaulting to `ArgIntent.in_`), applies optional overrides (index-based and name-based; name-based overrides take precedence), validates intents against parameter types (non-`in_` intents require reference-like types), and produces an IntentPlan describing which parameters are visible, which are returned via out_return, and which should be passed as pointer-like arguments.

    Parameters:
        params (list[Any]): Parameter-like objects (must have a `.name` attribute when name-based overrides are used).
        param_types (list[Any]): Type-like objects used for validation of reference capability.
        overrides (Mapping[str|int, Any] | None): Optional mapping from 0-based index (int) or parameter name (str) to intent. Values may be an `ArgIntent` or a string parseable by `ArgIntent.parse`. Index-based overrides are applied first; name-based overrides override them.
        allow_out_return (bool): If False, specifying `out_return` as an intent is rejected.

    Returns:
        IntentPlan: Contains:
          - intents: tuple[ArgIntent] for each parameter
          - visible_param_indices: tuple[int] indices of parameters that remain visible (not out_return)
          - out_return_indices: tuple[int] indices of parameters specified as `out_return`
          - pass_ptr_mask: tuple[bool] parallel to visible_param_indices indicating whether the parameter should be passed as a pointer (True for `inout_ptr`/`out_ptr`)

    Raises:
        ValueError: If `params` and `param_types` lengths differ, an index override is out of range, a named override refers to an unknown parameter, a non-`in_` intent is applied to a non-reference type, or `out_return` is disallowed.
        TypeError: If override keys are not `int` or `str`, or override values are not `str` or `ArgIntent`.
    """
    if len(params) != len(param_types):
        raise ValueError(
            f"params length ({len(params)}) must match param_types length ({len(param_types)})"
        )

    normalized: list[ArgIntent] = [ArgIntent.in_] * len(params)
    if overrides:
        # First apply index-based overrides, then name-based overrides so names win.
        for key, raw in overrides.items():
            if type(raw) not in (str, ArgIntent):
                raise TypeError(
                    "arg_intent values must be strings or ArgIntent enums"
                )
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
            if type(raw) not in (str, ArgIntent):
                raise TypeError(
                    "arg_intent values must be strings or ArgIntent enums"
                )
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
