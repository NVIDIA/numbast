# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from ast_canopy.decl import FunctionTemplate

from numbast.intent import compute_intent_plan


def _select_templated_overload(
    *,
    qualname: str,
    overloads: list[FunctionTemplate],
    param_types: tuple[Any, ...],
    kwds: dict[str, Any] | None = None,
    overrides: dict | None = None,
) -> FunctionTemplate:
    """
    Select a FunctionTemplate overload for a templated function/method.

    Today we only select by explicit argument count (visible arity). Keep this
    logic centralized so we can expand it with C++-style overload resolution:
    - filter viable candidates (arity/defaults/variadics, arg_intent visibility),
    - rank implicit conversions (exact > promotion > standard > user-defined),
    - prefer better ref/cv binding and non-variadic over variadic,
    - prefer more specialized templates / stronger constraints,
    - treat remaining ties as ambiguous.
    """
    arity = len(param_types)
    candidates: list[FunctionTemplate] = []
    intent_errors: list[Exception] = []

    for templ in overloads:
        if overrides is None:
            visible_arity = len(templ.function.params)
        else:
            try:
                plan = compute_intent_plan(
                    params=templ.function.params,
                    param_types=templ.function.param_types,
                    overrides=overrides,
                    allow_out_return=True,
                )
            except Exception as exc:
                intent_errors.append(exc)
                continue
            visible_arity = len(plan.visible_param_indices)

        if visible_arity == arity:
            candidates.append(templ)

    if overrides is not None and not candidates and intent_errors:
        raise TypeError(
            f"Failed to apply arg_intent overrides for {qualname}: "
            f"{intent_errors[0]}"
        )
    if not candidates:
        raise TypeError(
            f"No matching overload found for {qualname} with {arity} args. "
            f"Overload arities: {[len(t.function.params) for t in overloads]}"
        )
    if len(candidates) > 1:
        raise TypeError(
            f"Ambiguous overload for {qualname} with {arity} args. "
            f"Matching overload arities: {[len(t.function.params) for t in candidates]}"
        )

    return candidates[0]
