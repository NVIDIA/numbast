#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Iterable
import os
import re

from ast_canopy import pylibastcanopy
from ast_canopy.decl import Function, FunctionTemplate, StructMethod

from numba.cuda import types as nbtypes

from numbast.intent import compute_intent_plan
from numbast.types import to_c_type_str, to_numba_type


_SPACE_RE = re.compile(r"\s+")
_DEBUG_ENV_VAR = "NUMBAST_TAD_DEBUG"


def _debug_enabled(debug: bool | None) -> bool:
    if debug is not None:
        return debug
    value = os.environ.get(_DEBUG_ENV_VAR, "")
    return value.strip().lower() in ("1", "true", "yes", "on")


def _debug_print(debug: bool | None, msg: str) -> None:
    if _debug_enabled(debug):
        print(f"[numbast.deduction] {msg}")


def _normalize_cxx_type_str(type_str: str) -> str:
    """Normalize C++ type strings for comparison."""
    if not type_str:
        return type_str
    type_str = type_str.strip()
    type_str = _SPACE_RE.sub(" ", type_str)
    # Remove whitespace around pointer/reference markers.
    type_str = type_str.replace(" *", "*").replace("* ", "*")
    type_str = type_str.replace(" &", "&").replace("& ", "&")
    return type_str


def _apply_pass_ptr(cxx_param: str, pass_ptr: bool) -> str:
    if not pass_ptr:
        return cxx_param
    if "*" in cxx_param:
        return cxx_param
    return _normalize_cxx_type_str(f"{cxx_param}*")


def _normalize_numba_arg_type(arg: nbtypes.Type) -> nbtypes.Type:
    if isinstance(arg, nbtypes.Literal):
        literal_type = getattr(arg, "literal_type", None)
        if literal_type is not None:
            return literal_type
    return arg


def _numba_arg_to_cxx_type(arg: nbtypes.Type) -> str | None:
    arg = _normalize_numba_arg_type(arg)
    try:
        return _normalize_cxx_type_str(to_c_type_str(arg))
    except Exception:
        return None


def _param_type_matches_arg(cxx_type: str, arg: nbtypes.Type) -> bool:
    """Best-effort compatibility check for non-templated parameters."""
    nb_expected = to_numba_type(cxx_type)
    if nb_expected is nbtypes.undefined:
        return True
    return nb_expected == _normalize_numba_arg_type(arg)


def _deduce_from_type_pattern(
    cxx_type: str, arg_cxx: str, placeholders: Iterable[str]
) -> dict[str, str] | None:
    placeholders_in_type = [p for p in placeholders if p in cxx_type]
    if not placeholders_in_type:
        return None

    pattern = re.escape(cxx_type)
    order: list[str] = []
    for ph in placeholders_in_type:
        pattern = pattern.replace(re.escape(ph), r"(.*?)")
        order.append(ph)

    match = re.fullmatch(pattern, arg_cxx)
    if not match:
        return None

    deduced: dict[str, str] = {}
    for ph, value in zip(order, match.groups()):
        value = value.strip()
        if not value:
            return None
        if ph in deduced and deduced[ph] != value:
            return None
        deduced[ph] = value
    return deduced


def _replace_placeholders(type_str: str, replacements: dict[str, str]) -> str:
    for key, value in replacements.items():
        type_str = type_str.replace(key, value)
    return type_str


def _specialize_type(
    ast_type: pylibastcanopy.Type, replacements: dict[str, str]
) -> pylibastcanopy.Type:
    new_name = _replace_placeholders(ast_type.name, replacements)
    new_unqualified = _replace_placeholders(
        ast_type.unqualified_non_ref_type_name, replacements
    )
    return pylibastcanopy.Type(
        new_name,
        new_unqualified,
        ast_type.is_right_reference(),
        ast_type.is_left_reference(),
    )


def _specialize_function(
    func: Function, replacements: dict[str, str]
) -> Function:
    new_return = _specialize_type(func.return_type, replacements)
    new_params = [
        pylibastcanopy.ParamVar(p.name, _specialize_type(p.type_, replacements))
        for p in func.params
    ]

    if isinstance(func, StructMethod):
        return StructMethod(
            func.name,
            func.qual_name,
            new_return,
            new_params,
            func.kind,
            func.exec_space,
            func.is_constexpr,
            func.is_move_constructor,
            func.mangled_name,
            func.attributes,
            func.parse_entry_point,
        )

    return Function(
        func.name,
        func.qual_name,
        new_return,
        new_params,
        func.exec_space,
        func.is_constexpr,
        func.mangled_name,
        func.attributes,
        func.parse_entry_point,
    )


def _clone_function_template(
    templ: FunctionTemplate, func: Function
) -> FunctionTemplate:
    return FunctionTemplate(
        templ.template_parameters,
        templ.num_min_required_args,
        func,
        templ.qual_name,
        templ.parse_entry_point,
    )


def _unresolved_placeholders(
    func: Function, placeholder_names: Iterable[str]
) -> bool:
    placeholders = tuple(placeholder_names)
    if any(
        p in func.return_type.unqualified_non_ref_type_name
        for p in placeholders
    ):
        return True
    for param in func.params:
        if any(
            p in param.type_.unqualified_non_ref_type_name for p in placeholders
        ):
            return True
    return False


def deduce_templated_overloads(
    *,
    qualname: str,
    overloads: list[FunctionTemplate],
    args: tuple[nbtypes.Type, ...],
    overrides: dict | None = None,
    debug: bool | None = None,
) -> tuple[list[FunctionTemplate], list[Exception]]:
    """
    Perform template argument deduction for templated method overloads.

    Returns a list of FunctionTemplate objects with fully-specialized
    Function/Method types, plus any arg_intent-related errors encountered
    while computing visible arity.

    Enable debug output by passing debug=True or setting the
    NUMBAST_TAD_DEBUG=1 environment variable.
    """
    specialized: list[FunctionTemplate] = []
    intent_errors: list[Exception] = []

    _debug_print(
        debug,
        f"begin: {qualname}, overloads={len(overloads)}, args={len(args)}, "
        f"overrides={'yes' if overrides else 'no'}",
    )

    for idx, templ in enumerate(overloads):
        _debug_print(
            debug,
            f"overload[{idx}] {templ.function.name}: "
            f"params={[p.type_.unqualified_non_ref_type_name for p in templ.function.params]}",
        )
        if overrides is None:
            visible_param_indices = tuple(range(len(templ.function.params)))
            pass_ptr_mask = tuple(False for _ in visible_param_indices)
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
                _debug_print(
                    debug,
                    f"  intent plan error: {exc}",
                )
                continue
            visible_param_indices = plan.visible_param_indices
            pass_ptr_mask = plan.pass_ptr_mask
            _debug_print(
                debug,
                "  intent plan: "
                f"visible={plan.visible_param_indices}, "
                f"out_return={plan.out_return_indices}, "
                f"pass_ptr={plan.pass_ptr_mask}",
            )

        if len(visible_param_indices) != len(args):
            _debug_print(
                debug,
                "  skip: visible arity mismatch "
                f"(visible={len(visible_param_indices)} vs args={len(args)})",
            )
            continue

        placeholder_names = [
            tp.type_.name
            for tp in templ.template_parameters
            if tp.kind == pylibastcanopy.template_param_kind.type_
        ]
        _debug_print(
            debug,
            f"  placeholders={placeholder_names or 'none'}",
        )
        mapping: dict[str, str] = {}
        failed = False

        for vis_pos, (arg, param_idx) in enumerate(
            zip(args, visible_param_indices)
        ):
            param = templ.function.params[param_idx]
            cxx_param = _normalize_cxx_type_str(
                param.type_.unqualified_non_ref_type_name
            )
            pass_ptr = bool(pass_ptr_mask[vis_pos])
            cxx_param = _apply_pass_ptr(cxx_param, pass_ptr)
            arg_cxx = _numba_arg_to_cxx_type(arg)
            if arg_cxx is None:
                _debug_print(
                    debug,
                    f"  arg[{param_idx}] {param.name}: "
                    f"numba={arg} could not map to C++ type",
                )
                failed = True
                break

            _debug_print(
                debug,
                f"  arg[{param_idx}] {param.name}: "
                f"param={cxx_param}, arg={arg_cxx}, pass_ptr={pass_ptr}",
            )

            deduced = _deduce_from_type_pattern(
                cxx_param, arg_cxx, placeholder_names
            )
            if deduced is None:
                if not _param_type_matches_arg(cxx_param, arg):
                    _debug_print(
                        debug,
                        f"    mismatch: param={cxx_param}, arg={arg}",
                    )
                    failed = True
                    break
                _debug_print(
                    debug,
                    "    no deduction needed; param matches arg",
                )
                continue

            _debug_print(
                debug,
                f"    deduced={deduced}",
            )

            for key, value in deduced.items():
                prev = mapping.get(key)
                if prev is not None and prev != value:
                    _debug_print(
                        debug,
                        f"    conflict: {key}={prev} vs {value}",
                    )
                    failed = True
                    break
                mapping[key] = value
            if failed:
                break

        if failed:
            _debug_print(debug, "  skip: deduction failed")
            continue

        _debug_print(debug, f"  mapping={mapping}")
        specialized_func = _specialize_function(templ.function, mapping)
        if _unresolved_placeholders(specialized_func, placeholder_names):
            _debug_print(
                debug, "  skip: unresolved placeholders after specialize"
            )
            continue

        _debug_print(
            debug,
            "  specialized: "
            f"return={specialized_func.return_type.unqualified_non_ref_type_name}, "
            f"params={[p.type_.unqualified_non_ref_type_name for p in specialized_func.params]}",
        )
        specialized.append(_clone_function_template(templ, specialized_func))

    _debug_print(
        debug,
        f"end: {qualname}, specialized={len(specialized)}, intent_errors={len(intent_errors)}",
    )
    return specialized, intent_errors
