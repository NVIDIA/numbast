# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any
from collections import defaultdict
from warnings import warn
import logging
import re

from ast_canopy.pylibastcanopy import execution_space
from ast_canopy.decl import FunctionTemplate

from numba.cuda import types as nbtypes
from numba.cuda.typing import signature as nb_signature
from numba.cuda.typing.templates import AbstractTemplate
from numba.cuda.cudadecl import register, register_global
from numba.cuda.cudaimpl import lower
from numba.core.errors import TypingError

from numbast.callconv import FunctionCallConv
from numbast.deduction import deduce_templated_overloads
from numbast.intent import ArgIntent, compute_intent_plan
from numbast.types import to_c_type_str, to_numba_type
from numbast.utils import deduplicate_overloads, get_return_type_strings
from numbast.shim_writer import ShimWriterBase
from numbast.overload_selection import _select_templated_overload


logger = logging.getLogger(__name__)


def make_new_func_obj():
    def func():
        pass

    return func


# Registry key: (name, intent_key, shim_writer). name is str; intent_key is tuple
# or None for intent/overload dispatch; shim_writer is compared by identity so
# different writer instances get separate entries; make_new_func_obj is the
# default factory.
func_obj_registry: dict[tuple[str, tuple | None, object], object] = defaultdict(
    make_new_func_obj
)


def _normalize_overrides(overrides: dict | None) -> tuple | None:
    if not overrides:
        return None

    def _intent_value(value: Any) -> str:
        return value.value if isinstance(value, ArgIntent) else str(value)

    def _key_order(key: Any) -> tuple:
        if isinstance(key, int):
            return (0, key)
        if isinstance(key, str):
            return (1, key)
        return (2, repr(key))

    items = []
    for key, value in overrides.items():
        items.append((_key_order(key), key, _intent_value(value)))
    items.sort()
    return tuple((key, value) for _, key, value in items)


def _make_templated_function_shim(
    *,
    shim_name: str,
    func_name: str,
    return_type: str,
    formal_args_str: str,
    actual_args_str: str,
    includes: list[str] | None = None,
) -> str:
    function_binding_shim_template = """{includes}
extern "C" __device__ int
{shim_name}({return_type} &retval {formal_args}) {{
    {retval}{func_name}({actual_args});
    return 0;
}}
    """

    retval, return_type = get_return_type_strings(return_type)
    include_list = includes or []
    include_str = "\n".join(
        [f"#include <{include}>" for include in include_list]
    )
    return function_binding_shim_template.format(
        includes=include_str,
        shim_name=shim_name,
        return_type=return_type,
        func_name=func_name,
        formal_args=formal_args_str,
        retval=retval,
        actual_args=actual_args_str,
    )


_CXX_ARRAY_TYPE_RE = re.compile(r"^(?P<base>.*?)(?P<sizes>(\[\d+\])+)\s*$")


def _make_templated_function_shim_arg_strings(
    *,
    param_types_inner: tuple[nbtypes.Type, ...],
    cxx_params: list,
    pass_ptr_mask: tuple[bool, ...] | None = None,
) -> tuple[str, str]:
    """
    Build (formal_args_str, actual_args_str) for templated function shims.

    `formal_args_str` is either empty or begins with a leading comma (matching
    the expectations of `_make_templated_function_shim`).
    """
    if len(param_types_inner) != len(cxx_params):
        raise ValueError(
            "Templated function parameter mismatch: "
            f"sig has {len(param_types_inner)} params, but C++ decl has {len(cxx_params)} params."
        )
    if pass_ptr_mask is not None and len(pass_ptr_mask) != len(
        param_types_inner
    ):
        raise ValueError(
            "Templated function pass_ptr_mask mismatch: "
            f"sig has {len(param_types_inner)} params, but pass_ptr_mask has {len(pass_ptr_mask)} entries."
        )

    formal_parts: list[str] = []
    actual_parts: list[str] = []

    for i, (nb_ty, cxx_param) in enumerate(zip(param_types_inner, cxx_params)):
        c_ty = to_c_type_str(nb_ty)
        pass_ptr = (
            bool(pass_ptr_mask[i]) if pass_ptr_mask is not None else False
        )
        use_pass_ptr = pass_ptr and isinstance(nb_ty, nbtypes.CPointer)
        if use_pass_ptr:
            formal_default = f"{c_ty} arg{i}"
        else:
            formal_default = f"{c_ty}* arg{i}"
        actual_default = f"*arg{i}"

        cxx_ty = cxx_param.type_.unqualified_non_ref_type_name
        if use_pass_ptr:
            if "*" in cxx_ty:
                actual_default = f"arg{i}"
            else:
                actual_default = f"*arg{i}"
        m = _CXX_ARRAY_TYPE_RE.match(cxx_ty)
        is_lref = cxx_param.type_.is_left_reference()
        if m and is_lref:
            if not isinstance(nb_ty, nbtypes.CPointer):
                raise TypingError(
                    f"{cxx_param.name}: expected a pointer argument in Numba "
                    f"to satisfy C++ '{cxx_ty}&', got {nb_ty}"
                )

            pointee = getattr(nb_ty, "dtype", None)
            if pointee is None:
                raise TypingError(
                    f"{cxx_param.name}: cannot determine pointee type for {nb_ty}"
                )

            base = m.group("base").strip()
            sizes = m.group("sizes")

            nb_base = to_c_type_str(pointee).strip()
            base_norm = base.replace("const", "").strip()
            if base_norm != nb_base:
                raise TypingError(
                    f"{cxx_param.name}: Numba argument base type '{nb_base}' does not match "
                    f"C++ array-ref base type '{base}' for '{cxx_ty}&'"
                )

            formal_parts.append(f"{base} (&arg{i}){sizes}")
            actual_parts.append(f"arg{i}")
        else:
            formal_parts.append(formal_default)
            actual_parts.append(actual_default)

    formal_args_str = "," + ",".join(formal_parts) if formal_parts else ""
    actual_args_str = ",".join(actual_parts)
    return formal_args_str, actual_args_str


def bind_cxx_function_template(
    *,
    name: str,
    overloads: list[FunctionTemplate],
    shim_writer: ShimWriterBase,
    arg_intent: dict | None = None,
) -> object:
    """Create bindings for a (possibly overloaded) templated function."""
    overrides = arg_intent.get(name) if arg_intent else None
    intent_key = _normalize_overrides(overrides)
    func = func_obj_registry[(name, intent_key, shim_writer)]
    func.__name__ = name
    qualname = overloads[0].function.qual_name if overloads else name

    @register
    class MergedTemplatedFunctionDecl(AbstractTemplate):
        key = func

        def generic(self, args, kwds, overloads=overloads, overrides=overrides):
            param_types = tuple(args)

            specialized_overloads, intent_errors = deduce_templated_overloads(
                qualname=qualname,
                overloads=overloads,
                args=param_types,
                overrides=overrides,
            )
            if (
                not specialized_overloads
                and overrides is not None
                and intent_errors
            ):
                raise TypeError(
                    f"Failed to apply arg_intent overrides for {qualname}: "
                    f"{intent_errors[0]}"
                )

            templated_func = _select_templated_overload(
                qualname=qualname,
                overloads=specialized_overloads,
                param_types=param_types,
                kwds=kwds,
                overrides=overrides,
            )
            logger.debug(
                "SELECTED OVERLOAD: %s", templated_func.function.param_types
            )

            cxx_return_type = to_numba_type(
                templated_func.function.return_type.unqualified_non_ref_type_name
            )
            if overrides is None:
                func_plan = None
                intent_plan = None
                out_return_types = None
                return_type = cxx_return_type
            else:
                func_plan = compute_intent_plan(
                    params=templated_func.function.params,
                    param_types=templated_func.function.param_types,
                    overrides=overrides,
                    allow_out_return=True,
                )
                intent_plan = func_plan
                out_return_types = [
                    to_numba_type(
                        templated_func.function.param_types[
                            i
                        ].unqualified_non_ref_type_name
                    )
                    for i in func_plan.out_return_indices
                ]
                if out_return_types:
                    if cxx_return_type == nbtypes.void:
                        if len(out_return_types) == 1:
                            return_type = out_return_types[0]
                        else:
                            return_type = nbtypes.Tuple(tuple(out_return_types))
                    else:
                        return_type = nbtypes.Tuple(
                            tuple([cxx_return_type, *out_return_types])
                        )
                else:
                    return_type = cxx_return_type

            @lower(func, *param_types)
            def _impl(
                context,
                builder,
                sig,
                args,
                param_types=param_types,
                templated_func=templated_func,
                func_plan=func_plan,
                intent_plan=intent_plan,
                out_return_types=out_return_types,
                cxx_return_type=cxx_return_type,
            ):
                param_types_inner = sig.args
                if func_plan is None or not func_plan.out_return_indices:
                    param_types_for_shim = tuple(param_types_inner)
                    pass_ptr_mask_for_shim = (
                        None
                        if func_plan is None
                        else tuple(func_plan.pass_ptr_mask)
                    )
                else:
                    if len(param_types_inner) != len(
                        func_plan.visible_param_indices
                    ):
                        raise ValueError(
                            "Signature args do not match templated intent plan visible params: "
                            f"sig has {len(param_types_inner)} params but plan expects {len(func_plan.visible_param_indices)}"
                        )
                    out_return_map = {
                        orig_idx: out_pos
                        for out_pos, orig_idx in enumerate(
                            func_plan.out_return_indices
                        )
                    }
                    visible_iter = iter(param_types_inner)
                    visible_mask_iter = iter(func_plan.pass_ptr_mask)
                    param_types_for_shim_list = []
                    pass_ptr_mask_for_shim_list = []
                    for orig_idx in range(len(func_plan.intents)):
                        out_pos = out_return_map.get(orig_idx)
                        if out_pos is not None:
                            param_types_for_shim_list.append(
                                out_return_types[out_pos]
                            )
                            pass_ptr_mask_for_shim_list.append(False)
                        else:
                            param_types_for_shim_list.append(next(visible_iter))
                            pass_ptr_mask_for_shim_list.append(
                                next(visible_mask_iter)
                            )
                    param_types_for_shim = tuple(param_types_for_shim_list)
                    pass_ptr_mask_for_shim = tuple(pass_ptr_mask_for_shim_list)

                mangled_name = deduplicate_overloads(
                    templated_func.function.mangled_name
                )
                shim_func_name = f"{mangled_name}_nbst"
                formal_args_str, actual_args_str = (
                    _make_templated_function_shim_arg_strings(
                        param_types_inner=tuple(param_types_for_shim),
                        cxx_params=templated_func.function.params,
                        pass_ptr_mask=pass_ptr_mask_for_shim,
                    )
                )

                shim = _make_templated_function_shim(
                    shim_name=shim_func_name,
                    func_name=templated_func.function.name,
                    return_type=templated_func.function.return_type.unqualified_non_ref_type_name,
                    formal_args_str=formal_args_str,
                    actual_args_str=actual_args_str,
                )

                logger.debug("TEMPLATED FUNCTION SHIM: %s", shim)

                param_arg_is_ref = [
                    bool(t.is_left_reference() or t.is_right_reference())
                    for t in templated_func.function.param_types
                ]

                func_cc = FunctionCallConv(
                    mangled_name,
                    shim_writer,
                    shim,
                    arg_is_ref=param_arg_is_ref,
                    intent_plan=intent_plan,
                    out_return_types=out_return_types,
                    cxx_return_type=cxx_return_type,
                )

                return func_cc(builder, context, sig, args)

            return nb_signature(return_type, *param_types)

    register_global(func, nbtypes.Function(MergedTemplatedFunctionDecl))
    return func


def bind_cxx_function_templates(
    *,
    function_templates: list[FunctionTemplate],
    shim_writer: ShimWriterBase,
    skip_prefix: str | None = None,
    skip_non_device: bool = True,
    exclude: set[str] | None = None,
    arg_intent: dict | None = None,
) -> list[object]:
    """Create bindings for a list of C++ function templates."""
    exclude_names = exclude or set()
    overloads_by_name: dict[str, list[FunctionTemplate]] = defaultdict(list)

    for templ in function_templates:
        func_decl = templ.function
        if skip_non_device and func_decl.exec_space not in {
            execution_space.device,
            execution_space.host_device,
        }:
            warn(f"Skipped non device function {func_decl.name}.")
            continue
        if (skip_prefix and func_decl.name.startswith(skip_prefix)) or (
            func_decl.name in exclude_names
        ):
            continue
        if func_decl.is_overloaded_operator() or func_decl.is_operator:
            continue
        overloads_by_name[func_decl.name].append(templ)

    funcs = []
    for name, overloads in overloads_by_name.items():
        func = bind_cxx_function_template(
            name=name,
            overloads=overloads,
            shim_writer=shim_writer,
            arg_intent=arg_intent,
        )
        funcs.append(func)

    return funcs


__all__ = ["bind_cxx_function_template", "bind_cxx_function_templates"]
