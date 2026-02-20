# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional, cast
from collections import defaultdict
from tempfile import NamedTemporaryFile
import inspect
import logging
import re
import warnings

from ast_canopy import pylibastcanopy

from numba.cuda import types as nbtypes
from numba.cuda.extending import (
    register_model,
    make_attribute_wrapper,
    lower_builtin,
)
from numba.cuda.typing import signature as nb_signature
from numba.cuda.typing.templates import (
    ConcreteTemplate,
    AttributeTemplate,
    CallableTemplate,
    AbstractTemplate,
)
from numba.cuda.datamodel.models import StructModel, OpaqueModel
from numba.cuda.cudadecl import register_global, register, register_attr
from numba.cuda.cudaimpl import lower
from numba.cuda.core.imputils import numba_typeref_ctor
from numba.cuda.typing.npydecl import parse_dtype
from numba.cuda.core.errors import RequireLiteralValue, TypingError

from ast_canopy.api import parse_declarations_from_source
from ast_canopy.decl import (
    ClassTemplateSpecialization,
    StructMethod,
    ClassTemplate,
    FunctionTemplate,
)

from numbast.types import (
    CTYPE_MAPS as C2N,
    to_numba_type,
    is_c_floating_type,
    is_c_integral_type,
    to_c_type_str,
    to_numba_arg_type,
)
from numbast.intent import ArgIntent, IntentPlan, compute_intent_plan
from numbast.utils import (
    deduplicate_overloads,
    make_struct_ctor_shim,
    make_struct_regular_method_shim,
)
from numbast.callconv import FunctionCallConv
from numbast.shim_writer import ShimWriterBase
from numbast.deduction import (
    deduce_templated_overloads,
    deduce_templated_overloads_with_mappings,
)
from numbast.overload_selection import _select_templated_overload


logger = logging.getLogger(__name__)

ConcreteTypeCache: dict[object, nbtypes.TypeRef] = {}
ConcreteTypeDeclCache: dict[object, ClassTemplateSpecialization] = {}
_TEMPLATED_METHOD_LOWERING_CACHE: set[
    tuple[str, nbtypes.Type, tuple[nbtypes.Type, ...]]
] = set()


def clear_concrete_type_caches() -> None:
    """Clear class-template concrete type caches."""
    ConcreteTypeCache.clear()
    ConcreteTypeDeclCache.clear()


class MetaType(nbtypes.Type):
    def __init__(self, qualified_template_name):
        super().__init__(name=qualified_template_name + "MetaType")
        self.qualified_template_name = qualified_template_name


class MetaFunctionType(nbtypes.Type):
    def __init__(self, template_name):
        super().__init__(name=template_name + "MetaFunctionType")
        self.template_name = template_name


def bind_cxx_struct_ctor(
    ctor: StructMethod,
    struct_name: str,
    s_type_ref: nbtypes.TypeRef,
    shim_writer: ShimWriterBase,
) -> Optional[list]:
    """Create bindings for a C++ struct constructor and return its argument types.

    Parameters
    ----------

    ctor : StructMethod
        Constructor declaration of struct in CXX
    struct_name : str
        The name of the struct from which this constructor belongs to
    s_type : numba.types.Type
        The Numba type of the struct
    S : object
        The Python API of the struct
    shim_writer : ShimWriterBase
        The shim writer to write the shim layer code.

    Returns
    -------
    list of argument types, optional
        If the constructor is a move constructor, return ``None``. Otherwise,
        return the list of argument types.
    """

    if ctor.is_move_constructor:
        # move constructor is trivially supported in Numba / Python, skip
        return None

    param_types = [to_numba_arg_type(arg) for arg in ctor.param_types]
    arg_is_ref = [
        bool(t.is_left_reference() or t.is_right_reference())
        for t in ctor.param_types
    ]

    # Lowering
    # Note that libclang always consider the return type of a constructor
    # is void. So we need to manually specify the return type here.
    mangled_name = deduplicate_overloads(ctor.mangled_name)
    shim_func_name = f"{mangled_name}_nbst"

    # Dynamically generate the shim layer:
    # FIXME: All params are passed by pointers, then dereferenced in shim.
    # temporary solution for mismatching function prototype against definition.
    # See above lowering for details.
    shim = make_struct_ctor_shim(
        shim_name=shim_func_name, struct_name=struct_name, params=ctor.params
    )

    logger.debug("CTOR SHIM: %s", shim)

    ctor_cc = FunctionCallConv(
        mangled_name, shim_writer, shim, arg_is_ref=arg_is_ref
    )

    @lower(numba_typeref_ctor, s_type_ref, *param_types)
    def ctor_impl(context, builder, sig, args):
        # `numba_typeref_ctor` includes the typeref as the first argument; the
        # generated shim expects only the actual constructor params.
        ctor_sig = nb_signature(s_type_ref.instance_type, *param_types)
        return ctor_cc(builder, context, ctor_sig, args[1:])

    return param_types


def bind_cxx_ctsd_ctors(
    struct_decl: ClassTemplateSpecialization,
    s_type_ref: nbtypes.TypeRef,
    shim_writer: ShimWriterBase,
):
    """Given a C++ struct declaration, generate bindings for its constructors.

    Parameters
    ----------

    struct_decl: Struct
        The declaration of the struct in CXX
    S: object
        The Python API of the struct
    s_type: numba.types.Type
        The Numba type of the struct
    shim_writer: ShimWriterBase
        The shim writer to write the shim layer code.
    """

    s_type = s_type_ref.instance_type

    ctor_params: list[list[Any]] = []
    for ctor in struct_decl.constructors():
        param_types = bind_cxx_struct_ctor(
            ctor=ctor,
            struct_name=struct_decl.qual_name,
            s_type_ref=s_type_ref,
            shim_writer=shim_writer,
        )
        if param_types is not None:
            ctor_params.append(param_types)

    # Constructor typing:
    @register
    class TypeRefCallTemplate(ConcreteTemplate):
        key = numba_typeref_ctor
        cases = [
            nb_signature(s_type, s_type_ref, *arglist)
            for arglist in ctor_params
        ]

    register_global(numba_typeref_ctor, nbtypes.Function(TypeRefCallTemplate))

    @register
    class CtorTemplate(ConcreteTemplate):
        key = s_type
        cases = [nb_signature(s_type, *arglist) for arglist in ctor_params]


def _resolve_method_arg_intent(
    arg_intent: dict | None,
    struct_decl: ClassTemplateSpecialization,
    method_name: str,
) -> dict | None:
    """Resolve per-method arg_intent overrides for class-template methods.

    Lookup order:
    - Try the base template name (``struct_decl.name``).
    - Then the base name alias (``struct_decl.base_name``) if present.
    - Finally the fully instantiated C++ name (``struct_decl.qual_name``).

    The arg_intent schema is:
    ``{class_name: {method_name: {arg_name: intent_name}}}``.

    The first matching class_name key is selected, then method_name is looked
    up within that mapping. If no match is found, return ``None``.
    """
    if not arg_intent:
        return None

    candidates = []
    for attr in ("name", "base_name", "qual_name"):
        if hasattr(struct_decl, attr):
            value = getattr(struct_decl, attr)
            if value:
                candidates.append(value)

    seen = set()
    for base in candidates:
        if base in seen:
            continue
        seen.add(base)
        class_overrides = arg_intent.get(base)
        if not isinstance(class_overrides, dict):
            continue
        method_overrides = class_overrides.get(method_name)
        if method_overrides is not None:
            return method_overrides

    return None


def _normalize_arg_intent(arg_intent: dict | None) -> tuple | None:
    """Normalize arg_intent into a stable, hashable cache key."""
    if not arg_intent:
        return None

    def _intent_value(value: Any) -> str:
        return value.value if isinstance(value, ArgIntent) else str(value)

    def _key_order(key: Any) -> tuple:
        if isinstance(key, int):
            return (0, key)
        if isinstance(key, str):
            return (1, key)
        return (2, repr(key))

    normalized = []
    for class_key in sorted(arg_intent.keys()):
        class_overrides = arg_intent[class_key]
        if class_overrides is None:
            norm_class = None
        else:
            norm_methods = []
            for method_key in sorted(class_overrides.keys()):
                overrides = class_overrides[method_key]
                if overrides is None:
                    norm_overrides = None
                else:
                    items = []
                    for key, value in overrides.items():
                        items.append(
                            (_key_order(key), key, _intent_value(value))
                        )
                    items.sort()
                    norm_overrides = tuple(
                        (key, value) for _, key, value in items
                    )
                norm_methods.append((method_key, norm_overrides))
            norm_class = tuple(norm_methods)
        normalized.append((class_key, norm_class))

    return tuple(normalized)


def bind_cxx_struct_regular_method(
    struct_decl: ClassTemplateSpecialization,
    method_decl: StructMethod,
    s_type: nbtypes.Type,
    shim_writer: ShimWriterBase,
    *,
    arg_intent: dict | None = None,
) -> nb_signature:
    cxx_return_type = to_numba_type(
        method_decl.return_type.unqualified_non_ref_type_name
    )
    overrides = _resolve_method_arg_intent(
        arg_intent, struct_decl, method_decl.name
    )

    if overrides is None:
        param_types = [
            to_numba_arg_type(arg) for arg in method_decl.param_types
        ]
        param_arg_is_ref = [
            bool(t.is_left_reference() or t.is_right_reference())
            for t in method_decl.param_types
        ]
        # Numba method lowering signatures include the receiver as the first arg.
        arg_is_ref = [False, *param_arg_is_ref]
        return_type = cxx_return_type
        intent_plan = None
        out_return_types = None
    else:
        method_plan = compute_intent_plan(
            params=method_decl.params,
            param_types=method_decl.param_types,
            overrides=overrides,
            allow_out_return=True,
        )
        intent_plan = IntentPlan(
            intents=(ArgIntent.in_,) + method_plan.intents,
            visible_param_indices=(0,)
            + tuple(i + 1 for i in method_plan.visible_param_indices),
            out_return_indices=tuple(
                i + 1 for i in method_plan.out_return_indices
            ),
            pass_ptr_mask=(False,) + method_plan.pass_ptr_mask,
        )

        param_types = []
        for orig_idx in method_plan.visible_param_indices:
            base = to_numba_type(
                method_decl.param_types[orig_idx].unqualified_non_ref_type_name
            )
            if method_plan.intents[orig_idx].value in ("inout_ptr", "out_ptr"):
                param_types.append(nbtypes.CPointer(base))
            else:
                param_types.append(base)

        out_return_types = [
            to_numba_type(
                method_decl.param_types[i].unqualified_non_ref_type_name
            )
            for i in method_plan.out_return_indices
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
        arg_is_ref = None

    # Lowering
    mangled_name = deduplicate_overloads(f"__{method_decl.mangled_name}")
    shim_func_name = f"{mangled_name}_nbst"

    shim = make_struct_regular_method_shim(
        shim_name=shim_func_name,
        # For class-template specializations we must spell the instantiated C++
        # type (e.g. `Foo<128, int>`) in the shim, not just the base template name.
        struct_name=struct_decl.qual_name,
        method_name=method_decl.name,
        return_type=method_decl.return_type.unqualified_non_ref_type_name,
        params=method_decl.params,
    )

    method_cc = FunctionCallConv(
        mangled_name,
        shim_writer,
        shim,
        arg_is_ref=arg_is_ref,
        intent_plan=intent_plan,
        out_return_types=out_return_types,
        cxx_return_type=cxx_return_type,
    )

    qualname = f"{s_type}.{method_decl.name}"

    @lower(qualname, s_type, *param_types)
    def _method_impl(context, builder, sig, args):
        return method_cc(builder, context, sig, args)

    return nb_signature(return_type, *param_types, recvr=s_type)


def bind_cxx_struct_regular_methods(
    struct_decl: ClassTemplateSpecialization,
    s_type: nbtypes.Type,
    shim_writer: ShimWriterBase,
    *,
    arg_intent: dict | None = None,
) -> dict[str, ConcreteTemplate]:
    """

    Return
    ------

    Mapping from function names to list of signatures.
    """

    method_overloads: dict[str, list[nb_signature]] = defaultdict(list)

    for method in struct_decl.regular_member_functions():
        sig = bind_cxx_struct_regular_method(
            struct_decl, method, s_type, shim_writer, arg_intent=arg_intent
        )
        method_overloads[method.name].append(sig)

    method_templates: dict[str, ConcreteTemplate] = {}

    for name, sigs in method_overloads.items():

        class MethodDecl(ConcreteTemplate):
            key = f"{s_type}.{name}"
            cases = sigs

        method_templates[name] = MethodDecl

    return method_templates


def bind_cxx_struct_templated_method(
    struct_decl: ClassTemplateSpecialization,
    *,
    name: str,
    overloads: list[FunctionTemplate],
    s_type: nbtypes.Type,
    shim_writer: ShimWriterBase,
    arg_intent: dict | None = None,
) -> type[AbstractTemplate]:
    """Create bindings for a (possibly overloaded) templated method of a C++ struct.

    NOTE: We intentionally *do not* use `lower_attr(...)=dummy_value` for templated
    methods. Doing so loses the receiver value and the call ends up seeing `i8* null`
    for the first argument. Instead, we return a normal Numba `BoundFunction`, whose
    template's `generic()` runs at *call time* and can see the concrete argument types.
    """

    # NOTE: Templated methods can be overloaded (e.g. CCCL's BlockLoad::Load).
    # We merge all overloads under one AbstractTemplate and select at call time.
    qualname = f"{s_type}.{name}"

    class MergedTemplatedMethodDecl(AbstractTemplate):
        key = qualname

        def generic(
            self, args, kwds, overloads=overloads, arg_intent=arg_intent
        ):
            # BoundFunction passes only the explicit call args (receiver is implicit).
            recvr = self.this
            param_types = tuple(args)

            overrides = _resolve_method_arg_intent(
                arg_intent, struct_decl, name
            )

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

            templated_method = _select_templated_overload(
                qualname=qualname,
                overloads=specialized_overloads,
                param_types=param_types,
                kwds=kwds,
                overrides=overrides,
            )

            cxx_return_type = to_numba_type(
                templated_method.function.return_type.unqualified_non_ref_type_name
            )
            # Determine intent/return behavior: default to native C++ return,
            # or apply arg_intent overrides to surface out-params as returns.
            if overrides is None:
                method_plan = None
                intent_plan = None
                out_return_types = None
                return_type = cxx_return_type
            else:
                method_plan = compute_intent_plan(
                    params=templated_method.function.params,
                    param_types=templated_method.function.param_types,
                    overrides=overrides,
                    allow_out_return=True,
                )
                intent_plan = IntentPlan(
                    intents=(ArgIntent.in_,) + method_plan.intents,
                    visible_param_indices=(0,)
                    + tuple(i + 1 for i in method_plan.visible_param_indices),
                    out_return_indices=tuple(
                        i + 1 for i in method_plan.out_return_indices
                    ),
                    pass_ptr_mask=(False,) + method_plan.pass_ptr_mask,
                )
                out_return_types = [
                    to_numba_type(
                        templated_method.function.param_types[
                            i
                        ].unqualified_non_ref_type_name
                    )
                    for i in method_plan.out_return_indices
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

            lowering_key = (qualname, recvr, param_types)
            if lowering_key not in _TEMPLATED_METHOD_LOWERING_CACHE:
                _TEMPLATED_METHOD_LOWERING_CACHE.add(lowering_key)

                @lower(qualname, recvr, *param_types)
                def _impl(
                    context,
                    builder,
                    sig,
                    args,
                    param_types=param_types,
                    templated_method=templated_method,
                    method_plan=method_plan,
                    intent_plan=intent_plan,
                    out_return_types=out_return_types,
                    cxx_return_type=cxx_return_type,
                ):
                    # args[0] is the receiver value (a struct value), args[1:] are parameters.
                    param_types_inner = sig.args[1:]
                    if (
                        method_plan is None
                        or not method_plan.out_return_indices
                    ):
                        param_types_for_shim = tuple(param_types_inner)
                        pass_ptr_mask_for_shim = (
                            None
                            if method_plan is None
                            else tuple(method_plan.pass_ptr_mask)
                        )
                    else:
                        if len(param_types_inner) != len(
                            method_plan.visible_param_indices
                        ):
                            raise ValueError(
                                "Signature args do not match templated intent plan visible params: "
                                f"sig has {len(param_types_inner)} params but plan expects {len(method_plan.visible_param_indices)}"
                            )
                        out_return_map = {
                            orig_idx: out_pos
                            for out_pos, orig_idx in enumerate(
                                method_plan.out_return_indices
                            )
                        }
                        # Reconstruct full C++ param order by merging visible
                        # params with out_return slots, keeping a shim-aligned
                        # pass_ptr_mask.
                        param_types_for_shim_list = []
                        pass_ptr_mask_for_shim_list = []
                        visible_idx = 0
                        for orig_idx in range(len(method_plan.intents)):
                            out_pos = out_return_map.get(orig_idx)
                            if out_pos is not None:
                                param_types_for_shim_list.append(
                                    out_return_types[out_pos]
                                )
                                pass_ptr_mask_for_shim_list.append(False)
                            else:
                                param_types_for_shim_list.append(
                                    param_types_inner[visible_idx]
                                )
                                pass_ptr_mask_for_shim_list.append(
                                    method_plan.pass_ptr_mask[visible_idx]
                                )
                                visible_idx += 1
                        param_types_for_shim = tuple(param_types_for_shim_list)
                        pass_ptr_mask_for_shim = tuple(
                            pass_ptr_mask_for_shim_list
                        )

                    mangled_name = deduplicate_overloads(
                        f"__{templated_method.function.mangled_name}"
                    )
                    shim_func_name = f"{mangled_name}_nbst"
                    formal_args_str, actual_args_str = (
                        _make_templated_method_shim_arg_strings(
                            param_types_inner=tuple(param_types_for_shim),
                            cxx_params=templated_method.function.params,
                            pass_ptr_mask=pass_ptr_mask_for_shim,
                        )
                    )

                    shim = make_struct_regular_method_shim(
                        shim_name=shim_func_name,
                        struct_name=struct_decl.qual_name,
                        method_name=templated_method.function.name,
                        return_type=templated_method.function.return_type.unqualified_non_ref_type_name,
                        formal_args_str=formal_args_str,
                        actual_args_str=actual_args_str,
                    )

                    logger.debug("TEMPLATED METHOD SHIM: %s", shim)

                    param_arg_is_ref = [
                        bool(t.is_left_reference() or t.is_right_reference())
                        for t in templated_method.function.param_types
                    ]
                    arg_is_ref = [False, *param_arg_is_ref]

                    method_cc = FunctionCallConv(
                        mangled_name,
                        shim_writer,
                        shim,
                        arg_is_ref=arg_is_ref,
                        intent_plan=intent_plan,
                        out_return_types=out_return_types,
                        cxx_return_type=cxx_return_type,
                    )

                    return method_cc(builder, context, sig, args)

            # Return a method signature (receiver is implicit).
            return nb_signature(return_type, *param_types, recvr=recvr)

    return MergedTemplatedMethodDecl


_CXX_ARRAY_TYPE_RE = re.compile(r"^(?P<base>.*?)(?P<sizes>(\[[^\]]+\])+)\s*$")


def _make_templated_method_shim_arg_strings(
    *,
    param_types_inner: tuple[nbtypes.Type, ...],
    cxx_params: list[pylibastcanopy.ParamVar],
    pass_ptr_mask: tuple[bool, ...] | None = None,
) -> tuple[str, str]:
    """
    Build (formal_args_str, actual_args_str) for `make_struct_regular_method_shim`
    using:
    - `param_types_inner` (from Numba signature) to match the ABI/calling convention
    - `cxx_params` (from parsed C++ decl) to produce correct expressions when the
      C++ parameter type is not directly representable in Numba (e.g. `T (&)[N]`).

    `formal_args_str` is either empty or begins with a leading comma (matching the
    expectations of `make_struct_regular_method_shim`).
    """
    if len(param_types_inner) != len(cxx_params):
        raise ValueError(
            "Templated method parameter mismatch: "
            f"sig has {len(param_types_inner)} params, but C++ decl has {len(cxx_params)} params."
        )
    if pass_ptr_mask is not None and len(pass_ptr_mask) != len(
        param_types_inner
    ):
        raise ValueError(
            "Templated method pass_ptr_mask mismatch: "
            f"sig has {len(param_types_inner)} params, but pass_ptr_mask has {len(pass_ptr_mask)} entries."
        )

    formal_parts: list[str] = []
    actual_parts: list[str] = []

    for i, (nb_ty, cxx_param) in enumerate(zip(param_types_inner, cxx_params)):
        # Make the shim signature match the *Numba* ABI types.
        # Numba's default ABI passes pointer-to-value for each argument, so the
        # shim should accept a pointer to the Numba-visible type and then
        # dereference once when calling the C++ method.
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

        # NOTE: `unqualified_non_ref_type_name` strips references, so a true
        # `T (&)[N]` parameter will present as an array type `T [N]` here, and
        # we must consult `is_left_reference()` to know whether it was a ref.
        cxx_ty = cxx_param.type_.unqualified_non_ref_type_name
        if use_pass_ptr:
            # If C++ expects a pointer type, pass the pointer value directly.
            # If it expects a reference/value type, dereference once to bind.
            if "*" in cxx_ty:
                actual_default = f"arg{i}"
            else:
                actual_default = f"*arg{i}"
        m = _CXX_ARRAY_TYPE_RE.match(cxx_ty)
        is_lref = cxx_param.type_.is_left_reference()
        if m and is_lref:
            # C++ wants a reference-to-array parameter like: `int (&)[1]`.
            # Numba typically models the passed value as `CPointer(int)`, and
            # `base_expr` is `int*`.
            #
            # Validate we got a compatible Numba argument type, then rebuild the
            # shim signature to accept the array-ref directly. (At the LLVM IR
            # level this is still a pointer value, and Numba will pass `i32*`.)
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

            # Override the shim's formal parameter spelling to match the C++ API:
            #   int (&arg1)[1]
            # and pass it straight through to the underlying method call.
            formal_parts.append(f"{base} (&arg{i}){sizes}")
            actual_parts.append(f"arg{i}")
        else:
            formal_parts.append(formal_default)
            actual_parts.append(actual_default)

    formal_args_str = "," + ",".join(formal_parts) if formal_parts else ""
    actual_args_str = ",".join(actual_parts)
    return formal_args_str, actual_args_str


def bind_cxx_struct_templated_methods(
    struct_decl: ClassTemplateSpecialization,
    s_type: nbtypes.Type,
    shim_writer: ShimWriterBase,
    *,
    arg_intent: dict | None = None,
) -> dict[str, type[AbstractTemplate]]:
    """Create bindings for a templated method of a C++ struct."""

    # NOTE: Templated methods can be overloaded (e.g. CCCL's BlockLoad::Load).
    # If we key only by name, later overloads overwrite earlier ones. Instead we
    # merge all overloads under one AbstractTemplate and select at call time.
    method_overloads: dict[str, list[FunctionTemplate]] = defaultdict(list)

    for templated_method in struct_decl.templated_member_functions():
        method_overloads[templated_method.function.name].append(
            templated_method
        )

    method_to_template: dict[str, type[AbstractTemplate]] = {}

    for name, overloads in method_overloads.items():
        method_to_template[name] = bind_cxx_struct_templated_method(
            struct_decl,
            name=name,
            overloads=overloads,
            s_type=s_type,
            shim_writer=shim_writer,
            arg_intent=arg_intent,
        )

    return method_to_template


def bind_cxx_class_template_specialization(
    shim_writer: ShimWriterBase,
    struct_decl: ClassTemplateSpecialization,
    instance_type_ref: nbtypes.Type,
    aliases: dict[
        str, list[str]
    ] = {},  # XXX: this should be just a list of aliases
    *,
    arg_intent: dict | None = None,
) -> object:
    """
    Create bindings for a C++ struct.

    Parameters
    ----------
    shim_writer : ShimWriterBase
        The shim writer to write the shim layer code.
    struct_decl : Struct
        Declaration of the struct type in CXX
    parent_type : nbtypes.Type, optional
        Parent type of the Python API, by default nbtypes.Type
    data_model : type, optional
        Data model for the struct, by default StructModel
    aliases : dict[str, list[str]], optional
        Mappings from the name of the struct to a list of aliases.
        For example in C++: typedef A B; typedef A C; then
        aliases = {"A": ["B", "C"]}

    Returns
    -------
    S : object
        The Python API of the struct.
    shim: str
        The generated shim layer code for struct methods.
    """

    s_type = instance_type_ref.instance_type
    S_type = type(s_type)

    # Any type that was parsed from C++ should be added to type record:
    # It also needs to happen before method typings - because copy constructors
    # needs to know the type of itself even if the definition is incomplete.
    C2N[struct_decl.name] = s_type
    if struct_decl.name in aliases:
        for alias in aliases[struct_decl.name]:
            C2N[alias] = s_type

    # Data Model
    @register_model(S_type)
    class S_model(StructModel):
        def __init__(self, dmm, fe_type, struct_decl=struct_decl):
            members = [
                (
                    f.name,
                    to_numba_type(f.type_.unqualified_non_ref_type_name),
                )
                for f in struct_decl.fields
            ]
            super().__init__(dmm, fe_type, members)

    # ----------------------------------------------------------------------------------
    # Method, Attributes Typing and Lowering:

    method_templates = bind_cxx_struct_regular_methods(
        struct_decl, s_type, shim_writer, arg_intent=arg_intent
    )

    templated_method_to_template = bind_cxx_struct_templated_methods(
        struct_decl, s_type, shim_writer, arg_intent=arg_intent
    )

    public_fields_tys = {f.name: f.type_ for f in struct_decl.public_fields()}

    @register_attr
    class S_attr(AttributeTemplate):
        key = s_type

        def _field_ty(self, attr: str) -> nbtypes.Type:
            field_ty = public_fields_tys[attr]
            return to_numba_type(field_ty.unqualified_non_ref_type_name)

        def _method_ty(self, typ, attr: str) -> nbtypes.BoundFunction:
            template = method_templates[attr]
            return nbtypes.BoundFunction(template, typ)

        def _templated_method_ty(self, typ, attr: str) -> nbtypes.BoundFunction:
            template = templated_method_to_template[attr]
            return nbtypes.BoundFunction(template, typ)

        def generic_resolve(self, typ, attr):
            if attr in public_fields_tys:
                return self._field_ty(attr)
            has_regular = attr in method_templates
            has_templated = attr in templated_method_to_template
            if has_regular and has_templated:
                # FIXME: support shared names by doing TAD before overload selection.
                warnings.warn(
                    "Attribute name collision for "
                    f"'{attr}': present in both method_templates "
                    f"({method_templates[attr]}) and "
                    "templated_method_to_template "
                    f"({templated_method_to_template[attr]}). "
                    "Regular method will occlude templated method.",
                    UserWarning,
                    stacklevel=2,
                )
                return self._method_ty(typ, attr)
            elif has_regular:
                return self._method_ty(typ, attr)
            elif has_templated:
                return self._templated_method_ty(typ, attr)
            elif attr == "__call__":
                # Special case when invoking tranpoline typing of numba_typeref_ctor
                # Reject to look for internal typing.
                pass
            else:
                raise AttributeError(attr)

        for field_name in public_fields_tys.keys():
            make_attribute_wrapper(S_type, field_name, field_name)

    # ----------------------------------------------------------------------------------
    # Constructors:
    bind_cxx_ctsd_ctors(struct_decl, instance_type_ref, shim_writer)

    # Return the handle to the type in Numba
    return s_type


def concrete_typing():
    """Create a new concrete type object"""

    class ConcreteType(nbtypes.Type):
        def __init__(self, meta_type, **targs):
            self.meta_type = meta_type
            self.targs = targs
            super().__init__(name=self.angled_targs_str())

        def angled_targs_str(self) -> str:
            """Return type name with `key=value` Numba type format surrounded by angle brackets.
            Example: BlockScan<T=int32, BLOCK_DIM_X=128>
            Mostly used for debugging purposes.
            """
            return (
                self.meta_type.qualified_template_name
                + f"<{', '.join([f'{tparam_name}={targ}' for tparam_name, targ in self.targs.items()])}>"
            )

        def angled_targs_str_as_c(self) -> str:
            """Return C++ style template instantiation string.
            Example: BlockScan<int, 128>
            Used to format shim function strings.
            """
            return (
                self.meta_type.qualified_template_name
                + f"<{', '.join([f'{targ}' for targ in self.targs_dict_as_c().values()])}>"
            )

        def targs_dict_as_c(self):
            """Reversely map the template parameter types into C type strings."""

            def to_c_str(obj: nbtypes.Type | int | float) -> str:
                if isinstance(obj, nbtypes.Type):
                    return to_c_type_str(obj)

                if isinstance(obj, (int, float)):
                    return str(obj)

                if isinstance(obj, str):
                    return obj

                raise ValueError(
                    f"Unknown object to use in C shim function: {obj}"
                )

            return {
                tparam_name: to_c_str(targ)
                for tparam_name, targ in self.targs.items()
            }

    return ConcreteType


def struct_type_from_instantiation(
    instance: nbtypes.Type,
    shim_writer: ShimWriterBase,
    header_path: str,
    *,
    arg_intent: dict | None = None,
) -> tuple[nbtypes.TypeRef, ClassTemplateSpecialization]:
    # Clang determines to populate all members of an instantiated class based
    # on whether an explicit instantitation definition exists. The following
    # code snippet creates such instantiation.
    src = f"""\n
#include "{header_path}"
template class {instance.angled_targs_str_as_c()};
"""

    with NamedTemporaryFile("w") as f:
        f.write(src)
        f.flush()

        try:
            decls = parse_declarations_from_source(
                f.name, [header_path, f.name], compute_capability="sm_86"
            )
        except pylibastcanopy.ParseError as e:
            e.add_note(f"Error when parsing string: {src}")
            raise

    specializations = decls.class_template_specializations
    decl = specializations[0]

    instance_type_ref = nbtypes.TypeRef(instance)
    bind_cxx_class_template_specialization(
        shim_writer, decl, instance_type_ref, arg_intent=arg_intent
    )

    return instance_type_ref, decl


def _bind_tparams(
    decl: ClassTemplate, *args, **kwargs
) -> dict[str, nbtypes.Type]:
    def _get_literal_type_cls(c_typ_: str) -> nbtypes.Literal:
        """Get the corresbonding Numba literal type based on C type string.
        Fall back to Literal if unknown.
        """
        if is_c_integral_type(c_typ_):
            return nbtypes.IntegerLiteral
        elif is_c_floating_type(c_typ_):
            return nbtypes.FloatLiteral
        else:
            return nbtypes.Literal

    if args:
        raise TypingError(
            "Template parameter error message is only configurable via keyword argument, not positional argument."
        )

    res: dict[str, nbtypes.Type] = {}

    targs = decl.tparam_dict
    for tparam_name, type_ in kwargs.items():
        param_decl = targs.get(tparam_name, None)
        if param_decl:
            # If it's a non-type template parameter
            if param_decl.kind == pylibastcanopy.template_param_kind.non_type:
                literal_type = _get_literal_type_cls(param_decl.type_)
                if not isinstance(type_, literal_type):
                    # Require the input to be a literal value
                    raise RequireLiteralValue(kwargs[tparam_name])

                res[tparam_name] = type_.literal_value
            else:
                res[tparam_name] = parse_dtype(type_)

    # Reorder the result dict to the same key order as C++
    res = {k: res[k] for k in targs if k in res}
    return res


def _split_ctor_and_tparam_kwargs(
    decl: ClassTemplate, kwargs: dict[str, nbtypes.Type]
) -> tuple[dict[str, nbtypes.Type], dict[str, nbtypes.Type]]:
    """Split call keyword arguments into constructor kwargs and template kwargs.

    We require template-parameter keywords to appear after constructor keywords.
    This keeps lowering deterministic because constructor runtime arguments remain
    a stable prefix of the call argument list.
    """
    ctor_kwargs: dict[str, nbtypes.Type] = {}
    tparam_kwargs: dict[str, nbtypes.Type] = {}
    tparam_names = set(decl.tparam_dict.keys())

    seen_tparam_keyword = False
    for name, value in kwargs.items():
        if name in tparam_names:
            seen_tparam_keyword = True
            tparam_kwargs[name] = value
        else:
            if seen_tparam_keyword:
                raise TypingError(
                    "Constructor keyword arguments must appear before "
                    "template-parameter keywords."
                )
            ctor_kwargs[name] = value

    return ctor_kwargs, tparam_kwargs


def _bind_ctor_call_args(
    ctor: StructMethod,
    positional_args: tuple[nbtypes.Type, ...],
    ctor_kwargs: dict[str, nbtypes.Type],
) -> tuple[nbtypes.Type, ...]:
    """Bind a call-site argument list to a constructor's parameter order."""
    params = ctor.params
    n_params = len(params)

    if len(positional_args) > n_params:
        raise TypeError(
            f"{ctor.qual_name}: expected at most {n_params} constructor args, "
            f"got {len(positional_args)} positional args."
        )

    bound: list[nbtypes.Type | None] = [None] * n_params
    for i, arg in enumerate(positional_args):
        bound[i] = arg

    name_to_index = {
        param.name: i for i, param in enumerate(params) if param.name
    }
    last_bound_index = len(positional_args) - 1

    for kw_name, kw_value in ctor_kwargs.items():
        idx = name_to_index.get(kw_name)
        if idx is None:
            raise TypeError(
                f"{ctor.qual_name}: unexpected constructor keyword argument "
                f"'{kw_name}'."
            )
        if idx < len(positional_args):
            raise TypeError(
                f"{ctor.qual_name}: constructor argument '{kw_name}' was "
                "already provided positionally."
            )
        if bound[idx] is not None:
            raise TypeError(
                f"{ctor.qual_name}: duplicate constructor argument '{kw_name}'."
            )
        if idx < last_bound_index:
            raise TypeError(
                f"{ctor.qual_name}: constructor keyword arguments must follow "
                "constructor parameter declaration order."
            )
        last_bound_index = idx
        bound[idx] = kw_value

    missing = [
        params[i].name or f"arg{i}"
        for i, value in enumerate(bound)
        if value is None
    ]
    if missing:
        raise TypeError(
            f"{ctor.qual_name}: missing constructor argument(s): "
            f"{', '.join(missing)}."
        )

    return cast(tuple[nbtypes.Type, ...], tuple(bound))


def _get_ctor_candidates_from_template_record(
    record: Any,
) -> list[StructMethod]:
    """Collect constructor candidates from a class-template record.

    AST-Canopy exposes constructors for concrete specializations via
    ``constructors()``, but some versions expose class-template constructors
    only in ``record.methods`` with a constructor ``method_kind``.
    """
    ctors = [
        ctor for ctor in record.constructors() if not ctor.is_move_constructor
    ]
    if ctors:
        return ctors

    ctor_kinds = {
        pylibastcanopy.method_kind.default_constructor,
        pylibastcanopy.method_kind.copy_constructor,
        pylibastcanopy.method_kind.move_constructor,
        pylibastcanopy.method_kind.converting_constructor,
        pylibastcanopy.method_kind.other_constructor,
    }
    methods = getattr(record, "methods", [])
    return [
        method
        for method in methods
        if method.kind in ctor_kinds and not method.is_move_constructor
    ]


def _merge_explicit_and_deduced_targs(
    decl: ClassTemplate,
    explicit_targs: dict[str, Any],
    deduced_type_mapping: dict[str, str],
) -> dict[str, Any]:
    """Merge explicit template args with deduced type template arguments."""
    merged = dict(explicit_targs)

    for tparam in decl.template_parameters:
        if tparam.kind != pylibastcanopy.template_param_kind.type_:
            continue

        deduced_cxx_ty = deduced_type_mapping.get(tparam.name)
        if deduced_cxx_ty is None:
            placeholder_name = getattr(
                getattr(tparam, "type_", None), "name", None
            )
            if placeholder_name:
                deduced_cxx_ty = deduced_type_mapping.get(placeholder_name)
        if deduced_cxx_ty is None:
            continue

        deduced_nb_ty = to_numba_type(deduced_cxx_ty)
        if deduced_nb_ty is nbtypes.undefined or isinstance(
            deduced_nb_ty, nbtypes.Opaque
        ):
            raise TypingError(
                "Failed to map deduced C++ template type "
                f"'{deduced_cxx_ty}' for template parameter "
                f"'{tparam.name}' to a Numba type."
            )

        explicit_ty = merged.get(tparam.name)
        if explicit_ty is not None and explicit_ty != deduced_nb_ty:
            raise TypingError(
                "Template parameter conflict for "
                f"'{tparam.name}': explicitly provided {explicit_ty} but "
                f"deduced {deduced_nb_ty} from constructor arguments."
            )

        merged[tparam.name] = deduced_nb_ty

    tparam_names = [tp.name for tp in decl.template_parameters]
    required_names = tparam_names[: decl.num_min_required_args]
    missing_required = [name for name in required_names if name not in merged]
    if missing_required:
        raise TypingError(
            "Missing required template parameter(s): "
            f"{', '.join(missing_required)}."
        )

    provided_positions = [
        i for i, name in enumerate(tparam_names) if name in merged
    ]
    if not provided_positions:
        return {}

    last_provided = max(provided_positions)
    missing_prefix = [
        tparam_names[i]
        for i in range(last_provided + 1)
        if tparam_names[i] not in merged
    ]
    if missing_prefix:
        raise TypingError(
            "Cannot skip template parameter(s) "
            f"{', '.join(missing_prefix)} while providing later template "
            "parameters. Supply all template parameters up to the last one "
            "you provide/deduce."
        )

    return {name: merged[name] for name in tparam_names[: last_provided + 1]}


def _get_or_bind_concrete_type(
    *,
    cache_key: object,
    instance: nbtypes.Type,
    shim_writer: ShimWriterBase,
    header_path: str,
    arg_intent: dict | None,
) -> tuple[nbtypes.TypeRef, ClassTemplateSpecialization, bool]:
    """Get or create a bound concrete class-template specialization."""
    if cache_key in ConcreteTypeCache and cache_key in ConcreteTypeDeclCache:
        return (
            ConcreteTypeCache[cache_key],
            ConcreteTypeDeclCache[cache_key],
            False,
        )

    instance_type_ref, specialization_decl = struct_type_from_instantiation(
        instance,
        shim_writer,
        header_path,
        arg_intent=arg_intent,
    )
    ConcreteTypeCache[cache_key] = instance_type_ref
    ConcreteTypeDeclCache[cache_key] = specialization_decl
    return instance_type_ref, specialization_decl, True


def _ctor_signature_key(
    ctor: StructMethod,
) -> tuple[tuple[str, bool, bool], ...]:
    """Build a stable constructor identity signature from parameter types."""
    return tuple(
        (
            param.unqualified_non_ref_type_name,
            bool(param.is_left_reference()),
            bool(param.is_right_reference()),
        )
        for param in ctor.param_types
    )


def _resolve_specialization_ctor(
    *,
    specialization_decl: ClassTemplateSpecialization,
    template_ctor: StructMethod,
    ctor_call_args: tuple[nbtypes.Type, ...] | None = None,
) -> StructMethod:
    """Resolve specialization ctor by identity, not by positional index."""
    specialization_ctors = [
        ctor
        for ctor in specialization_decl.constructors()
        if not ctor.is_move_constructor
    ]
    if not specialization_ctors:
        raise TypingError(
            "No usable constructors found for specialization "
            f"{specialization_decl.qual_name}."
        )

    template_mangled_name = getattr(template_ctor, "mangled_name", None)
    template_sig = _ctor_signature_key(template_ctor)
    template_ref_mask = tuple(
        bool(param.is_left_reference() or param.is_right_reference())
        for param in template_ctor.param_types
    )

    if template_mangled_name:
        mangled_matches = [
            ctor
            for ctor in specialization_ctors
            if ctor.mangled_name == template_mangled_name
        ]
        if len(mangled_matches) == 1:
            return mangled_matches[0]
        if len(mangled_matches) > 1:
            sig_matches = [
                ctor
                for ctor in mangled_matches
                if _ctor_signature_key(ctor) == template_sig
            ]
            if len(sig_matches) == 1:
                return sig_matches[0]

    if ctor_call_args is not None:
        call_matches = []
        for ctor in specialization_ctors:
            ctor_param_types = tuple(
                to_numba_arg_type(arg) for arg in ctor.param_types
            )
            if ctor_param_types == ctor_call_args:
                call_matches.append(ctor)
        if len(call_matches) == 1:
            return call_matches[0]
        if len(call_matches) > 1:
            ref_matches = [
                ctor
                for ctor in call_matches
                if tuple(
                    bool(
                        param.is_left_reference() or param.is_right_reference()
                    )
                    for param in ctor.param_types
                )
                == template_ref_mask
            ]
            if len(ref_matches) == 1:
                return ref_matches[0]

    sig_matches = [
        ctor
        for ctor in specialization_ctors
        if _ctor_signature_key(ctor) == template_sig
    ]
    if len(sig_matches) == 1:
        return sig_matches[0]

    template_qual_name = getattr(template_ctor, "qual_name", None)
    if template_qual_name:
        qual_matches = [
            ctor
            for ctor in specialization_ctors
            if ctor.qual_name == template_qual_name
        ]
        if len(qual_matches) == 1:
            return qual_matches[0]

    ref_matches = [
        ctor
        for ctor in specialization_ctors
        if len(tuple(ctor.param_types)) == len(tuple(template_ctor.param_types))
        and tuple(
            bool(param.is_left_reference() or param.is_right_reference())
            for param in ctor.param_types
        )
        == template_ref_mask
    ]
    if len(ref_matches) == 1:
        return ref_matches[0]

    raise TypingError(
        "Failed to resolve constructor for specialization "
        f"{specialization_decl.qual_name} from template constructor "
        f"{getattr(template_ctor, 'qual_name', '<unknown>')}."
    )


def _ensure_ctor_callconv(
    *,
    specialization_decl: ClassTemplateSpecialization,
    template_ctor: StructMethod,
    ctor_call_args: tuple[nbtypes.Type, ...],
    instance_type: nbtypes.Type,
    shim_writer: ShimWriterBase,
    ctor_callconv_cache: dict[
        tuple[nbtypes.Type, tuple[nbtypes.Type, ...]], FunctionCallConv
    ],
) -> tuple[FunctionCallConv, tuple[nbtypes.Type, ...]]:
    """Build (or reuse) constructor callconv for a concrete specialization."""
    ctor = _resolve_specialization_ctor(
        specialization_decl=specialization_decl,
        template_ctor=template_ctor,
        ctor_call_args=ctor_call_args,
    )
    ctor_param_types = tuple(to_numba_arg_type(arg) for arg in ctor.param_types)
    cache_key = (instance_type, ctor_param_types)
    cached = ctor_callconv_cache.get(cache_key)
    if cached is not None:
        return cached, ctor_param_types

    mangled_name = deduplicate_overloads(ctor.mangled_name)
    shim_func_name = f"{mangled_name}_nbst"
    shim = make_struct_ctor_shim(
        shim_name=shim_func_name,
        struct_name=specialization_decl.qual_name,
        params=ctor.params,
    )
    arg_is_ref = [
        bool(t.is_left_reference() or t.is_right_reference())
        for t in ctor.param_types
    ]
    ctor_cc = FunctionCallConv(
        mangled_name,
        shim_writer,
        shim,
        arg_is_ref=arg_is_ref,
    )
    ctor_callconv_cache[cache_key] = ctor_cc
    return ctor_cc, ctor_param_types


def _register_meta_type(
    stub: object,
    meta_type: nbtypes.Type,
    ctd: ClassTemplate,
    shim_writer: ShimWriterBase,
    header_path: str,
    ctor_callconv_cache: dict[
        tuple[nbtypes.Type, tuple[nbtypes.Type, ...]], FunctionCallConv
    ],
    ctor_lowering_cache: dict[
        tuple[nbtypes.Type, tuple[nbtypes.Type, ...]],
        tuple[FunctionCallConv, int, tuple[nbtypes.Type, ...]],
    ],
    *,
    arg_intent: dict | None = None,
):
    ConcreteType = concrete_typing()

    @register
    class MetaType_template_decl(CallableTemplate):
        key = stub

        def generic(
            self,
            meta_type=meta_type,
            decl=ctd,
            shim_writer=shim_writer,
            header_path=header_path,
            ctor_callconv_cache=ctor_callconv_cache,
            ctor_lowering_cache=ctor_lowering_cache,
            arg_intent=arg_intent,
        ):
            def typer(*args, **kwargs):
                # Fold keyword-only template/constructor call styles into a
                # positionalized pysig that Numba's CallableTemplate expects.
                typer.pysig = inspect.Signature(
                    parameters=[
                        inspect.Parameter(
                            f"p{i}",
                            inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        )
                        for i in range(len(args))
                    ]
                    + [
                        inspect.Parameter(
                            name,
                            inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        )
                        for name in kwargs
                    ]
                )

                ctor_kwargs, tparam_kwargs = _split_ctor_and_tparam_kwargs(
                    decl, kwargs
                )
                explicit_targs = _bind_tparams(decl, **tparam_kwargs)

                constructors = _get_ctor_candidates_from_template_record(
                    decl.record
                )
                if not constructors:
                    raise TypingError(
                        f"No usable constructors found for {decl.record.qual_name}."
                    )

                candidate_bind_errors: list[str] = []
                viable: list[
                    tuple[
                        nbtypes.Type,
                        tuple[nbtypes.Type, ...],
                        FunctionCallConv,
                        int,
                        bool,
                    ]
                ] = []
                full_call_arg_types = tuple(args) + tuple(kwargs.values())
                intent_key = _normalize_arg_intent(arg_intent)

                for ctor in constructors:
                    try:
                        ctor_call_args = _bind_ctor_call_args(
                            ctor, tuple(args), ctor_kwargs
                        )
                    except TypeError as exc:
                        candidate_bind_errors.append(str(exc))
                        continue

                    ctor_template = FunctionTemplate(
                        decl.template_parameters,
                        decl.num_min_required_args,
                        ctor,
                        ctor.qual_name,
                        decl.parse_entry_point,
                    )
                    specialized_with_mappings, _ = (
                        deduce_templated_overloads_with_mappings(
                            qualname=ctor.qual_name,
                            overloads=[ctor_template],
                            args=ctor_call_args,
                            overrides=None,
                        )
                    )
                    if not specialized_with_mappings:
                        continue

                    _, deduced_type_mapping = specialized_with_mappings[0]
                    try:
                        merged_targs = _merge_explicit_and_deduced_targs(
                            decl,
                            explicit_targs,
                            deduced_type_mapping,
                        )
                    except TypingError as exc:
                        candidate_bind_errors.append(str(exc))
                        continue

                    instance = ConcreteType(meta_type, **merged_targs)
                    unique_id = instance.name
                    cache_key = (
                        (unique_id, intent_key)
                        if intent_key is not None
                        else unique_id
                    )
                    (
                        instance_type_ref,
                        specialization_decl,
                        created,
                    ) = _get_or_bind_concrete_type(
                        cache_key=cache_key,
                        instance=instance,
                        shim_writer=shim_writer,
                        header_path=header_path,
                        arg_intent=arg_intent,
                    )
                    ctor_cc, ctor_param_types = _ensure_ctor_callconv(
                        specialization_decl=specialization_decl,
                        template_ctor=ctor,
                        ctor_call_args=ctor_call_args,
                        instance_type=instance_type_ref.instance_type,
                        shim_writer=shim_writer,
                        ctor_callconv_cache=ctor_callconv_cache,
                    )
                    viable.append(
                        (
                            instance_type_ref.instance_type,
                            ctor_param_types,
                            ctor_cc,
                            len(ctor_param_types),
                            created,
                        )
                    )

                if not viable:
                    detail = (
                        f" Candidate bind errors: {candidate_bind_errors[0]}"
                        if candidate_bind_errors
                        else ""
                    )
                    raise TypingError(
                        "No viable constructor/template-parameter binding found "
                        f"for {decl.record.qual_name} with {len(args)} positional "
                        f"and {len(kwargs)} keyword arguments.{detail}"
                    )

                if len(viable) > 1:
                    raise TypingError(
                        "Ambiguous constructor selection for "
                        f"{decl.record.qual_name}: {len(viable)} viable "
                        "constructor/template bindings."
                    )

                (
                    instance_type,
                    _ctor_param_types,
                    ctor_cc,
                    ctor_arg_count,
                    created,
                ) = viable[0]

                if created:
                    self.context.refresh()

                ctor_lowering_cache[(instance_type, full_call_arg_types)] = (
                    ctor_cc,
                    ctor_arg_count,
                    _ctor_param_types,
                )
                return instance_type

            return typer

    register_global(stub, nbtypes.Function(MetaType_template_decl))


def bind_cxx_class_template(
    class_template_decl: ClassTemplate,
    shim_writer: ShimWriterBase,
    header_path: str,
    *,
    arg_intent: dict | None = None,
):
    # Stub class
    TC = type(class_template_decl.record.qual_name, (object,), {})

    # Typing
    TC_templated_type = MetaType(class_template_decl.record.qual_name)

    # Data model
    @register_model(MetaType)
    class TC_Template_model(OpaqueModel):
        def __init__(self, dmm, fe_type):
            OpaqueModel.__init__(self, dmm, fe_type)

    ctor_callconv_cache: dict[
        tuple[nbtypes.Type, tuple[nbtypes.Type, ...]], FunctionCallConv
    ] = {}
    ctor_lowering_cache: dict[
        tuple[nbtypes.Type, tuple[nbtypes.Type, ...]],
        tuple[FunctionCallConv, int, tuple[nbtypes.Type, ...]],
    ] = {}

    # Class-template constructor lowering:
    # typing records which concrete constructor callconv to use for a given
    # (return_type, call-arg-types) pair; lowering retrieves and invokes it.
    @lower_builtin(TC, nbtypes.VarArg(nbtypes.Any))
    def lower_ctor_dispatch(
        context,
        builder,
        sig,
        args,
        ctor_lowering_cache=ctor_lowering_cache,
    ):
        key = (sig.return_type, tuple(sig.args))
        lowered = ctor_lowering_cache.get(key)
        if lowered is None:
            raise TypingError(
                "Missing constructor lowering for class template "
                f"{class_template_decl.record.qual_name} with return type "
                f"{sig.return_type} and args {tuple(sig.args)}."
            )

        ctor_cc, ctor_arg_count, _ = lowered
        ctor_sig = nb_signature(sig.return_type, *sig.args[:ctor_arg_count])
        return ctor_cc(builder, context, ctor_sig, args[:ctor_arg_count])

    _register_meta_type(
        TC,
        TC_templated_type,
        class_template_decl,
        shim_writer,
        header_path,
        ctor_callconv_cache,
        ctor_lowering_cache,
        arg_intent=arg_intent,
    )

    return TC


def bind_cxx_class_templates(
    class_templates: list[ClassTemplate],
    header_path: str,
    shim_writer: ShimWriterBase,
    *,
    arg_intent: dict | None = None,
) -> list[object]:
    """
    Create bindings for a list of C++ class templates.

    Parameters
    ----------
    shim_writer : ShimWriter
        The shim writer to write the shim layer code.
    class_templates : list[ClassTemplate]
        List of declarations of the class template types in CXX
    header_path : str
        The path to the header file containing the class template declarations.
    arg_intent : dict, optional
        Argument intent overrides to apply to bound methods.

    Returns
    -------
    list[object]
        The Python APIs of the class templates.
    """

    python_apis = []
    for ct in class_templates:
        # Bind the cxx class template
        TC = bind_cxx_class_template(
            class_template_decl=ct,
            shim_writer=shim_writer,
            header_path=header_path,
            arg_intent=arg_intent,
        )
        python_apis.append(TC)

    return python_apis
