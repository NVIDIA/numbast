# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUDA-Oxide Rust binding plan built from AST Canopy declarations."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from numbast.errors import CudaOxideBindingError
from numbast.name_policy import apply_prefix_removal
from numbast.rust_types import (
    CUDA_ABI_ALIASES,
    PRIMITIVE_RUST_TYPES,
    CudaOxideType,
    cuda_abi_alias_for_arch,
    is_identifier,
    parse_cuda_oxide_type,
    parse_cuda_oxide_type_spelling,
    render_rust_type,
    rust_identifier,
    rust_parameter_name,
    rust_struct_storage,
)

_INTEGER_LITERAL = re.compile(
    r"^[+-]?(?:0[xX][0-9A-Fa-f]+|0[bB][01]+|0[0-7]*|[0-9]+)(?:[uUlL]+)?$"
)

_EXECUTION_SPACE_NAMES = {
    "execution_space.undefined": "undefined",
    "execution_space.host": "host",
    "execution_space.device": "device",
    "execution_space.host_device": "host_device",
    "execution_space.global_": "global",
}
_CUDA_OXIDE_RESERVED_PREFIX = "cuda_oxide_"
_LEGACY_NVVM_SMALL_C_TYPES = frozenset(
    {
        "_Bool",
        "bool",
        "char",
        "int8_t",
        "int16_t",
        "short",
        "signed char",
        "signed short",
        "uint8_t",
        "uint16_t",
        "unsigned char",
        "unsigned short",
        "__half",
        "half",
        "__nv_bfloat16",
    }
)
_LEGACY_NVVM_SMALL_RUST_TYPES = frozenset(
    {"bool", "i8", "i16", "u8", "u16", "f16"}
)


@dataclass(frozen=True)
class CudaOxideParameter:
    c_name: str
    rust_name: str
    type_: CudaOxideType


@dataclass(frozen=True)
class CudaOxideFunction:
    native_name: str
    public_name: str
    execution_space: str
    return_type: CudaOxideType
    parameters: tuple[CudaOxideParameter, ...]

    @property
    def signature_key(self):
        return (
            self.native_name,
            self.return_type,
            tuple(parameter.type_ for parameter in self.parameters),
        )


@dataclass(frozen=True)
class CudaOxideEnum:
    name: str
    rust_underlying_type: str
    enumerators: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class CudaOxideStruct:
    name: str
    size: int
    alignment: int
    storage_type: str
    fields: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class CudaOxideTypeAlias:
    name: str
    underlying: CudaOxideType


@dataclass
class CudaOxideBindingPlan:
    functions: list[CudaOxideFunction] = field(default_factory=list)
    enums: list[CudaOxideEnum] = field(default_factory=list)
    structs: list[CudaOxideStruct] = field(default_factory=list)
    type_aliases: list[CudaOxideTypeAlias] = field(default_factory=list)
    cuda_aliases: dict[str, tuple[str, int, int]] = field(default_factory=dict)
    exclusions: list[dict[str, str]] = field(default_factory=list)


def translate_constant_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    rendered = str(value).strip()
    if not _INTEGER_LITERAL.fullmatch(rendered):
        raise ValueError(f"unsupported non-literal constant value {value!r}")
    return re.sub(r"[uUlL]+$", "", rendered)


def _validate_type(
    type_: CudaOxideType,
    typedefs: dict[str, Any],
    enums: dict[str, Any],
    records: dict[str, Any],
    diagnostics: list[str],
    context: str,
    seen: set[str] | None = None,
    behind_pointer: bool = False,
):
    base = type_.base_name
    if base in PRIMITIVE_RUST_TYPES:
        return
    if base in CUDA_ABI_ALIASES:
        if base == "double2" and not type_.pointer_depth and not behind_pointer:
            diagnostics.append(
                f"{context}: CUDA vector {base!r} is passed by value; CUDA-Oxide "
                "only supports it behind a pointer"
            )
        return
    if base in enums:
        return
    if base in records:
        if not type_.pointer_depth and not behind_pointer:
            diagnostics.append(
                f"{context}: record {base!r} is passed by value; CUDA-Oxide "
                "Round 1 only supports record pointees"
            )
        return
    if base in typedefs:
        seen = set() if seen is None else seen
        if base in seen:
            diagnostics.append(f"{context}: cyclic typedef involving {base!r}")
            return
        seen.add(base)
        try:
            underlying = parse_cuda_oxide_type_spelling(
                typedefs[base].underlying_name
            )
        except ValueError as error:
            diagnostics.append(f"{context}: typedef {base!r}: {error}")
            return
        _validate_type(
            underlying,
            typedefs,
            enums,
            records,
            diagnostics,
            context,
            seen,
            behind_pointer or bool(type_.pointer_depth),
        )
        return
    diagnostics.append(f"{context}: unsupported C ABI type {base!r}")


def build_cuda_oxide_binding_plan(
    declarations: Any, config: Any
) -> CudaOxideBindingPlan:
    """Select declarations and build the strict Round 1 binding plan."""

    plan = CudaOxideBindingPlan()
    diagnostics: list[str] = []
    typedef_decls = {item.name: item for item in declarations.typedefs}
    enum_decls = {item.name: item for item in declarations.enums if item.name}
    record_decls = {
        item.name: item
        for item in declarations.structs
        if item.name not in config.exclude_structs
    }
    prefix_removal = config.api_prefix_removal.get("Function", [])

    device_candidates = [
        function
        for function in declarations.functions
        if _EXECUTION_SPACE_NAMES[str(function.exec_space)]
        in {"device", "host_device"}
        and function.name not in config.exclude_functions
        and not (
            config.skip_prefix and function.name.startswith(config.skip_prefix)
        )
    ]
    if device_candidates and any(
        getattr(function, "is_c_linkage", None) is None
        for function in device_candidates
    ):
        raise CudaOxideBindingError(
            [
                (
                    "AST Canopy does not expose Function.is_c_linkage; install the "
                    "AST Canopy version shipped with this Numbast checkout"
                )
            ]
        )

    seen_native: dict[str, CudaOxideFunction] = {}
    seen_public: dict[str, str] = {}
    for function in declarations.functions:
        space = _EXECUTION_SPACE_NAMES[str(function.exec_space)]
        if function.name in config.exclude_functions:
            plan.exclusions.append(
                {
                    "kind": "function",
                    "name": function.name,
                    "reason": "configured",
                }
            )
            continue
        if config.skip_prefix and function.name.startswith(config.skip_prefix):
            plan.exclusions.append(
                {
                    "kind": "function",
                    "name": function.name,
                    "reason": "skip-prefix",
                }
            )
            continue
        if space not in {"device", "host_device"}:
            plan.exclusions.append(
                {
                    "kind": "function",
                    "name": function.name,
                    "reason": f"execution-space:{space}",
                }
            )
            continue
        if not function.is_c_linkage:
            diagnostics.append(
                f"function {function.name!r}: device declaration does not have C linkage"
            )
            continue
        if function.is_variadic:
            diagnostics.append(
                f"function {function.name!r}: variadic device declarations are unsupported"
            )
            continue
        if function.mangled_name != function.name:
            diagnostics.append(
                f"function {function.name!r}: C symbol mismatch "
                f"({function.mangled_name!r})"
            )
            continue
        if not is_identifier(function.name):
            diagnostics.append(
                f"function {function.name!r}: native C symbol cannot be represented "
                "as an exact CUDA-Oxide Rust identifier"
            )
            continue
        native_rust_name = rust_identifier(function.name)
        if native_rust_name not in {function.name, f"r#{function.name}"}:
            diagnostics.append(
                f"function {function.name!r}: native C symbol cannot be represented "
                "as an exact CUDA-Oxide Rust identifier"
            )
            continue
        if function.name.startswith(_CUDA_OXIDE_RESERVED_PREFIX):
            diagnostics.append(
                f"function {function.name!r}: native C symbol uses CUDA-Oxide's "
                f"reserved {_CUDA_OXIDE_RESERVED_PREFIX!r} prefix"
            )
            continue

        try:
            return_type = parse_cuda_oxide_type(function.return_type)
        except ValueError as error:
            diagnostics.append(
                f"function {function.name!r} return type: {error}"
            )
            continue
        _validate_type(
            return_type,
            typedef_decls,
            enum_decls,
            record_decls,
            diagnostics,
            f"function {function.name!r} return type",
        )
        if return_type.array_dimensions:
            diagnostics.append(
                f"function {function.name!r}: array return types are unsupported"
            )

        parameters = []
        used_parameter_names = set()
        for index, parameter in enumerate(function.params):
            try:
                type_ = parse_cuda_oxide_type(parameter.type_)
            except ValueError as error:
                diagnostics.append(
                    f"function {function.name!r} parameter {index}: {error}"
                )
                continue
            _validate_type(
                type_,
                typedef_decls,
                enum_decls,
                record_decls,
                diagnostics,
                f"function {function.name!r} parameter {index}",
            )
            if type_.array_dimensions and not type_.pointer_depth:
                diagnostics.append(
                    f"function {function.name!r} parameter {index}: by-value arrays "
                    "are unsupported"
                )
            parameter_name = rust_parameter_name(parameter.name, index)
            if parameter_name in used_parameter_names:
                parameter_name = f"{parameter_name}_{index}"
            used_parameter_names.add(parameter_name)
            parameters.append(
                CudaOxideParameter(
                    c_name=parameter.name,
                    rust_name=parameter_name,
                    type_=type_,
                )
            )

        public_name = apply_prefix_removal(function.name, prefix_removal)
        try:
            rust_identifier(public_name)
        except ValueError as error:
            diagnostics.append(
                f"function {function.name!r} public name: {error}"
            )
            continue

        bound = CudaOxideFunction(
            native_name=function.name,
            public_name=public_name,
            execution_space=space,
            return_type=return_type,
            parameters=tuple(parameters),
        )
        previous = seen_native.get(bound.native_name)
        if previous is not None:
            if previous.signature_key == bound.signature_key:
                plan.exclusions.append(
                    {
                        "kind": "function",
                        "name": function.name,
                        "reason": "duplicate-declaration",
                    }
                )
            else:
                diagnostics.append(
                    f"function {function.name!r}: C symbol has conflicting signatures"
                )
            continue
        rust_public_name = rust_identifier(public_name)
        prior_native = seen_public.get(rust_public_name)
        if prior_native is not None and prior_native != function.name:
            diagnostics.append(
                f"function name collision after prefix removal: {prior_native!r} and "
                f"{function.name!r} both map to {public_name!r}"
            )
            continue
        seen_native[bound.native_name] = bound
        seen_public[rust_public_name] = bound.native_name
        plan.functions.append(bound)

    native_rust_names = {
        rust_identifier(function.native_name): function.native_name
        for function in plan.functions
    }
    for function in plan.functions:
        if function.public_name == function.native_name:
            continue
        rendered_public = rust_identifier(function.public_name)
        conflicting_native = native_rust_names.get(rendered_public)
        if (
            conflicting_native is not None
            and conflicting_native != function.native_name
        ):
            diagnostics.append(
                f"function public name {function.public_name!r} for "
                f"{function.native_name!r} conflicts with native symbol "
                f"{conflicting_native!r}"
            )

    for template in declarations.function_templates:
        plan.exclusions.append(
            {
                "kind": "function-template",
                "name": template.function.name,
                "reason": "round-1-c-api-only",
            }
        )
    for template in declarations.class_templates:
        plan.exclusions.append(
            {
                "kind": "class-template",
                "name": template.record.name,
                "reason": "round-1-c-api-only",
            }
        )

    used_bases = {
        type_.base_name
        for function in plan.functions
        for type_ in [
            function.return_type,
            *(parameter.type_ for parameter in function.parameters),
        ]
    }
    pending = list(used_bases)
    while pending:
        base = pending.pop()
        if base in CUDA_ABI_ALIASES:
            plan.cuda_aliases[base] = cuda_abi_alias_for_arch(
                base, config.gpu_arch[0]
            )
        if base in typedef_decls and all(
            item.name != base for item in plan.type_aliases
        ):
            try:
                underlying = parse_cuda_oxide_type_spelling(
                    typedef_decls[base].underlying_name
                )
            except ValueError as error:
                diagnostics.append(f"typedef {base!r}: {error}")
            else:
                identity_tag_alias = (
                    underlying.base_name == base
                    and not underlying.pointer_depth
                    and not underlying.array_dimensions
                    and (base in record_decls or base in enum_decls)
                )
                if not identity_tag_alias:
                    plan.type_aliases.append(
                        CudaOxideTypeAlias(base, underlying)
                    )
                    pending.append(underlying.base_name)
        if base in enum_decls and all(item.name != base for item in plan.enums):
            declaration = enum_decls[base]
            try:
                underlying = parse_cuda_oxide_type(declaration.underlying_type)
                rust_underlying = render_rust_type(
                    underlying, plan, typedef_decls
                )
                enumerators = tuple(
                    (name, translate_constant_literal(value))
                    for name, value in zip(
                        declaration.enumerators, declaration.enumerator_values
                    )
                )
            except ValueError as error:
                diagnostics.append(f"enum {base!r}: {error}")
            else:
                plan.enums.append(
                    CudaOxideEnum(base, rust_underlying, enumerators)
                )
        if base in record_decls and all(
            item.name != base for item in plan.structs
        ):
            declaration = record_decls[base]
            try:
                storage = rust_struct_storage(
                    declaration.sizeof_, declaration.alignof_
                )
            except ValueError as error:
                diagnostics.append(f"record {base!r}: {error}")
            else:
                plan.structs.append(
                    CudaOxideStruct(
                        name=base,
                        size=declaration.sizeof_,
                        alignment=declaration.alignof_,
                        storage_type=storage,
                        fields=tuple(
                            (field.name, field.type_.name)
                            for field in declaration.fields
                        ),
                    )
                )

    # Named and anonymous enum values are useful C API constants even when the
    # enum itself is not present in a selected signature.
    known_enum_names = {item.name for item in plan.enums}
    for declaration in declarations.enums:
        if declaration.name in known_enum_names:
            continue
        try:
            underlying = parse_cuda_oxide_type(declaration.underlying_type)
            rust_underlying = render_rust_type(underlying, plan, typedef_decls)
            enumerators = tuple(
                (name, translate_constant_literal(value))
                for name, value in zip(
                    declaration.enumerators, declaration.enumerator_values
                )
            )
        except ValueError as error:
            diagnostics.append(
                f"enum {declaration.name or '<anonymous>'!r}: {error}"
            )
        else:
            synthetic_name = declaration.name or ""
            plan.enums.append(
                CudaOxideEnum(synthetic_name, rust_underlying, enumerators)
            )

    plan.functions.sort(key=lambda item: item.native_name)
    plan.enums.sort(key=lambda item: (item.name, item.enumerators))
    plan.structs.sort(key=lambda item: item.name)
    plan.type_aliases.sort(key=lambda item: item.name)
    plan.exclusions.sort(
        key=lambda item: (item["kind"], item["name"], item["reason"])
    )
    if diagnostics:
        raise CudaOxideBindingError(diagnostics)
    return plan


def modern_nvvm_required_symbols(plan: CudaOxideBindingPlan) -> list[str]:
    """Return symbols whose by-value ABI needs CUDA-Oxide's sm_100+ path."""

    type_aliases = {item.name: item.underlying for item in plan.type_aliases}
    enums = {
        item.name: item.rust_underlying_type for item in plan.enums if item.name
    }

    def is_small_by_value(
        type_: CudaOxideType, seen: set[str] | None = None
    ) -> bool:
        if type_.pointer_depth:
            return False
        base = type_.base_name
        if base in _LEGACY_NVVM_SMALL_C_TYPES:
            return True
        if base in enums:
            return enums[base] in _LEGACY_NVVM_SMALL_RUST_TYPES
        if base not in type_aliases:
            return False
        seen = set() if seen is None else seen
        if base in seen:
            return False
        return is_small_by_value(type_aliases[base], {*seen, base})

    required = []
    for function in plan.functions:
        signature_types = [
            function.return_type,
            *(parameter.type_ for parameter in function.parameters),
        ]
        if any(is_small_by_value(type_) for type_ in signature_types):
            required.append(function.native_name)
    return required
