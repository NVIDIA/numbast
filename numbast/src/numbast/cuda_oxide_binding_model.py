# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUDA-Oxide Rust binding model built from AST Canopy declarations."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from numbast.name_policy import apply_prefix_removal
from numbast.rust_types import (
    CUDA_ABI_ALIASES,
    CAbiType,
    PRIMITIVE_RUST_TYPES,
    cuda_abi_alias_for_arch,
    is_identifier,
    parse_c_abi_type,
    parse_c_type_spelling,
    rust_identifier,
    rust_parameter_name,
    rust_record_storage,
    rust_type,
)

_INTEGER_LITERAL = re.compile(
    r"^[+-]?(?:0[xX][0-9A-Fa-f]+|0[bB][01]+|0[0-7]*|[0-9]+)(?:[uUlL]+)?$"
)

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


class BindingModelError(ValueError):
    """Raised after collecting every actionable model diagnostic."""

    def __init__(self, diagnostics: list[str]):
        self.diagnostics = sorted(set(diagnostics))
        details = "\n".join(f"  - {item}" for item in self.diagnostics)
        super().__init__(f"CUDA-Oxide binding generation failed:\n{details}")


@dataclass(frozen=True)
class AbiParameter:
    c_name: str
    rust_name: str
    type_: CAbiType


@dataclass(frozen=True)
class AbiFunction:
    native_name: str
    public_name: str
    execution_space: str
    return_type: CAbiType
    parameters: tuple[AbiParameter, ...]

    @property
    def signature_key(self):
        return (
            self.native_name,
            self.return_type,
            tuple(parameter.type_ for parameter in self.parameters),
        )


@dataclass(frozen=True)
class AbiEnum:
    name: str
    rust_underlying_type: str
    enumerators: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class AbiRecord:
    name: str
    size: int
    alignment: int
    storage_type: str
    fields: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class AbiTypedef:
    name: str
    underlying: CAbiType


@dataclass
class BindingModel:
    functions: list[AbiFunction] = field(default_factory=list)
    enums: list[AbiEnum] = field(default_factory=list)
    records: list[AbiRecord] = field(default_factory=list)
    typedefs: list[AbiTypedef] = field(default_factory=list)
    cuda_aliases: dict[str, tuple[str, int, int]] = field(default_factory=dict)
    exclusions: list[dict[str, str]] = field(default_factory=list)


def _execution_space_name(value: Any) -> str:
    name = getattr(value, "name", None)
    if name:
        return name[:-1] if name.endswith("_") else name
    rendered = str(value).rsplit(".", 1)[-1]
    return rendered[:-1] if rendered.endswith("_") else rendered


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
    type_: CAbiType,
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
            underlying = parse_c_type_spelling(typedefs[base].underlying_name)
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


def build_c_device_model(declarations: Any, config: Any) -> BindingModel:
    """Select declarations and build the strict Round 1 C-device model."""

    model = BindingModel()
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
        if _execution_space_name(function.exec_space)
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
        raise BindingModelError(
            [
                (
                    "AST Canopy does not expose Function.is_c_linkage; install the "
                    "AST Canopy version shipped with this Numbast checkout"
                )
            ]
        )

    seen_native: dict[str, AbiFunction] = {}
    seen_public: dict[str, str] = {}
    for function in declarations.functions:
        space = _execution_space_name(function.exec_space)
        if function.name in config.exclude_functions:
            model.exclusions.append(
                {
                    "kind": "function",
                    "name": function.name,
                    "reason": "configured",
                }
            )
            continue
        if config.skip_prefix and function.name.startswith(config.skip_prefix):
            model.exclusions.append(
                {
                    "kind": "function",
                    "name": function.name,
                    "reason": "skip-prefix",
                }
            )
            continue
        if space not in {"device", "host_device"}:
            model.exclusions.append(
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
            return_type = parse_c_abi_type(function.return_type)
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
                type_ = parse_c_abi_type(parameter.type_)
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
                AbiParameter(
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

        bound = AbiFunction(
            native_name=function.name,
            public_name=public_name,
            execution_space=space,
            return_type=return_type,
            parameters=tuple(parameters),
        )
        previous = seen_native.get(bound.native_name)
        if previous is not None:
            if previous.signature_key == bound.signature_key:
                model.exclusions.append(
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
        model.functions.append(bound)

    native_rust_names = {
        rust_identifier(function.native_name): function.native_name
        for function in model.functions
    }
    for function in model.functions:
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
        model.exclusions.append(
            {
                "kind": "function-template",
                "name": template.function.name,
                "reason": "round-1-c-api-only",
            }
        )
    for template in declarations.class_templates:
        model.exclusions.append(
            {
                "kind": "class-template",
                "name": template.record.name,
                "reason": "round-1-c-api-only",
            }
        )

    used_bases = {
        type_.base_name
        for function in model.functions
        for type_ in [
            function.return_type,
            *(parameter.type_ for parameter in function.parameters),
        ]
    }
    pending = list(used_bases)
    while pending:
        base = pending.pop()
        if base in CUDA_ABI_ALIASES:
            model.cuda_aliases[base] = cuda_abi_alias_for_arch(
                base, config.gpu_arch[0]
            )
        if base in typedef_decls and all(
            item.name != base for item in model.typedefs
        ):
            try:
                underlying = parse_c_type_spelling(
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
                    model.typedefs.append(AbiTypedef(base, underlying))
                    pending.append(underlying.base_name)
        if base in enum_decls and all(
            item.name != base for item in model.enums
        ):
            declaration = enum_decls[base]
            try:
                underlying = parse_c_abi_type(declaration.underlying_type)
                rust_underlying = rust_type(underlying, model, typedef_decls)
                enumerators = tuple(
                    (name, translate_constant_literal(value))
                    for name, value in zip(
                        declaration.enumerators, declaration.enumerator_values
                    )
                )
            except ValueError as error:
                diagnostics.append(f"enum {base!r}: {error}")
            else:
                model.enums.append(AbiEnum(base, rust_underlying, enumerators))
        if base in record_decls and all(
            item.name != base for item in model.records
        ):
            declaration = record_decls[base]
            try:
                storage = rust_record_storage(
                    declaration.sizeof_, declaration.alignof_
                )
            except ValueError as error:
                diagnostics.append(f"record {base!r}: {error}")
            else:
                model.records.append(
                    AbiRecord(
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
    known_enum_names = {item.name for item in model.enums}
    for declaration in declarations.enums:
        if declaration.name in known_enum_names:
            continue
        try:
            underlying = parse_c_abi_type(declaration.underlying_type)
            rust_underlying = rust_type(underlying, model, typedef_decls)
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
            model.enums.append(
                AbiEnum(synthetic_name, rust_underlying, enumerators)
            )

    model.functions.sort(key=lambda item: item.native_name)
    model.enums.sort(key=lambda item: (item.name, item.enumerators))
    model.records.sort(key=lambda item: item.name)
    model.typedefs.sort(key=lambda item: item.name)
    model.exclusions.sort(
        key=lambda item: (item["kind"], item["name"], item["reason"])
    )
    if diagnostics:
        raise BindingModelError(diagnostics)
    return model


def modern_nvvm_required_symbols(model: BindingModel) -> list[str]:
    """Return symbols whose by-value ABI needs CUDA-Oxide's sm_100+ path."""

    typedefs = {item.name: item.underlying for item in model.typedefs}
    enums = {
        item.name: item.rust_underlying_type
        for item in model.enums
        if item.name
    }

    def is_small_by_value(
        type_: CAbiType, seen: set[str] | None = None
    ) -> bool:
        if type_.pointer_depth:
            return False
        base = type_.base_name
        if base in _LEGACY_NVVM_SMALL_C_TYPES:
            return True
        if base in enums:
            return enums[base] in _LEGACY_NVVM_SMALL_RUST_TYPES
        if base not in typedefs:
            return False
        seen = set() if seen is None else seen
        if base in seen:
            return False
        return is_small_by_value(typedefs[base], {*seen, base})

    required = []
    for function in model.functions:
        signature_types = [
            function.return_type,
            *(parameter.type_ for parameter in function.parameters),
        ]
        if any(is_small_by_value(type_) for type_ in signature_types):
            required.append(function.native_name)
    return required
