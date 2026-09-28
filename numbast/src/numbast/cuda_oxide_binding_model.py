# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUDA-Oxide Rust binding plan built from AST Canopy declarations."""

from __future__ import annotations

from collections.abc import Iterable
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
    rust_identifier,
    rust_parameter_name,
    rust_struct_storage,
)

_EXECUTION_SPACE_NAMES = {
    "execution_space.undefined": "undefined",
    "execution_space.host": "host",
    "execution_space.device": "device",
    "execution_space.host_device": "host_device",
    "execution_space.global_": "global",
}
_CUDA_OXIDE_RESERVED_PREFIX = "cuda_oxide_"


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

    def render_rust_type(
        self,
        type_: CudaOxideType,
        typedef_decls: dict[str, Any] | None = None,
    ) -> str:
        """Render a type using the names defined by this binding plan."""

        base = type_.base_name
        if base in PRIMITIVE_RUST_TYPES:
            rendered = PRIMITIVE_RUST_TYPES[base]
        elif (
            base in self.cuda_aliases
            or any(item.name == base for item in self.enums)
            or any(item.name == base for item in self.structs)
            or any(item.name == base for item in self.type_aliases)
            or typedef_decls
            and base in typedef_decls
        ):
            rendered = rust_identifier(base)
        else:
            raise ValueError(f"unsupported C ABI type {base!r}")

        for dimension in reversed(type_.array_dimensions):
            rendered = f"[{rendered}; {dimension}]"
        for pointer_kind in type_.pointer_kinds:
            rendered = f"*{pointer_kind} {rendered}"
        return rendered

    def _add_exclusion(self, kind: str, name: str, reason: str):
        self.exclusions.append({"kind": kind, "name": name, "reason": reason})

    def _select_functions(
        self, functions: Iterable[Any], config: Any
    ) -> list[tuple[Any, str]]:
        selected = []
        for function in functions:
            space = _EXECUTION_SPACE_NAMES[str(function.exec_space)]
            if function.name in config.exclude_functions:
                self._add_exclusion("function", function.name, "configured")
                continue
            if config.skip_prefix and function.name.startswith(
                config.skip_prefix
            ):
                self._add_exclusion("function", function.name, "skip-prefix")
                continue
            if space not in {"device", "host_device"}:
                self._add_exclusion(
                    "function", function.name, f"execution-space:{space}"
                )
                continue
            selected.append((function, space))
        return selected

    def _require_linkage_metadata(self, functions: Iterable[Any]):
        if any(
            getattr(function, "is_c_linkage", None) is None
            for function in functions
        ):
            raise CudaOxideBindingError(
                [
                    (
                        "AST Canopy does not expose Function.is_c_linkage; install "
                        "the AST Canopy version shipped with this Numbast checkout"
                    )
                ]
            )

    def _validate_type(
        self,
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
            if (
                base == "double2"
                and not type_.pointer_depth
                and not behind_pointer
            ):
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
                diagnostics.append(
                    f"{context}: cyclic typedef involving {base!r}"
                )
                return
            seen.add(base)
            try:
                underlying = parse_cuda_oxide_type_spelling(
                    typedefs[base].underlying_name
                )
            except ValueError as error:
                diagnostics.append(f"{context}: typedef {base!r}: {error}")
                return
            self._validate_type(
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

    def _parse_and_validate_type(
        self,
        type_: Any,
        typedefs: dict[str, Any],
        enums: dict[str, Any],
        records: dict[str, Any],
        diagnostics: list[str],
        context: str,
    ) -> CudaOxideType | None:
        try:
            parsed = parse_cuda_oxide_type(type_)
        except ValueError as error:
            diagnostics.append(f"{context}: {error}")
            return None
        self._validate_type(
            parsed,
            typedefs,
            enums,
            records,
            diagnostics,
            context,
        )
        return parsed

    def _add_template_exclusions(self, declarations: Any):
        for template in declarations.function_templates:
            self._add_exclusion(
                "function-template",
                template.function.name,
                "round-1-c-api-only",
            )
        for template in declarations.class_templates:
            self._add_exclusion(
                "class-template",
                template.record.name,
                "round-1-c-api-only",
            )

    def _add_enum(
        self,
        declaration: Any,
        typedefs: dict[str, Any],
        diagnostics: list[str],
        context: str,
    ):
        try:
            underlying = parse_cuda_oxide_type(declaration.underlying_type)
            rust_underlying = self.render_rust_type(underlying, typedefs)
        except ValueError as error:
            diagnostics.append(f"{context}: {error}")
            return
        self.enums.append(
            CudaOxideEnum(
                declaration.name or "",
                rust_underlying,
                tuple(
                    zip(declaration.enumerators, declaration.enumerator_values)
                ),
            )
        )

    def _add_remaining_enums(
        self,
        declarations: Any,
        typedefs: dict[str, Any],
        diagnostics: list[str],
    ):
        # Named and anonymous enum values are useful C API constants even when
        # the enum itself is not present in a selected signature.
        known_names = {item.name for item in self.enums}
        for declaration in declarations.enums:
            if declaration.name in known_names:
                continue
            name = declaration.name or "<anonymous>"
            self._add_enum(declaration, typedefs, diagnostics, f"enum {name!r}")

    def _sort(self):
        self.functions.sort(key=lambda item: item.native_name)
        self.enums.sort(key=lambda item: (item.name, item.enumerators))
        self.structs.sort(key=lambda item: item.name)
        self.type_aliases.sort(key=lambda item: item.name)
        self.exclusions.sort(
            key=lambda item: (item["kind"], item["name"], item["reason"])
        )

    @classmethod
    def from_declarations(
        cls, declarations: Any, config: Any
    ) -> CudaOxideBindingPlan:
        """Select declarations and build the strict Round 1 binding plan."""

        plan = cls()
        diagnostics: list[str] = []
        typedef_decls = {item.name: item for item in declarations.typedefs}
        enum_decls = {
            item.name: item for item in declarations.enums if item.name
        }
        record_decls = {
            item.name: item
            for item in declarations.structs
            if item.name not in config.exclude_structs
        }
        prefix_removal = config.api_prefix_removal.get("Function", [])

        # Apply configured exclusions and skip prefixes, then retain only
        # device and host-device functions as Round 1 binding candidates.
        candidates = plan._select_functions(declarations.functions, config)
        plan._require_linkage_metadata(
            function for function, _space in candidates
        )

        seen_native: dict[str, CudaOxideFunction] = {}
        seen_public: dict[str, str] = {}
        for function, space in candidates:
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

            # Normalize and validate the return type against the supported
            # CUDA-Oxide ABI surface.
            return_context = f"function {function.name!r} return type"
            return_type = plan._parse_and_validate_type(
                function.return_type,
                typedef_decls,
                enum_decls,
                record_decls,
                diagnostics,
                return_context,
            )
            if return_type is None:
                continue
            if return_type.array_dimensions:
                diagnostics.append(
                    f"function {function.name!r}: array return types are unsupported"
                )

            # Normalize parameter types and turn C parameter names into
            # Rust-compatible identifiers.
            parameters = []
            used_parameter_names = set()
            for index, parameter in enumerate(function.params):
                parameter_context = (
                    f"function {function.name!r} parameter {index}"
                )
                type_ = plan._parse_and_validate_type(
                    parameter.type_,
                    typedef_decls,
                    enum_decls,
                    record_decls,
                    diagnostics,
                    parameter_context,
                )
                if type_ is None:
                    continue
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
            # The same native C symbol is either a duplicate declaration or an
            # ABI conflict; C linkage does not support overloads.
            previous = seen_native.get(bound.native_name)
            if previous is not None:
                if previous.signature_key == bound.signature_key:
                    plan._add_exclusion(
                        "function", function.name, "duplicate-declaration"
                    )
                else:
                    diagnostics.append(
                        f"function {function.name!r}: C symbol has conflicting signatures"
                    )
                continue
            # Prefix removal must not map two native C symbols to the same
            # public Rust identifier.
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

        # Native extern declarations and public aliases share the Rust value
        # namespace. Check aliases against every native name after collection
        # so the result does not depend on declaration order. This restriction
        # belongs to the current flat renderer; a renderer that keeps native
        # externs in a private `ffi` module could allow these names to overlap.
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

        plan._add_template_exclusions(declarations)

        # Collect declarations for every base type referenced by function
        # returns and parameters. Typedefs enqueue their underlying base types
        # so the worklist covers the complete transitive dependency chain.
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
            if base in enum_decls and all(
                item.name != base for item in plan.enums
            ):
                declaration = enum_decls[base]
                plan._add_enum(
                    declaration, typedef_decls, diagnostics, f"enum {base!r}"
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

        plan._add_remaining_enums(declarations, typedef_decls, diagnostics)

        plan._sort()
        if diagnostics:
            raise CudaOxideBindingError(diagnostics)
        return plan
