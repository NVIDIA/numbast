# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Language-neutral C-device ABI model built from AST Canopy declarations."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from numbast.name_policy import (
    apply_prefix_removal,
    is_identifier,
    rust_identifier,
    rust_parameter_name,
)

_QUALIFIERS = re.compile(r"\b(?:const|volatile|restrict|__restrict__)\b")
_ARRAY_SUFFIX = re.compile(r"\[\s*([0-9]+)\s*\]\s*$")
_POINTER_TO_ARRAY = re.compile(
    r"^(?P<base>.+?)\(\s*(?P<pointers>(?:\*\s*"
    r"(?:(?:const|volatile|restrict|__restrict__)\s*)*)+)\)"
    r"(?P<arrays>(?:\s*\[\s*[0-9]+\s*\])+\s*)$"
)
_TAG_PREFIX = re.compile(r"^(?:struct|enum|union)\s+")
_INTEGER_LITERAL = re.compile(
    r"^[+-]?(?:0[xX][0-9A-Fa-f]+|0[bB][01]+|0[0-7]*|[0-9]+)(?:[uUlL]+)?$"
)


PRIMITIVE_RUST_TYPES = {
    "_Bool": "bool",
    "bool": "bool",
    "char": "i8",
    "double": "f64",
    "float": "f32",
    "int": "i32",
    "int8_t": "i8",
    "int16_t": "i16",
    "int32_t": "i32",
    "int64_t": "i64",
    "intptr_t": "isize",
    "long": "i64",
    "long long": "i64",
    "ptrdiff_t": "isize",
    "short": "i16",
    "signed char": "i8",
    "signed int": "i32",
    "signed long": "i64",
    "signed long long": "i64",
    "signed short": "i16",
    "size_t": "usize",
    "uint8_t": "u8",
    "uint16_t": "u16",
    "uint32_t": "u32",
    "uint64_t": "u64",
    "uintptr_t": "usize",
    "unsigned char": "u8",
    "unsigned int": "u32",
    "unsigned long": "u64",
    "unsigned long long": "u64",
    "unsigned short": "u16",
    "void": "core::ffi::c_void",
}

# CUDA-Oxide currently admits scalar and fixed-array pointees at device extern
# boundaries.  These aliases preserve the CUDA storage ABI without requiring a
# C++ bridge.
CUDA_ABI_ALIASES = {
    # Pre-Blackwell CUDA-Oxide uses a legacy NVVM dialect that cannot carry
    # half or sub-32-bit values at an extern boundary. The u16 storage spelling
    # keeps pointer-based APIs usable there; build_c_device_model selects f16
    # for CUDA __half on the modern sm_100+ path, where by-value FFI is legal.
    "__half": ("u16", 2, 2),
    "half": ("u16", 2, 2),
    "__nv_bfloat16": ("u16", 2, 2),
    # CUDA gives double2 16-byte alignment. A Rust [f64; 2] is only 8-byte
    # aligned, so use an opaque 128-bit storage cell at the raw ABI boundary.
    "double2": ("[u128; 1]", 16, 16),
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


def cuda_abi_alias_for_arch(name: str, gpu_arch: str) -> tuple[str, int, int]:
    storage, size, alignment = CUDA_ABI_ALIASES[name]
    architecture = int(gpu_arch.split("_", 1)[1].split("a", 1)[0])
    if architecture >= 100 and name in {"__half", "half"}:
        storage = "f16"
    return storage, size, alignment


class BindingModelError(ValueError):
    """Raised after collecting every actionable model diagnostic."""

    def __init__(self, diagnostics: list[str]):
        self.diagnostics = sorted(set(diagnostics))
        details = "\n".join(f"  - {item}" for item in self.diagnostics)
        super().__init__(f"CUDA-Oxide binding generation failed:\n{details}")


@dataclass(frozen=True)
class CAbiType:
    c_spelling: str
    base_name: str
    pointer_kinds: tuple[str, ...] = ()
    array_dimensions: tuple[int, ...] = ()

    @property
    def pointer_depth(self) -> int:
        return len(self.pointer_kinds)

    def manifest_dict(self) -> dict[str, Any]:
        return asdict(self)


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


def parse_c_abi_type(type_obj: Any) -> CAbiType:
    """Normalize the qualified C spelling reported by AST Canopy."""

    if type_obj.is_left_reference() or type_obj.is_right_reference():
        raise ValueError("C++ references are outside the Round 1 C ABI")

    spelling = " ".join(type_obj.name.strip().split())
    if not spelling:
        raise ValueError("empty type spelling")
    parse_spelling = spelling
    pointer_to_array = _POINTER_TO_ARRAY.fullmatch(parse_spelling)
    if pointer_to_array is not None:
        parse_spelling = (
            f"{pointer_to_array.group('base')} "
            f"{pointer_to_array.group('pointers')}"
            f"{pointer_to_array.group('arrays')}"
        )
    if "(" in parse_spelling or ")" in parse_spelling:
        raise ValueError(
            "function/member pointers are outside the Round 1 C ABI"
        )
    if "&" in parse_spelling:
        raise ValueError("C++ references are outside the Round 1 C ABI")

    dimensions = []
    array_source = parse_spelling
    while True:
        match = _ARRAY_SUFFIX.search(array_source)
        if match is None:
            break
        dimensions.insert(0, int(match.group(1)))
        array_source = array_source[: match.start()].rstrip()

    segments = array_source.split("*")
    base_segment = segments[0].strip()
    pointer_kinds = tuple(
        "const" if re.search(r"\bconst\b", segment) else "mut"
        for segment in segments[:-1]
    )
    base_name = _QUALIFIERS.sub("", base_segment)
    base_name = _TAG_PREFIX.sub("", " ".join(base_name.split()))
    if not base_name:
        raise ValueError(f"unable to find a base type in {spelling!r}")

    return CAbiType(
        c_spelling=spelling,
        base_name=base_name,
        pointer_kinds=pointer_kinds,
        array_dimensions=tuple(dimensions),
    )


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


def _record_storage(size: int, alignment: int) -> str:
    cells = {1: "u8", 2: "u16", 4: "u32", 8: "u64", 16: "u128"}
    cell = cells.get(alignment)
    if cell is None or size <= 0 or size % alignment:
        raise ValueError(
            f"cannot represent size={size}, alignment={alignment} as CUDA-Oxide storage"
        )
    return f"[{cell}; {size // alignment}]"


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
            underlying = parse_c_abi_type(
                _TypedefType(typedefs[base].underlying_name)
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


class _TypedefType:
    """Small adapter allowing typedef spellings to use the common parser."""

    def __init__(self, name: str):
        self.name = name

    def is_left_reference(self):
        return False

    def is_right_reference(self):
        return False


def parse_c_type_spelling(spelling: str) -> CAbiType:
    """Parse a configured C type spelling with the AST type normalization."""

    return parse_c_abi_type(_TypedefType(spelling))


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
                underlying = parse_c_abi_type(
                    _TypedefType(typedef_decls[base].underlying_name)
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
                storage = _record_storage(
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


def rust_type(
    type_: CAbiType,
    model: BindingModel,
    typedef_decls: dict[str, Any] | None = None,
) -> str:
    """Render a normalized C ABI type as a CUDA-Oxide-compatible Rust type."""

    base = type_.base_name
    if base in PRIMITIVE_RUST_TYPES:
        rendered = PRIMITIVE_RUST_TYPES[base]
    elif (
        base in model.cuda_aliases
        or any(item.name == base for item in model.enums)
        or any(item.name == base for item in model.records)
        or any(item.name == base for item in model.typedefs)
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
