# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUDA-Oxide Rust type and identifier policy."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from numbast.cuda_oxide_binding_model import CudaOxideBindingPlan


_QUALIFIERS = re.compile(r"\b(?:const|volatile|restrict|__restrict__)\b")
_ARRAY_SUFFIX = re.compile(r"\[\s*([0-9]+)\s*\]\s*$")
_POINTER_TO_ARRAY = re.compile(
    r"^(?P<base>.+?)\(\s*(?P<pointers>(?:\*\s*"
    r"(?:(?:const|volatile|restrict|__restrict__)\s*)*)+)\)"
    r"(?P<arrays>(?:\s*\[\s*[0-9]+\s*\])+\s*)$"
)
_TAG_PREFIX = re.compile(r"^(?:struct|enum|union)\s+")
_RUST_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

RUST_KEYWORDS = frozenset(
    {
        "Self",
        "abstract",
        "as",
        "async",
        "await",
        "become",
        "box",
        "break",
        "const",
        "continue",
        "crate",
        "do",
        "dyn",
        "else",
        "enum",
        "extern",
        "false",
        "final",
        "fn",
        "for",
        "gen",
        "if",
        "impl",
        "in",
        "let",
        "loop",
        "macro",
        "macro_rules",
        "match",
        "mod",
        "move",
        "mut",
        "override",
        "priv",
        "pub",
        "ref",
        "return",
        "safe",
        "self",
        "static",
        "struct",
        "super",
        "trait",
        "true",
        "try",
        "type",
        "typeof",
        "union",
        "unsafe",
        "unsized",
        "use",
        "virtual",
        "where",
        "while",
        "yield",
    }
)

_RUST_NON_RAW_IDENTIFIERS = frozenset({"Self", "_", "crate", "self", "super"})

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
# boundaries. These aliases preserve the CUDA storage ABI without requiring a
# C++ bridge.
CUDA_ABI_ALIASES = {
    # Pre-Blackwell CUDA-Oxide uses a legacy NVVM dialect that cannot carry
    # half or sub-32-bit values at an extern boundary. The u16 storage spelling
    # keeps pointer-based APIs usable there; build_cuda_oxide_binding_plan
    # selects f16
    # for CUDA __half on the modern sm_100+ path, where by-value FFI is legal.
    "__half": ("u16", 2, 2),
    "half": ("u16", 2, 2),
    "__nv_bfloat16": ("u16", 2, 2),
    # CUDA gives double2 16-byte alignment. A Rust [f64; 2] is only 8-byte
    # aligned, so use an opaque 128-bit storage cell at the raw ABI boundary.
    "double2": ("[u128; 1]", 16, 16),
}


@dataclass(frozen=True)
class CudaOxideType:
    c_spelling: str
    base_name: str
    pointer_kinds: tuple[str, ...] = ()
    array_dimensions: tuple[int, ...] = ()

    @property
    def pointer_depth(self) -> int:
        return len(self.pointer_kinds)

    def manifest_dict(self) -> dict[str, Any]:
        return asdict(self)


def is_identifier(name: str) -> bool:
    return bool(_RUST_IDENTIFIER.fullmatch(name))


def rust_identifier(name: str) -> str:
    """Return a source-level Rust identifier for a known-valid identifier."""

    if not is_identifier(name):
        raise ValueError(f"Not a valid C/Rust identifier: {name!r}")
    if name in _RUST_NON_RAW_IDENTIFIERS:
        return f"{name}_"
    if name in RUST_KEYWORDS:
        return f"r#{name}"
    return name


def rust_parameter_name(name: str, index: int) -> str:
    candidate = name or f"arg{index}"
    candidate = re.sub(r"[^A-Za-z0-9_]", "_", candidate)
    if not candidate or candidate[0].isdigit():
        candidate = f"arg_{candidate}"
    return rust_identifier(candidate)


def parse_cuda_oxide_type(type_obj: Any) -> CudaOxideType:
    """Build a CUDA-Oxide type from an AST Canopy C type spelling."""

    if type_obj.is_left_reference() or type_obj.is_right_reference():
        raise ValueError(
            "C++ references are outside the supported CUDA-Oxide bindings"
        )

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
            "function/member pointers are outside the supported CUDA-Oxide bindings"
        )
    if "&" in parse_spelling:
        raise ValueError(
            "C++ references are outside the supported CUDA-Oxide bindings"
        )

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

    return CudaOxideType(
        c_spelling=spelling,
        base_name=base_name,
        pointer_kinds=pointer_kinds,
        array_dimensions=tuple(dimensions),
    )


class _TypedefType:
    """Small adapter allowing typedef spellings to use the common parser."""

    def __init__(self, name: str):
        self.name = name

    def is_left_reference(self):
        return False

    def is_right_reference(self):
        return False


def parse_cuda_oxide_type_spelling(spelling: str) -> CudaOxideType:
    """Build a CUDA-Oxide type from a configured C type spelling."""

    return parse_cuda_oxide_type(_TypedefType(spelling))


def cuda_abi_alias_for_arch(name: str, gpu_arch: str) -> tuple[str, int, int]:
    storage, size, alignment = CUDA_ABI_ALIASES[name]
    architecture = int(gpu_arch.split("_", 1)[1].split("a", 1)[0])
    if architecture >= 100 and name in {"__half", "half"}:
        storage = "f16"
    return storage, size, alignment


def rust_struct_storage(size: int, alignment: int) -> str:
    """Return an aligned Rust storage type for an opaque C record."""

    cells = {1: "u8", 2: "u16", 4: "u32", 8: "u64", 16: "u128"}
    cell = cells.get(alignment)
    if cell is None or size <= 0 or size % alignment:
        raise ValueError(
            f"cannot represent size={size}, alignment={alignment} as CUDA-Oxide storage"
        )
    return f"[{cell}; {size // alignment}]"


def render_rust_type(
    type_: CudaOxideType,
    plan: CudaOxideBindingPlan,
    typedef_decls: dict[str, Any] | None = None,
) -> str:
    """Render a CUDA-Oxide type as Rust source."""

    base = type_.base_name
    if base in PRIMITIVE_RUST_TYPES:
        rendered = PRIMITIVE_RUST_TYPES[base]
    elif (
        base in plan.cuda_aliases
        or any(item.name == base for item in plan.enums)
        or any(item.name == base for item in plan.structs)
        or any(item.name == base for item in plan.type_aliases)
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
