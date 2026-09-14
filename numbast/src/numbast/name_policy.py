# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Target-neutral and Rust-specific public-name policy."""

from __future__ import annotations

import re

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
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def apply_prefix_removal(name: str, prefixes: list[str]) -> str:
    """Remove the first matching prefix, preserving configured order."""

    for prefix in prefixes:
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def is_identifier(name: str) -> bool:
    return bool(_IDENTIFIER.fullmatch(name))


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
