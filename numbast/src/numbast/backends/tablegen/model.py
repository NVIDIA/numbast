# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The dialect a recipe implies, as data.

Analysis resolves curated entry points into these; rendering turns them into
TableGen text. Nothing here knows ``.td`` syntax and nothing here reads the
recipe -- by the time a model exists, every configurable decision has already
been made.

The split matters most for the type vocabulary. Which types a dialect
declares is not a property of the header: it is whatever the curated
signatures reach, transitively, under the recipe's out-parameter policy. That
set is discovered while resolving the ops, so it can only be written down once
-- here. Everything downstream reads :attr:`DialectModel.types` rather than
re-deriving it, the lowering descriptor above all, whose ``TypeConverter``
table has to list exactly the types the ops reference.
"""

from dataclasses import dataclass

__all__ = [
    "DialectModel",
    "DialectSpec",
    "OpSpec",
    "Operand",
    "TypeSpec",
]


@dataclass(frozen=True)
class Operand:
    """One entry in an op's ``arguments`` or ``results`` list.

    ``name`` is already legal and unique within its op: reserved words are
    escaped during analysis, because whether a name collides depends on the
    other names in the same op and rendering sees one at a time.
    """

    type_ref: str
    name: str


@dataclass(frozen=True)
class OpSpec:
    """One ``def {Prefix}{Record}Op`` definition.

    Out-parameters have already been lifted into ``results`` and the status
    return appended, so the operand/result split here is the dialect's, not
    the C function's.
    """

    record: str
    mnemonic: str
    operands: tuple[Operand, ...]
    results: tuple[Operand, ...]
    shim: str


@dataclass(frozen=True)
class TypeSpec:
    """One ``def {Prefix}{Record}Type`` definition.

    ``kind`` records where the type came from -- ``"status"``, ``"ptr"``,
    ``"handle"`` or ``"record"`` -- which rendering ignores but diagnostics
    need: a mnemonic collision has to name the records behind it, and only
    the modelled records can collide.
    """

    record: str
    mnemonic: str
    summary: str
    kind: str


@dataclass(frozen=True)
class DialectSpec:
    """The dialect definition, and the names every other file is keyed on."""

    name: str
    cpp_class_prefix: str
    summary: str
    description: str

    @property
    def guard(self) -> str:
        return self.name.upper()

    @property
    def cpp_namespace(self) -> str:
        """Derived rather than configured.

        The generated C++ has to agree with it, and two spellings of one fact
        can disagree.
        """
        return f"::mlir::{self.name}"


@dataclass(frozen=True)
class DialectModel:
    """A whole dialect: its definition, its type vocabulary, and its ops."""

    dialect: DialectSpec
    types: tuple[TypeSpec, ...]
    ops: tuple[OpSpec, ...]
