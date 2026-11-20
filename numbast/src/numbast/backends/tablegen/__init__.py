# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TableGen (MLIR dialect) emission backend.

Emits MLIR dialect definitions rather than Numba bindings. The registered
names keep the two apart: ``Backend: numba-cuda-mlir`` selects the backend
that emits Numba typing and lowering registries against an MLIR runtime, and
this one emits the MLIR dialect itself. Naming either of them ``mlir`` would
describe both.

Two phases. :mod:`plan` resolves the curated entry points into a
:class:`~numbast.backends.tablegen.model.DialectModel` -- every decision, no
syntax. Then one renderer per artifact turns that model into text:
``dialect_td`` the dialect definition, ``types_td`` the type vocabulary,
``ops_td`` the ops. The renderers are pure functions of the model and of each
other's nothing, so the order they run in does not matter and neither does
how many times.

Those three files are the vocabulary, not a buildable dialect. The C access
shim the ops lower to, the lowering descriptor, and the C++ and CMake
scaffolding that turns all of it into a loadable shared object are later
slices, so this backend is not yet reachable from the command line and is not
registered as a name one can select.
"""

from numbast.backends.tablegen.config import (
    OutParamPolicy,
    TableGenConfig,
    TableGenOptions,
)
from numbast.backends.tablegen.dialect_td import render_dialect_td
from numbast.backends.tablegen.facts import header_facts
from numbast.backends.tablegen.model import (
    DialectModel,
    DialectSpec,
    OpSpec,
    Operand,
    TypeSpec,
)
from numbast.backends.tablegen.ops_td import render_ops_td
from numbast.backends.tablegen.plan import (
    DuplicateTypeMnemonicError,
    classify_params,
    mnemonic_for,
    plan_dialect,
)
from numbast.backends.tablegen.type_mapper import TypeMapper
from numbast.backends.tablegen.typedefs import typedef_facts
from numbast.backends.tablegen.types_td import render_types_td

__all__ = [
    "DialectModel",
    "DialectSpec",
    "DuplicateTypeMnemonicError",
    "OpSpec",
    "Operand",
    "OutParamPolicy",
    "TableGenConfig",
    "TableGenOptions",
    "TypeMapper",
    "TypeSpec",
    "classify_params",
    "header_facts",
    "mnemonic_for",
    "plan_dialect",
    "render_dialect_files",
    "render_dialect_td",
    "render_ops_td",
    "render_types_td",
    "typedef_facts",
]


def render_dialect_files(model: DialectModel) -> dict:
    """The three ``.td`` files a dialect is made of, by file name.

    The names are the dialect's C++ class prefix plus the artifact, which is
    what the include paths the generated files use already assume.
    """
    dialect = model.dialect
    prefix = dialect.cpp_class_prefix
    return {
        f"{prefix}Dialect.td": render_dialect_td(dialect),
        f"{prefix}Types.td": render_types_td(model.types, dialect),
        f"{prefix}Ops.td": render_ops_td(model.ops, dialect),
    }
