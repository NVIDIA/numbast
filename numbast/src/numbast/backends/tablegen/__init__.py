# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TableGen (MLIR dialect) emission backend.

Emits MLIR dialect definitions rather than Numba bindings. Note the name
collision hazard: ``Backend: mlir`` selects the *numba-cuda-mlir* backend,
which still emits Numba typing and lowering registries. This backend emits
MLIR itself.

What is here reads a parsed header and renders the dialect: the type
vocabulary, the op definitions, and the ``.td`` files that hold them. Calling
into those ops -- the C access shim, the lowering, the C++ scaffolding around
them -- is not, so this backend is not yet reachable from the command line and
is not registered as a name one can select.
"""

from numbast.backends.tablegen.config import (
    OutParamPolicy,
    Resolver,
    ShimOptions,
    TableGenOptions,
    UnresolvedPolicy,
)
from numbast.backends.tablegen.facts import header_facts
from numbast.backends.tablegen.function import (
    TableGenFunctionsRenderer,
    classify_params,
    mnemonic_for,
)
from numbast.backends.tablegen.types import TypeMapper, llvm_token
from numbast.backends.tablegen.types_td import (
    TableGenTypesRenderer,
    render_dialect_td,
)
from numbast.backends.tablegen.typedefs import recover_typedefs

__all__ = [
    "OutParamPolicy",
    "Resolver",
    "ShimOptions",
    "TableGenConfig",
    "TableGenFunctionsRenderer",
    "TableGenOptions",
    "TableGenTypesRenderer",
    "TypeMapper",
    "UnresolvedPolicy",
    "classify_params",
    "header_facts",
    "llvm_token",
    "mnemonic_for",
    "recover_typedefs",
    "render_dialect_td",
]


def __getattr__(name):
    # TableGenConfig subclasses the core Config, which pulls in the Numba
    # stack; keep that out of the import path unless it is asked for.
    if name == "TableGenConfig":
        from numbast.backends.tablegen.config import TableGenConfig

        return TableGenConfig
    raise AttributeError(name)
