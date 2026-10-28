# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TableGen (MLIR dialect) emission backend.

Emits MLIR dialect definitions rather than Numba bindings. The registered
names keep the two apart: ``Backend: numba-cuda-mlir`` selects the backend
that emits Numba typing and lowering registries against an MLIR runtime, and
this one emits the MLIR dialect itself. Naming either of them ``mlir`` would
describe both.

What is here reads a parsed header and renders the dialect: the type
vocabulary, the op definitions, and the ``.td`` files that hold them. Calling
into those ops -- the C access shim, the lowering, the C++ scaffolding around
them -- is not, so this backend is not yet reachable from the command line and
is not registered as a name one can select.
"""

from numbast.backends.tablegen.config import (
    OutParamPolicy,
    TableGenConfig,
    TableGenOptions,
)
from numbast.backends.tablegen.facts import header_facts
from numbast.backends.tablegen.function import (
    TableGenFunctionsRenderer,
    classify_params,
    mnemonic_for,
)
from numbast.backends.tablegen.types import TypeMapper
from numbast.backends.tablegen.types_td import (
    TableGenTypesRenderer,
    render_dialect_td,
)
from numbast.backends.tablegen.typedefs import recover_typedefs

__all__ = [
    "OutParamPolicy",
    "TableGenConfig",
    "TableGenFunctionsRenderer",
    "TableGenOptions",
    "TableGenTypesRenderer",
    "TypeMapper",
    "classify_params",
    "header_facts",
    "mnemonic_for",
    "recover_typedefs",
    "render_dialect_td",
]
