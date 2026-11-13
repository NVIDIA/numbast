# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TableGen (MLIR dialect) emission backend.

Emits MLIR dialect definitions rather than Numba bindings. The registered
names keep the two apart: ``Backend: numba-cuda-mlir`` selects the backend
that emits Numba typing and lowering registries against an MLIR runtime, and
this one emits the MLIR dialect itself. Naming either of them ``mlir`` would
describe both.

One module per artifact: ``dialect_td`` emits the dialect definition,
``types_td`` the type vocabulary, and ``ops_td`` one op per curated entry
point. ``type_mapper`` is the mapper both renderers consume rather than a
fourth artifact -- the ops renderer *references* types while rendering, which
is what populates the registries the types renderer then declares.

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
from numbast.backends.tablegen.ops_td import (
    TableGenFunctionsRenderer,
    classify_params,
    mnemonic_for,
)
from numbast.backends.tablegen.type_mapper import TypeMapper
from numbast.backends.tablegen.types_td import TableGenTypesRenderer
from numbast.backends.tablegen.typedefs import typedef_facts

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
    "typedef_facts",
    "render_dialect_td",
]
