# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from collections import defaultdict
from ast_canopy import (
    parse_declarations_from_source,
)

from numbast import (
    ShimWriter,
    bind_cxx_structs,
    bind_cxx_functions,
)

from numba import types, config, cuda
from numba.core.datamodel.models import PrimitiveModel, StructModel

CUDA_INCLUDE_PATH = config.CUDA_INCLUDE_PATH
COMPUTE_CAPABILITY = cuda.get_current_device().compute_capability

cuda_bf16 = os.path.join(CUDA_INCLUDE_PATH, "cuda_bf16.h")
cuda_bf16_hpp = os.path.join(CUDA_INCLUDE_PATH, "cuda_bf16.hpp")


structs, functions, _, _, typedefs, _ = parse_declarations_from_source(
    cuda_bf16,
    [cuda_bf16, cuda_bf16_hpp],
    f"sm_{COMPUTE_CAPABILITY[0]}{COMPUTE_CAPABILITY[1]}",
    cudatoolkit_include_dir=CUDA_INCLUDE_PATH,
)

TYPE_SPECIALIZATION = {
    "__nv_bfloat16_raw": types.Number,
    "__nv_bfloat16": types.Number,
    "__nv_bfloat162_raw": types.Type,
    "__nv_bfloat162": types.Type,
    "nv_bfloat16": types.Number,
    "nv_bfloat162": types.Type,
}
DATA_MODEL_SPECIALIZATION = {
    "__nv_bfloat16_raw": PrimitiveModel,
    "__nv_bfloat16": PrimitiveModel,
    "nv_bfloat16": PrimitiveModel,
    "__nv_bfloat162_raw": StructModel,
    "__nv_bfloat162": StructModel,
    "nv_bfloat162": StructModel,
}

functions_to_ignore = {
    "atomicAdd"  # TODO: build bindings to numba.cuda.atomic, not as a standalone function
}

numba_struct_types = []
numba_functions = []

aliases = defaultdict(list)
for typedef in typedefs:
    aliases[typedef.underlying_name].append(typedef.name)

shim_writer = ShimWriter("bf16_shim.cu", f'#include "{cuda_bf16}"\n')

numba_struct_types += bind_cxx_structs(
    shim_writer, structs, TYPE_SPECIALIZATION, DATA_MODEL_SPECIALIZATION, aliases
)

numba_functions += bind_cxx_functions(
    shim_writer, functions, exclude=functions_to_ignore
)

# Export
globals().update({s.__name__: s for s in numba_struct_types})
globals().update({f.__name__: f for f in numba_functions})
for underlying_name, names in aliases.items():
    for name in names:
        if name not in globals() and underlying_name in globals():
            globals()[name] = globals()[underlying_name]

__all__ = list(
    set(s.__name__ for s in numba_struct_types)
    | set(f.__name__ for f in numba_functions)
    | set(typedef.name for typedef in typedefs)
)
