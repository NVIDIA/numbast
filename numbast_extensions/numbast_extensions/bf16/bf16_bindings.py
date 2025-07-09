# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from collections import defaultdict
from ast_canopy import (
    parse_declarations_from_source,
)

from numbast import (
    MemoryShimWriter,
    bind_cxx_structs,
    bind_cxx_functions,
)

from numba import types, cuda
from numba.cuda import cuda_paths
from numba.core.datamodel.models import PrimitiveModel, StructModel

if cuda.get_current_device().compute_capability < (8, 0):
    import warnings

    warnings.warn(
        "bf16 bindings are only supported on compute capability 8.0 and later, "
        "most bf16 features may not be available.",
    )

COMPUTE_CAPABILITY = cuda.get_current_device().compute_capability
include_path_tuple = cuda_paths.get_cuda_paths()["include_dir"]
if include_path_tuple is None:
    raise RuntimeError("No CUDA installation found!")
include_path = include_path_tuple.info

cuda_bf16 = os.path.join(include_path, "cuda_bf16.h")
cuda_bf16_hpp = os.path.join(include_path, "cuda_bf16.hpp")


decls = parse_declarations_from_source(
    cuda_bf16,
    [cuda_bf16, cuda_bf16_hpp],
    f"sm_{COMPUTE_CAPABILITY[0]}{COMPUTE_CAPABILITY[1]}",
    cudatoolkit_include_dir=include_path,
)
structs, functions, typedefs, enums = (
    decls.structs,
    decls.functions,
    decls.typedefs,
    decls.enums,
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

# WAR for NVBug 4549708: ABI mismatching between the calling convention of the
# bf16 definition against the linked library, causing a unspecified launch failure.
pretext = f"""#define __FORCE_INCLUDE_CUDA_FP16_HPP_FROM_FP16_H__
#define __FORCE_INCLUDE_CUDA_BF16_HPP_FROM_BF16_H__
#include "{cuda_bf16}"
"""

shim_writer = MemoryShimWriter(pretext)

numba_struct_types += bind_cxx_structs(
    shim_writer,
    structs,
    TYPE_SPECIALIZATION,
    DATA_MODEL_SPECIALIZATION,
    aliases,
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
globals().update({"get_shims": shim_writer.links})

__all__ = list(  # noqa: F822
    set(s.__name__ for s in numba_struct_types)
    | set(f.__name__ for f in numba_functions)
    | set(typedef.name for typedef in typedefs)
) + ["get_shims"]
