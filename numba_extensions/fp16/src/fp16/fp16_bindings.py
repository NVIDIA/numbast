# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from collections import defaultdict

from ast_canopy import parse_declarations_from_source

from numbast import bind_cxx_structs, bind_cxx_functions, MemoryShimWriter

from numba import types, config, cuda
from numba.core.datamodel.models import PrimitiveModel, StructModel

if cuda.get_current_device().compute_capability < (8, 0):
    import warnings

    # fp16 contains a forward decl ctor from bfloat16 that's defined in cuda_bf16 header
    # even compute capability doesn't support bfloat16 ops, we still need its decl to
    # pass typing.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import bf16
else:
    import bf16

CUDA_INCLUDE_PATH = config.CUDA_INCLUDE_PATH
COMPUTE_CAPABILITY = cuda.get_current_device().compute_capability

cuda_fp16 = os.path.join(CUDA_INCLUDE_PATH, "cuda_fp16.h")
cuda_fp16_hpp = os.path.join(CUDA_INCLUDE_PATH, "cuda_fp16.hpp")

cuda_bf16 = bf16.bf16_bindings.cuda_bf16

structs, functions, _, _, typedefs, enums = parse_declarations_from_source(
    cuda_fp16,
    [cuda_fp16, cuda_fp16_hpp],
    f"sm_{COMPUTE_CAPABILITY[0]}{COMPUTE_CAPABILITY[1]}",
    cudatoolkit_include_dir=CUDA_INCLUDE_PATH,
)

TYPE_SPECIALIZATION = {
    "__half_raw": types.Number,
    "__half": types.Number,
    "half": types.Number,
    "__nv_half": types.Number,
    "__nv_half_raw": types.Number,
    "nv_half": types.Number,
    "__half2_raw": types.Type,
    "__half2": types.Type,
    "half2": types.Type,
    "__nv_half2_raw": types.Type,
    "__nv_half2": types.Type,
    "nv_half2": types.Type,
}
DATA_MODEL_SPECIALIZATION = {
    "__half_raw": PrimitiveModel,
    "__half": PrimitiveModel,
    "half": PrimitiveModel,
    "__nv_half": PrimitiveModel,
    "__nv_half_raw": PrimitiveModel,
    "nv_half": PrimitiveModel,
    "__half2_raw": StructModel,
    "__half2": StructModel,
    "half2": StructModel,
    "__nv_half2_raw": StructModel,
    "__nv_half2": StructModel,
    "nv_half2": StructModel,
}

functions_to_ignore = {
    "atomicAdd"  # TODO: build bindings to numba.cuda.atomic, not as a standalone function
}

numba_struct_types = []
numba_functions = []

aliases = defaultdict(list)
for typedef in typedefs:
    aliases[typedef.underlying_name].append(typedef.name)

shim_writer = MemoryShimWriter(
    f'#include "{cuda_fp16}"\n' + f'#include "{cuda_bf16}"\n'
)

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
globals().update({"get_shims": shim_writer.links})

__all__ = list(  # noqa: F822
    set(s.__name__ for s in numba_struct_types)
    | set(f.__name__ for f in numba_functions)
    | set(typedef.name for typedef in typedefs)
) + ["get_shims"]
