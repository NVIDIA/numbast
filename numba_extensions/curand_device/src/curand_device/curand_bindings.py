# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from collections import defaultdict
import logging

from ast_canopy import parse_declarations_from_source
from numbast import (
    bind_cxx_enum,
    bind_cxx_structs,
    bind_cxx_functions,
    MemoryShimWriter,
)

from numba import types, config, cuda
from numba.core.datamodel.models import StructModel

from curand_device.curand_states import make_curand_states

logger = logging.getLogger("curand")

config.CUDA_USE_NVIDIA_BINDING = True

CUDA_INCLUDE_PATH = config.CUDA_INCLUDE_PATH
COMPUTE_CAPABILITY = cuda.get_current_device().compute_capability

curand_h = os.path.join(CUDA_INCLUDE_PATH, "curand.h")
curand_kernel_h = os.path.join(CUDA_INCLUDE_PATH, "curand_kernel.h")
curand_headers = [
    "curand_discrete2.h",
    "curand_discrete.h",
    "curand_globals.h",
    "curand.h",
    "curand_kernel.h",
    "curand_lognormal.h",
    "curand_mrg32k3a.h",
    "curand_mtgp32dc_p_11213.h",
    "curand_mtgp32.h",
    "curand_mtgp32_host.h",
    "curand_mtgp32_kernel.h",
    "curand_normal.h",
    "curand_normal_static.h",
    "curand_philox4x32_x.h",
    "curand_poisson.h",
    "curand_precalc.h",
    "curand_uniform.h",
]

curand_files = [os.path.join(CUDA_INCLUDE_PATH, f) for f in curand_headers]

curand_files += [os.path.normpath(p) for p in curand_files]

# Mysteriously, clang AST sometimes fails to stick to the one cuda include path
# for source range and instead uses the unwrapped path behind the symlink. E.g.
# /usr/local/cuda-12.3/include/curand_uniform.h instead of
# /usr/local/cuda/include/curand_uniform.h
with open(os.path.join(CUDA_INCLUDE_PATH, "cuda.h"), "r") as f:
    cuda_header_version_macro = [
        line for line in f if line.startswith("#define CUDA_VERSION")
    ]
    try:
        cuda_header_version = int(
            cuda_header_version_macro[0].split()[-1]
        )  # e.g. 12030
        major = cuda_header_version // 1000  # e.g. 12
        minor = (cuda_header_version % 1000) // 10  # e.g. 3

        curand_files += [
            os.path.join(f"/usr/local/cuda-{major}.{minor}/include/", f)
            for f in curand_headers
        ]
    except Exception as e:
        logger.warn(
            f"Failed to parse CUDA version from cuda.h, some functionality in curand may not be properly bound. {e}"
        )


curand_files = [h for h in curand_files if os.path.exists(h)]


structs, functions, _, _, typedefs, enums = parse_declarations_from_source(
    curand_kernel_h,
    curand_files,
    f"sm_{COMPUTE_CAPABILITY[0]}{COMPUTE_CAPABILITY[1]}",
    cudatoolkit_include_dir=CUDA_INCLUDE_PATH,
)


TYPE_SPECIALIZATION = {
    "curandDistributionShift_st": types.Type,
    "curandHistogramM2_st": types.Type,
    "curandDistributionM2Shift_st": types.Type,
    "curandDiscreteDistribution_st": types.Type,
    "curandStatePhilox4_32_10": types.Type,
    "normal_args_st": types.Type,
    "normal_args_double_st": types.Type,
    "curandStateTest": types.Type,
    "curandStateXORWOW": types.Type,
    "curandStateMRG32k3a": types.Type,
    "curandStateSobol32": types.Type,
    "curandStateScrambledSobol32": types.Type,
    "curandStateSobol64": types.Type,
    "curandStateScrambledSobol64": types.Type,
    "curandStateMtgp32": types.Type,
    "mtgp32_params_fast": types.Type,
    "mtgp32_kernel_params": types.Type,
}
DATA_MODEL_SPECIALIZATION = {
    "curandDistributionShift_st": StructModel,
    "curandHistogramM2_st": StructModel,
    "curandDistributionM2Shift_st": StructModel,
    "curandDiscreteDistribution_st": StructModel,
    "curandStatePhilox4_32_10": StructModel,
    "normal_args_st": StructModel,
    "normal_args_double_st": StructModel,
    "curandStateTest": StructModel,
    "curandStateXORWOW": StructModel,
    "curandStateMRG32k3a": StructModel,
    "curandStateSobol32": StructModel,
    "curandStateScrambledSobol32": StructModel,
    "curandStateSobol64": StructModel,
    "curandStateScrambledSobol64": StructModel,
    "curandStateMtgp32": StructModel,
    "mtgp32_params_fast": StructModel,
    "mtgp32_kernel_params": StructModel,
}

functions_to_ignore = {
    "_skipahead_scratch",
    "_skipahead_sequence_scratch",
    "__curand_generate_skipahead_matrix_xor",
    "_skipahead_inplace",
    "_skipahead_sequence_inplace",
}

numba_struct_types = []
numba_functions = []
numba_enums = []

aliases = defaultdict(list)
for typedef in typedefs:
    aliases[typedef.underlying_name].append(typedef.name)

shim_writer = MemoryShimWriter(
    f'#include "{curand_h}"\n' + f'#include "{curand_kernel_h}"\n'
)

# Enum type creation needs to happen before function and struct binding.
for e in enums:
    E = bind_cxx_enum(e)
    numba_enums.append(E)


numba_struct_types += bind_cxx_structs(
    shim_writer, structs, TYPE_SPECIALIZATION, DATA_MODEL_SPECIALIZATION, aliases
)

numba_functions += bind_cxx_functions(
    shim_writer, functions, exclude=functions_to_ignore
)


curandStates = [
    "curandStateTest",
    "curandStateXORWOW",
    "curandStateMRG32k3a",
    "curandStateSobol32",
    "curandStateScrambledSobol32",
    "curandStateSobol64",
    "curandStateScrambledSobol64",
    "curandStatePhilox4_32_10",
    # "curandStateMtgp32", # Require additional type parsing of mtgp32_params_fast and mtgp32_kernel_params
]

numpy_curand_states = []
states_arg_handlers = []

for state in curandStates:
    decl = next(s for s in structs if s.name == state)
    states_obj, arg_handler = make_curand_states(decl)
    numpy_curand_states.append(states_obj)
    states_arg_handlers.append(arg_handler)


# Export
globals().update({s.__name__: s for s in numba_struct_types})
globals().update({f.__name__: f for f in numba_functions})
globals().update({e.__name__: e for e in numba_enums})
globals().update({s.__name__: s for s in numpy_curand_states})
globals().update({"states_arg_handlers": states_arg_handlers})
for underlying_name, names in aliases.items():
    for name in names:
        if name not in globals() and underlying_name in globals():
            globals()[name] = globals()[underlying_name]
globals().update({"get_shims": shim_writer.links})

__all__ = list(  # noqa: F822
    set(s.__name__ for s in numba_struct_types)
    | set(f.__name__ for f in numba_functions)
    | set(typedef.name for typedef in typedefs)
    | set(e.__name__ for e in numba_enums)
    | set(s.__name__ for s in numpy_curand_states)
    | {"states_arg_handlers"}
) + ["get_shims"]
