# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import os
import math

import numpy as np

from numba import cuda, types
from numba.cuda import cuda_paths
from numba.core.datamodel.models import PrimitiveModel, StructModel

from ast_canopy import parse_declarations_from_source
from numbast.static.struct import StaticStructsRenderer
from numbast.static.function import (
    StaticFunctionsRenderer,
    clear_function_apis_registry,
)
from numbast.tools.static_binding_generator import _typedef_to_aliases
from numbast.static.typedef import render_aliases
from numbast.static.renderer import (
    clear_base_renderer_cache,
    get_pynvjitlink_guard,
    get_shim,
    get_rendered_imports,
)
from numbast.static.types import reset_types

include_path_tuple = cuda_paths.get_cuda_paths()["include_dir"]
if include_path_tuple is None:
    raise RuntimeError("No CUDA installation found!")
include_path = include_path_tuple.info
COMPUTE_CAPABILITY = cuda.get_current_device().compute_capability


@pytest.fixture(scope="module")
def bfloat16():
    reset_types()
    clear_base_renderer_cache()
    clear_function_apis_registry()

    cuda_bf16 = os.path.join(include_path, "cuda_bf16.h")
    cuda_bf16_hpp = os.path.join(include_path, "cuda_bf16.hpp")

    decls = parse_declarations_from_source(
        cuda_bf16,
        [cuda_bf16, cuda_bf16_hpp],
        f"sm_{COMPUTE_CAPABILITY[0]}{COMPUTE_CAPABILITY[1]}",
        cudatoolkit_include_dir=include_path,
    )
    structs = decls.structs
    functions = decls.functions
    typedefs = decls.typedefs

    specs = {
        "__nv_bfloat16_raw": (types.Number, PrimitiveModel, cuda_bf16),
        "__nv_bfloat16": (types.Number, PrimitiveModel, cuda_bf16),
        "__nv_bfloat162_raw": (types.Type, StructModel, cuda_bf16),
        "__nv_bfloat162": (types.Type, StructModel, cuda_bf16),
        "nv_bfloat16": (types.Number, PrimitiveModel, cuda_bf16),
        "nv_bfloat162": (types.Type, StructModel, cuda_bf16),
    }

    aliases = _typedef_to_aliases(typedefs)

    typedef_bindings = render_aliases(aliases)

    SSR = StaticStructsRenderer(structs, specs, default_header=cuda_bf16)
    SFR = StaticFunctionsRenderer(functions, cuda_bf16)

    struct_bindings = SSR.render_as_str(
        require_pynvjitlink=False, with_imports=False, with_shim_stream=False
    )

    function_bindings = SFR.render_as_str(
        require_pynvjitlink=False, with_imports=False, with_shim_stream=False
    )

    prefix_str = get_pynvjitlink_guard()
    include = f"'#include <{cuda_bf16}>'"
    shim_stream_str = get_shim(include)
    imports_str = get_rendered_imports()

    bindings = f"""
{prefix_str}
{imports_str}
{shim_stream_str}
{struct_bindings}
{function_bindings}
{typedef_bindings}
"""

    globals = {}
    exec(bindings, globals)

    public_apis = ["nv_bfloat16", "nv_bfloat16", "hsin"]
    assert all(public_api in globals for public_api in public_apis)

    return {k: globals[k] for k in public_apis}


def test_bfloat16(bfloat16):
    bf16 = bfloat16["nv_bfloat16"]
    hsin = bfloat16["hsin"]

    @cuda.jit
    def kernel(arr):
        three = bf16(1.0) + bf16(2.0)
        sin_three = hsin(three)
        arr[0] = types.float32(three)
        arr[1] = types.float32(sin_three)

    arr = np.array([0, 0], dtype="f8")

    kernel[1, 1](arr)

    np.testing.assert_allclose(arr, [3.0, math.sin(3.0)], atol=1e-3)
