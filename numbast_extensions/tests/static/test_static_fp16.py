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
def float16():
    reset_types()
    clear_base_renderer_cache()
    clear_function_apis_registry()

    cuda_fp16 = os.path.join(include_path, "cuda_fp16.h")
    cuda_fp16_hpp = os.path.join(include_path, "cuda_fp16.hpp")

    decls = parse_declarations_from_source(
        cuda_fp16,
        [cuda_fp16, cuda_fp16_hpp],
        f"sm_{COMPUTE_CAPABILITY[0]}{COMPUTE_CAPABILITY[1]}",
        cudatoolkit_include_dir=include_path,
    )
    structs = decls.structs
    functions = decls.functions
    typedefs = decls.typedefs

    specs = {
        "__half_raw": (types.Number, PrimitiveModel, cuda_fp16),
        "__half": (types.Number, PrimitiveModel, cuda_fp16),
        "half": (types.Number, PrimitiveModel, cuda_fp16),
        "__nv_half": (types.Number, PrimitiveModel, cuda_fp16),
        "__nv_half_raw": (types.Number, PrimitiveModel, cuda_fp16),
        "nv_half": (types.Number, PrimitiveModel, cuda_fp16),
        "__half2_raw": (types.Type, StructModel, cuda_fp16),
        "__half2": (types.Type, StructModel, cuda_fp16),
        "half2": (types.Type, StructModel, cuda_fp16),
        "__nv_half2_raw": (types.Type, StructModel, cuda_fp16),
        "__nv_half2": (types.Type, StructModel, cuda_fp16),
        "nv_half2": (types.Type, StructModel, cuda_fp16),
    }

    aliases = _typedef_to_aliases(typedefs)
    typedef_bindings = render_aliases(aliases)

    SSR = StaticStructsRenderer(structs, specs, default_header=cuda_fp16)
    SFR = StaticFunctionsRenderer(functions, cuda_fp16)

    struct_bindings = SSR.render_as_str(
        require_pynvjitlink=False, with_imports=False, with_shim_stream=False
    )

    function_bindings = SFR.render_as_str(
        require_pynvjitlink=False, with_imports=False, with_shim_stream=False
    )

    prefix_str = get_pynvjitlink_guard()
    include = f"'#include <{cuda_fp16}>'"
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

    public_apis = ["half", "half2", "hsin"]
    assert all(public_api in globals for public_api in public_apis)

    return {k: globals[k] for k in public_apis}


def test_float16(float16):
    fp16 = float16["half"]
    hsin = float16["hsin"]

    @cuda.jit
    def kernel(arr):
        three = fp16(1.0) + fp16(2.0)
        sin_three = hsin(three)
        arr[0] = types.float32(three)
        arr[1] = types.float32(sin_three)

    arr = np.array([0, 0], dtype="f8")

    kernel[1, 1](arr)

    np.testing.assert_allclose(arr, [3.0, math.sin(3.0)], atol=1e-5)
