# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import os
from functools import partial
import math

import numpy as np

from numba import cuda, config, types
from numba.core.datamodel.models import PrimitiveModel, StructModel

from ast_canopy import parse_declarations_from_source
from numbast.static.struct import StaticStructsRenderer
from numbast.static.function import StaticFunctionsRenderer
from numbast.tools.static_binding_generator import _typedef_to_aliases
from numbast.static.typedef import render_aliases
from numbast.static.renderer import (
    get_prefix,
    get_rendered_shims,
    get_rendered_imports,
)

CUDA_INCLUDE_PATH = config.CUDA_INCLUDE_PATH
COMPUTE_CAPABILITY = cuda.get_current_device().compute_capability


@pytest.fixture(scope="module")
def float16():
    cuda_fp16 = os.path.join(CUDA_INCLUDE_PATH, "cuda_fp16.h")
    cuda_fp16_hpp = os.path.join(CUDA_INCLUDE_PATH, "cuda_fp16.hpp")

    structs, functions, _, _, typedefs, _ = parse_declarations_from_source(
        cuda_fp16,
        [cuda_fp16, cuda_fp16_hpp],
        f"sm_{COMPUTE_CAPABILITY[0]}{COMPUTE_CAPABILITY[1]}",
        cudatoolkit_include_dir=CUDA_INCLUDE_PATH,
    )

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
        with_prefix=False, with_imports=False, with_shim_functions=False
    )

    function_bindings = SFR.render_as_str(
        with_prefix=False, with_imports=False, with_shim_functions=False
    )

    prefix_str = get_prefix()
    imports_str = get_rendered_imports()
    shim_function_str = get_rendered_shims()

    bindings = f"""
{prefix_str}
{imports_str}
{struct_bindings}
{function_bindings}
{typedef_bindings}
{shim_function_str}
"""

    globals = {}
    exec(bindings, globals)

    public_apis = ["half", "half2", "hsin", "c_ext_shim_source"]
    assert all(public_api in globals for public_api in public_apis)

    return {k: globals[k] for k in public_apis}


@pytest.fixture(scope="module")
def numbast_jit(float16):
    c_ext_shim_source = float16["c_ext_shim_source"]
    return partial(cuda.jit, link=[c_ext_shim_source])


def test_float16(float16, numbast_jit):
    fp16 = float16["half"]
    hsin = float16["hsin"]

    @numbast_jit
    def kernel(arr):
        three = fp16(1.0) + fp16(2.0)
        sin_three = hsin(three)
        arr[0] = types.float32(three)
        arr[1] = types.float32(sin_three)

    arr = np.array([0, 0], dtype="f8")

    kernel[1, 1](arr)

    np.testing.assert_allclose(arr, [3.0, math.sin(3.0)], atol=1e-5)
