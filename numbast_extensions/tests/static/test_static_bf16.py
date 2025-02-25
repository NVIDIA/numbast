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
    get_rendered_imports,
)

CUDA_INCLUDE_PATH = config.CUDA_INCLUDE_PATH
COMPUTE_CAPABILITY = cuda.get_current_device().compute_capability


@pytest.fixture(scope="module")
def bfloat16():
    cuda_bf16 = os.path.join(CUDA_INCLUDE_PATH, "cuda_bf16.h")
    cuda_bf16_hpp = os.path.join(CUDA_INCLUDE_PATH, "cuda_bf16.hpp")

    decls = parse_declarations_from_source(
        cuda_bf16,
        [cuda_bf16, cuda_bf16_hpp],
        f"sm_{COMPUTE_CAPABILITY[0]}{COMPUTE_CAPABILITY[1]}",
        cudatoolkit_include_dir=CUDA_INCLUDE_PATH,
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
        with_prefix=False, with_imports=False, with_shim_functions=False
    )

    function_bindings = SFR.render_as_str(
        with_prefix=False, with_imports=False, with_shim_functions=False
    )

    prefix_str = get_prefix()
    imports_str = get_rendered_imports()

    bindings = f"""
{prefix_str}
{imports_str}
{struct_bindings}
{function_bindings}
{typedef_bindings}
"""

    globals = {}

    with open("/tmp/bfloat16.py", "w") as f:
        f.write(bindings)

    exec(bindings, globals)

    public_apis = ["nv_bfloat16", "nv_bfloat16", "hsin", "c_ext_shim_source"]
    assert all(public_api in globals for public_api in public_apis)

    return {k: globals[k] for k in public_apis}


@pytest.fixture(scope="module")
def numbast_jit(bfloat16):
    c_ext_shim_source = bfloat16["c_ext_shim_source"]
    return partial(cuda.jit, link=[c_ext_shim_source])


def test_bfloat16(bfloat16, numbast_jit):
    bf16 = bfloat16["nv_bfloat16"]
    hsin = bfloat16["hsin"]

    @numbast_jit
    def kernel(arr):
        three = bf16(1.0) + bf16(2.0)
        sin_three = hsin(three)
        arr[0] = types.float32(three)
        arr[1] = types.float32(sin_three)

    arr = np.array([0, 0], dtype="f8")

    kernel[1, 1](arr)

    np.testing.assert_allclose(arr, [3.0, math.sin(3.0)], atol=1e-3)
