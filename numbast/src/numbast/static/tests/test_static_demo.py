# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import numpy as np

from numba import cuda
from numba.cuda.types import Number, float64
from numba.cuda.datamodel import PrimitiveModel
from numba.cuda import device_array

from ast_canopy import parse_declarations_from_source
from numbast.static.struct import StaticStructsRenderer
from numbast.static.renderer import clear_base_renderer_cache, registry_setup
from numbast.static.function import clear_function_apis_registry


@pytest.fixture(scope="module")
def decl(data_folder):
    clear_base_renderer_cache()
    clear_function_apis_registry()

    header = data_folder("demo.cuh")

    specs = {"__myfloat16": (Number, PrimitiveModel, header)}

    decls = parse_declarations_from_source(header, [header], "sm_50")
    structs = decls.structs

    assert len(structs) == 1

    registry_setup(use_separate_registry=False)
    SSR = StaticStructsRenderer(structs, specs, header)

    bindings = SSR.render_as_str(with_imports=True, with_shim_stream=True)

    globals = {}
    exec(bindings, globals)

    public_apis = [*specs]
    assert all(public_api in globals for public_api in public_apis)

    return {k: globals[k] for k in public_apis}


def test_demo(decl):
    __myfloat16 = decl["__myfloat16"]

    @cuda.jit
    def kernel(arr):
        foo = __myfloat16(3.14)  # noqa: F841

        arr[0] = float64(foo)

    arr = device_array((1,), "float64")
    kernel[1, 1](arr)
    host = arr.copy_to_host()
    assert np.allclose(host, [3.14], rtol=1e-3, atol=1e-3)
