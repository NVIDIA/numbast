# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from functools import partial

from numba import cuda
from numba.types import Number, float64
from numba.core.datamodel import PrimitiveModel
from numba.cuda import device_array

from ast_canopy import parse_declarations_from_source
from numbast.static.struct import StaticStructsRenderer


@pytest.fixture(scope="module")
def cuda_struct(data_folder):
    header = data_folder("demo.cuh")

    specs = {"__myfloat16": (Number, PrimitiveModel, header)}

    decls = parse_declarations_from_source(header, [header], "sm_50")
    structs = decls.structs

    assert len(structs) == 1

    SSR = StaticStructsRenderer(structs, specs)

    bindings = SSR.render_as_str(
        with_prefix=True, with_imports=True, with_shim_functions=True
    )

    globals = {}
    exec(bindings, globals)

    public_apis = [*specs, "c_ext_shim_source"]
    assert all(public_api in globals for public_api in public_apis)

    return {k: globals[k] for k in public_apis}


@pytest.fixture(scope="module")
def numbast_jit(cuda_struct):
    c_ext_shim_source = cuda_struct["c_ext_shim_source"]
    return partial(cuda.jit, link=[c_ext_shim_source])


def test_demo(cuda_struct, numbast_jit):
    __myfloat16 = cuda_struct["__myfloat16"]

    @numbast_jit
    def kernel(arr):
        foo = __myfloat16(3.14)  # noqa: F841

        arr[0] = float64(foo)

    arr = device_array((1,), "float64")
    kernel[1, 1](arr)
    assert all(arr.copy_to_host() == [])
