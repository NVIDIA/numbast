# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from functools import partial

from numba import cuda
from numba.types import Type, Number
from numba.core.datamodel import StructModel, PrimitiveModel
from numba.cuda import device_array

from ast_canopy import parse_declarations_from_source
from numbast.static.renderer import clear_base_renderer_cache
from numbast.static.struct import StaticStructsRenderer


@pytest.fixture(scope="module")
def cuda_struct(data_folder):
    clear_base_renderer_cache()

    header = data_folder("struct.cuh")

    specs = {
        "Foo": (Type, StructModel, header),
        "Bar": (Type, StructModel, header),
        "MyInt": (Number, PrimitiveModel, header),
    }

    decls = parse_declarations_from_source(header, [header], "sm_50")
    structs = decls.structs

    assert len(structs) == 3

    SSR = StaticStructsRenderer(structs, specs)

    bindings = SSR.render_as_str(
        with_prefix=True, with_imports=True, with_shim_functions=True
    )

    globals = {}
    with open("/tmp/data.py", "w") as f:
        f.write(bindings)

    exec(bindings, globals)

    public_apis = [*specs, "c_ext_shim_source"]
    assert all(public_api in globals for public_api in public_apis)

    return {k: globals[k] for k in public_apis}


@pytest.fixture(scope="module")
def numbast_jit(cuda_struct):
    c_ext_shim_source = cuda_struct["c_ext_shim_source"]
    return partial(cuda.jit, link=[c_ext_shim_source])


def test_foo_ctor_default_simple(cuda_struct, numbast_jit):
    Foo = cuda_struct["Foo"]

    @numbast_jit
    def kernel(arr):
        foo = Foo()  # noqa: F841
        foo2 = Foo(42)

        arr[0] = foo.x
        arr[1] = foo2.x

    arr = device_array((2,), "int32")
    kernel[1, 1](arr)
    assert all(arr.copy_to_host() == [0, 42])


def test_bar_ctor_overloads(cuda_struct, numbast_jit):
    Bar = cuda_struct["Bar"]

    from numba.types import int32, float32

    @numbast_jit
    def kernel(arr):
        bar = Bar(int32(3.14))
        bar2 = Bar(float32(3.14))
        arr[0] = bar.x
        arr[1] = bar2.x

    arr = device_array((2,), "float32")
    kernel[1, 1](arr)
    assert arr.copy_to_host() == pytest.approx([3, 3.14])


def test_myint_cast(cuda_struct, numbast_jit):
    MyInt = cuda_struct["MyInt"]

    from numba.types import int32

    @numbast_jit
    def kernel(arr):
        i = MyInt(42)
        arr[0] = int32(i)

    arr = device_array((1,), "int32")
    kernel[1, 1](arr)
    assert arr.copy_to_host() == pytest.approx([42])
