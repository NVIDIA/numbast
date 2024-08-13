# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from numba.types import Type
from numba.core.datamodel import StructModel
from numba.cuda import device_array

from ast_canopy import parse_declarations_from_source
from numbast.static.struct import StaticStructRenderer


@pytest.fixture
def one_struct(sample):
    header = sample("one_struct.cuh")
    structs, *_ = parse_declarations_from_source(header, [header], "sm_50")

    assert len(structs) == 1

    FooDecl = structs[0]
    SSR = StaticStructRenderer(FooDecl, "Foo", Type, StructModel, header_path=header)

    bindings = SSR.render_as_str()
    globals = {}
    exec(bindings, globals)

    public_apis = ["Foo", "c_ext_shim_source"]
    assert all(public_api in globals for public_api in public_apis)

    return {k: globals[k] for k in public_apis}


def test_foo_ctor(one_struct):
    Foo = one_struct["Foo"]
    c_ext_shim_source = one_struct["c_ext_shim_source"]

    from numba import cuda

    @cuda.jit(link=[c_ext_shim_source])
    def kernel(arr):
        foo = Foo()  # noqa: F841
        foo2 = Foo(42)

        arr[0] = foo.x
        arr[1] = foo2.x

    arr = device_array((2,), "int32")
    kernel[1, 1](arr)
    assert all(arr.copy_to_host() == [0, 42])
