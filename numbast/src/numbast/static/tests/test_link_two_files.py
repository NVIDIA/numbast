# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from numba import cuda
from numba.types import Type, Number
from numba.core.datamodel import StructModel, PrimitiveModel


@pytest.fixture(scope="module")
def header(data_folder):
    return {
        "struct": data_folder("struct.cuh"),
        "function": data_folder("function.cuh"),
    }


@pytest.fixture(scope="function")
def foo_decl(header, make_binding):
    types = {
        "Foo": Type,
        "Bar": Type,
        "MyInt": Number,
    }
    datamodels = {
        "Foo": StructModel,
        "Bar": StructModel,
        "MyInt": PrimitiveModel,
    }

    res = make_binding("struct.cuh", types, datamodels, "sm_50")
    bindings = res["bindings"]

    public_apis = ["Foo", "Bar", "MyInt"]
    assert all(public_api in bindings for public_api in public_apis)

    return bindings


@pytest.fixture(scope="function")
def function_decl(make_binding):
    res = make_binding("function.cuh", {}, {}, "sm_50")
    bindings = res["bindings"]

    public_apis = ["add", "minus_i32_f32", "set_42"]
    assert all(public_api in bindings for public_api in public_apis)

    return bindings


@pytest.fixture(scope="module")
def foo_impl(data_folder, header):
    src = data_folder("src", "struct.cu")
    header = header["struct"]
    with open(src) as f:
        impl = f.read()

    header_str = f"#include <{header}>"
    return cuda.CUSource(header_str + "\n" + impl)


@pytest.fixture(scope="module")
def function_impl(data_folder):
    return data_folder("src", "function.cu")


@pytest.fixture(scope="module")
def impl(foo_impl, function_impl):
    return [foo_impl, function_impl]


def test_interleave_calls(foo_decl, function_decl, impl):
    Foo = foo_decl["Foo"]
    add = function_decl["add"]

    @cuda.jit(link=impl)
    def foo_kernel():
        foo = Foo()  # noqa: F841

    @cuda.jit(link=impl)
    def add_kernel():
        add(1, 2)

    foo_kernel[1, 1]()
    add_kernel[1, 1]()
    foo_kernel[1, 1]()
    add_kernel[1, 1]()


def test_back_to_back_calls(foo_decl, function_decl, impl):
    Foo = foo_decl["Foo"]
    add = function_decl["add"]

    @cuda.jit(link=impl)
    def foo_kernel():
        foo = Foo()  # noqa: F841

    @cuda.jit(link=impl)
    def add_kernel():
        add(1, 2)

    foo_kernel[1, 1]()
    foo_kernel[1, 1]()
    add_kernel[1, 1]()
    add_kernel[1, 1]()
