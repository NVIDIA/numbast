# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from numba import cuda
from numba.cuda.types import Type, Number
from numba.cuda.datamodel import StructModel, PrimitiveModel

from ast_canopy import parse_declarations_from_source
from numbast.static.renderer import clear_base_renderer_cache, registry_setup
from numbast.static.struct import StaticStructsRenderer
from numbast.static.function import (
    StaticFunctionsRenderer,
    clear_function_apis_registry,
)


@pytest.fixture(scope="module")
def header(data_folder):
    return {
        "struct": data_folder("struct.cuh"),
        "function": data_folder("function.cuh"),
    }


@pytest.fixture(scope="module")
def foo_decl(header):
    clear_base_renderer_cache()
    clear_function_apis_registry()

    header = header["struct"]

    specs = {
        "Foo": (Type, StructModel, header),
        "Bar": (Type, StructModel, header),
        "MyInt": (Number, PrimitiveModel, header),
    }

    decls = parse_declarations_from_source(header, [header], "sm_50")
    structs = decls.structs

    assert len(structs) == 3

    registry_setup(use_separate_registry=False)
    SSR = StaticStructsRenderer(structs, specs, header)

    bindings = SSR.render_as_str(with_imports=True, with_shim_stream=True)

    globals = {}
    exec(bindings, globals)

    public_apis = [*specs]
    assert all(public_api in globals for public_api in public_apis)

    return {k: globals[k] for k in public_apis}


@pytest.fixture(scope="module")
def function_decl(header):
    clear_base_renderer_cache()
    clear_function_apis_registry()

    header = header["function"]

    decls = parse_declarations_from_source(header, [header], "sm_50")
    functions = decls.functions

    assert len(functions) == 5

    registry_setup(use_separate_registry=False)
    SFR = StaticFunctionsRenderer(functions, header)

    bindings = SFR.render_as_str(with_imports=True, with_shim_stream=True)
    globals = {}
    exec(bindings, globals)

    public_apis = ["add", "minus_i32_f32", "set_42"]
    assert all(public_api in globals for public_api in public_apis)

    return {k: globals[k] for k in public_apis}


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
