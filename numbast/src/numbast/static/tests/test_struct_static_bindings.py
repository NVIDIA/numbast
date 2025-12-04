# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import numpy as np

from numba import cuda
from numba.cuda.types import Type, Number
from numba.cuda.datamodel import StructModel, PrimitiveModel
from numba.cuda import device_array

from ast_canopy import parse_declarations_from_source
from numbast.static.renderer import clear_base_renderer_cache, registry_setup
from numbast.static.function import clear_function_apis_registry
from numbast.static.struct import StaticStructsRenderer


@pytest.fixture(scope="module")
def header(data_folder):
    return data_folder("struct.cuh")


@pytest.fixture(scope="module")
def decl(data_folder, header):
    clear_base_renderer_cache()
    clear_function_apis_registry()
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
def impl(data_folder, header):
    src = data_folder("src", "struct.cu")
    with open(src) as f:
        impl = f.read()

    header_str = f"#include <{header}>"
    return cuda.CUSource(header_str + "\n" + impl)


def test_foo_ctor_default_simple(decl, impl):
    Foo = decl["Foo"]

    @cuda.jit(link=[impl])
    def kernel(arr):
        foo = Foo()  # noqa: F841
        foo2 = Foo(42)

        arr[0] = foo.x
        arr[1] = foo2.x

    arr = device_array((2,), "int32")
    kernel[1, 1](arr)
    assert all(arr.copy_to_host() == [0, 42])


def test_foo_attriubte_readonly_access(decl, impl):
    Foo = decl["Foo"]

    @cuda.jit(link=[impl])
    def kernel(arr):
        foo = Foo()
        arr[0] = foo.x

    arr = np.ones((1,), dtype="int32")
    kernel[1, 1](arr)
    assert arr == pytest.approx([0])


def test_bar_ctor_overloads(decl, impl):
    Bar = decl["Bar"]

    from numba.types import int32, float32

    @cuda.jit(link=[impl])
    def kernel(arr):
        bar = Bar(int32(3.14))
        bar2 = Bar(float32(3.14))
        arr[0] = bar.x
        arr[1] = bar2.x

    arr = device_array((2,), "float32")
    kernel[1, 1](arr)
    assert arr.copy_to_host() == pytest.approx([3, 3.14])


def test_myint_cast(decl, impl):
    MyInt = decl["MyInt"]

    from numba.types import int32

    @cuda.jit(link=[impl])
    def kernel(arr):
        i = MyInt(42)
        arr[0] = int32(i)

    arr = device_array((1,), "int32")
    kernel[1, 1](arr)
    assert arr.copy_to_host() == pytest.approx([42])


def test_static_type_check(decl, impl):
    MyInt = decl["MyInt"]

    from numba.types import int32

    @cuda.jit(link=[impl])
    def kernel(arr):
        i = MyInt(42)
        if isinstance(i, MyInt):
            arr[0] = int32(i)
        else:
            arr[0] = 0

    arr = device_array((1,), "int32")
    kernel[1, 1](arr)
    assert arr.copy_to_host() == pytest.approx([42])


def test_struct_methods_simple_static(decl, impl):
    Foo = decl["Foo"]

    @cuda.jit(link=[impl])
    def kernel(arr):
        foo = Foo()
        arr[0] = foo.get_x()

    arr = device_array((1,), "int32")
    kernel[1, 1](arr)
    assert arr.copy_to_host()[0] == 0


def test_struct_methods_argument_static(decl, impl):
    from numba.types import float32 as nb_float32

    Foo = decl["Foo"]

    @cuda.jit(link=[impl])
    def kernel(arr):
        foo = Foo()
        arr[0] = foo.add_one(nb_float32(42.0))

    arr = device_array((1,), "float32")
    kernel[1, 1](arr)
    assert arr.copy_to_host() == pytest.approx([43.0])


def test_struct_void_return_static(decl, impl, capfd):
    Foo = decl["Foo"]

    @cuda.jit(link=[impl])
    def kernel():
        foo = Foo()
        foo.print()

    kernel[1, 1]()
    cuda.synchronize()
    captured = capfd.readouterr()
    assert "Foo: 0" in captured.out
