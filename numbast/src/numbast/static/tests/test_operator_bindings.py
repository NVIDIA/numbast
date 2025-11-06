# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from numba import cuda
from numba.cuda.types import Type
from numba.cuda.datamodel import StructModel
from numba.cuda import device_array

from ast_canopy import parse_declarations_from_source

from numbast.static.renderer import (
    get_rendered_imports,
    get_shim,
    registry_setup,
)
from numbast.static.renderer import clear_base_renderer_cache
from numbast.static.function import clear_function_apis_registry
from numbast.static.struct import StaticStructsRenderer
from numbast.static.function import StaticFunctionsRenderer


@pytest.fixture(scope="module")
def cuda_decls(data_folder):
    clear_base_renderer_cache()
    clear_function_apis_registry()

    header = data_folder("operator.cuh")

    specs = {"Foo": (Type, StructModel, header)}

    decls = parse_declarations_from_source(header, [header], "sm_50")
    structs = decls.structs
    functions = decls.functions

    assert len(structs) == 1

    registry_setup(use_separate_registry=False)
    SSR = StaticStructsRenderer(structs, specs)
    SFR = StaticFunctionsRenderer(functions, header)

    struct_bindings = SSR.render_as_str(
        with_imports=False, with_shim_stream=False
    )
    function_bindings = SFR.render_as_str(
        with_imports=False, with_shim_stream=False
    )

    shim_include = f'"#include<{header}>"'
    bindings = "\n".join(
        [
            get_rendered_imports(),
            get_shim(shim_include),
            struct_bindings,
            function_bindings,
        ]
    )

    globals = {}
    exec(bindings, globals)

    public_apis = [*specs]
    assert all(public_api in globals for public_api in public_apis)

    return {k: globals[k] for k in public_apis}


@pytest.fixture(scope="module")
def impl(data_folder):
    header = data_folder("operator.cuh")
    src = data_folder("src", "operator.cu")

    with open(src) as f:
        impl = f.read()

    return cuda.CUSource(f"#include <{header}>" + "\n" + impl)


def test_custom_type_operators(cuda_decls, impl):
    Foo = cuda_decls["Foo"]

    @cuda.jit(link=[impl])
    def kernel(arr):
        foo = Foo(43)  # noqa: F841
        foo2 = Foo(42)
        foo3 = foo + foo2

        arr[0] = foo3.x

    arr = device_array((1,), "int32")
    kernel[1, 1](arr)
    assert all(arr.copy_to_host() == [85])
