# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from numba.types import Type
from numba.core.datamodel import StructModel

from ast_canopy import parse_declarations_from_source
from numbast.static.struct import StaticStructRenderer


@pytest.fixture
def Foo_static_struct_bindings(sample_struct):
    structs, *_ = parse_declarations_from_source(
        sample_struct, [sample_struct], "sm_50"
    )

    assert len(structs) == 1

    FooDecl = structs[0]
    SSR = StaticStructRenderer(
        FooDecl, "Foo", Type, StructModel, header_path=sample_struct
    )

    bindings = SSR.render_as_str()
    globals = {}
    exec(bindings, globals)

    public_apis = ["Foo", "c_ext_shim_source"]
    assert all(public_api in globals for public_api in public_apis)

    return {k: globals[k] for k in public_apis}


def test_foo_ctor(Foo_static_struct_bindings):
    Foo = Foo_static_struct_bindings["Foo"]
    c_ext_shim_source = Foo_static_struct_bindings["c_ext_shim_source"]

    from numba import cuda

    @cuda.jit(link=[c_ext_shim_source])
    def kernel():
        foo = Foo()  # noqa: F841

    kernel[1, 1]()
