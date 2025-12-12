# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from numba import cuda
from numba.cuda.types import Type
from numba.cuda.datamodel import StructModel
from numba.cuda import device_array


@pytest.fixture(scope="function")
def cuda_decls(make_binding):
    types = {
        "Foo": Type,
    }
    datamodels = {
        "Foo": StructModel,
    }

    res = make_binding("operator.cuh", types, datamodels, "sm_50")
    bindings = res["bindings"]

    public_apis = ["Foo"]
    assert all(public_api in bindings for public_api in public_apis)

    return bindings


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
