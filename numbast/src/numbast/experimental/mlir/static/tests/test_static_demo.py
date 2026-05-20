# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import numpy as np

from numba_cuda_mlir import cuda
from numba_cuda_mlir.types import Number, float64
from numba_cuda_mlir.models import PrimitiveModel
from numba_cuda_mlir.numba_cuda import device_array


@pytest.fixture(scope="function")
def decl(make_binding):
    # Needs config items
    types = {
        "__myfloat16": Number,
    }
    datamodels = {
        "__myfloat16": PrimitiveModel,
    }
    res = make_binding("demo.cuh", types, datamodels, "sm_50")
    bindings = res["bindings"]

    with open("/tmp/bindings.py", "w") as f:
        f.write(res["src"])

    public_apis = ["__myfloat16"]
    assert all(public_api in bindings for public_api in public_apis)

    return bindings


@pytest.mark.skip(reason="Add config item for backend type of PrimitiveModels")
def test_demo(decl):
    __myfloat16 = decl["__myfloat16"]

    @cuda.jit(dump=True)
    def kernel(arr):
        foo = __myfloat16(3.14)  # noqa: F841

        arr[0] = float64(foo)

    arr = device_array((1,), "float64")
    kernel[1, 1](arr)
    host = arr.copy_to_host()
    assert np.allclose(host, [3.14], rtol=1e-3, atol=1e-3)
