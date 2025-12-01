# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from numba import cuda
import numpy as np
import cffi


@pytest.fixture(scope="function")
def cuda_enum(make_binding):
    res = make_binding("enum.cuh", {}, {}, "sm_50")
    return res["bindings"]


def test_enum(cuda_enum):
    Fruit = cuda_enum["Fruit"]

    assert Fruit.Apple == 1
    assert Fruit.Banana == 3
    assert Fruit.Orange == 5

    assert Fruit(1) == Fruit.Apple
    assert Fruit(3) == Fruit.Banana
    assert Fruit(5) == Fruit.Orange


def test_enum_class(cuda_enum):
    Animal = cuda_enum["Animal"]

    assert Animal.Cat == 0
    assert Animal.Dog == 1
    assert Animal.Horse == 2

    assert Animal(0) == Animal.Cat
    assert Animal(1) == Animal.Dog
    assert Animal(2) == Animal.Horse


def test_enum_used_in_function_argument(cuda_enum):
    ffi = cffi.FFI()
    feed = cuda_enum["feed"]
    Animal = cuda_enum["Animal"]

    @cuda.jit
    def kernel(out):
        slot = ffi.from_buffer(out[0:1])
        feed(Animal.Cat, slot)

    out = np.zeros(1, dtype=np.int32)
    kernel[1, 1](out)
    assert np.array_equal(out, [1])
