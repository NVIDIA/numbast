# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from numba import cuda
import numpy as np
import cffi


@pytest.fixture(scope="function")
def decl(make_binding):
    res = make_binding("enum.cuh", {}, {}, "sm_50")
    with open("/tmp/binding.py", "w") as f:
        f.write(res["src"])
    return res


@pytest.fixture(scope="function")
def cuda_enum(decl):
    return decl["bindings"]


@pytest.fixture(scope="function")
def cuda_enum_src(decl):
    return decl["src"]


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


def test_enum_with_different_underlying_integer_types(cuda_enum, cuda_enum_src):
    Car = cuda_enum["Car"]
    assert Car.Sedan == 0
    assert Car.SUV == 1
    assert Car.Pickup == 2
    assert Car.Hatchback == 3

    Color = cuda_enum["Color"]
    assert Color.Red == 0
    assert Color.Green == 1
    assert Color.Blue == 2
    assert Color.Black == -1

    ffi = cffi.FFI()
    car_with_color = cuda_enum["car_with_color"]

    @cuda.jit
    def check_car_with_color(car, color, out):
        ptr = ffi.from_buffer(out)
        car_with_color(car, color, ptr)

    out = np.zeros(256, dtype=np.int8)
    check_car_with_color[1, 1](Car.Sedan, Color.Red, out)

    end = np.where(out == 0)
    end = end[0][0] if end[0].size > 0 else len(out)

    assert out[:end].tobytes().decode("ascii") == "Red Sedan"

    assert '"Fruit":types.uint32' in cuda_enum_src
    assert '"Animal":types.int32' in cuda_enum_src
    assert '"Car":types.uint8' in cuda_enum_src
    assert '"Color":types.int16' in cuda_enum_src
