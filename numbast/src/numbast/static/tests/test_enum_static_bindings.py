# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest


from ast_canopy import parse_declarations_from_source
from numbast.static.renderer import clear_base_renderer_cache
from numbast.static.enum import StaticEnumsRenderer


@pytest.fixture(scope="module")
def cuda_enum(data_folder):
    clear_base_renderer_cache()

    header = data_folder("enum.cuh")

    decls = parse_declarations_from_source(header, [header], "sm_50")
    enums = decls.enums

    assert len(enums) == 2

    SER = StaticEnumsRenderer(enums)

    bindings = SER.render_as_str(
        with_prefix=True, with_imports=True, with_shim_functions=False
    )

    globals = {}
    with open("/tmp/data.py", "w") as f:
        f.write(bindings)

    exec(bindings, globals)

    public_apis = ["Fruit", "Animal"]
    assert all(public_api in globals for public_api in public_apis)

    return {k: globals[k] for k in public_apis}


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
