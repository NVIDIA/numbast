# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from numbast.utils import make_function_shim


class _DummyType:
    def __init__(self, name: str):
        self.unqualified_non_ref_type_name = name


class _DummyParam:
    def __init__(self, type_name: str, name: str = ""):
        self.type_ = _DummyType(type_name)
        self.name = name
        self.unqualified_non_ref_type_name = type_name


def test_make_function_shim_names_unnamed_parameters():
    params = [_DummyParam("Foo", "")]

    shim = make_function_shim("shim", "useFoo", "bool", params)

    assert "Foo* arg0" in shim
    assert "useFoo(*arg0);" in shim


def test_make_function_shim_disambiguates_duplicate_names():
    params = [_DummyParam("Foo", "x"), _DummyParam("Bar", "x")]

    shim = make_function_shim("shim", "useFoo", "bool", params)

    assert "Foo* x" in shim
    # The second argument should be suffixed to avoid clashing with the first.
    assert "Bar* x_1" in shim
    assert "useFoo(*x, *x_1);" in shim
