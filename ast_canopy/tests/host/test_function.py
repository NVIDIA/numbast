# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest

from ast_canopy import parse_declarations_from_source
from ast_canopy.pylibastcanopy import execution_space


@pytest.fixture(scope="module")
def host_function_source():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_directory, "data", "sample_host_function.cpp")


def test_parse_host_functions(host_function_source):
    decls = parse_declarations_from_source(
        host_function_source, [host_function_source], "sm_80"
    )

    functions = decls.functions
    expected_names = {
        "add",
        "scale",
        "set_value",
        "host_offset",
        "add_host_device",
    }

    assert len(functions) == len(expected_names)
    assert {func.name for func in functions} == expected_names

    funcs_by_name = {func.name: func for func in functions}

    add = funcs_by_name["add"]
    assert add.return_type.name == "int"
    assert [param.name for param in add.params] == ["a", "b"]
    assert [param.type_.name for param in add.params] == ["int", "int"]
    assert add.exec_space == execution_space.undefined

    scale = funcs_by_name["scale"]
    assert scale.return_type.name == "float"
    assert [param.name for param in scale.params] == ["value", "factor"]
    assert [param.type_.name for param in scale.params] == ["float", "float"]
    assert scale.exec_space == execution_space.undefined

    set_value = funcs_by_name["set_value"]
    assert set_value.return_type.name == "void"
    assert [param.name for param in set_value.params] == ["out", "value"]
    assert [param.type_.name for param in set_value.params] == [
        "int *",
        "int",
    ]
    assert set_value.exec_space == execution_space.undefined

    host_offset = funcs_by_name["host_offset"]
    assert host_offset.return_type.name == "double"
    assert [param.name for param in host_offset.params] == ["x", "offset"]
    assert [param.type_.name for param in host_offset.params] == [
        "double",
        "double",
    ]
    assert host_offset.exec_space == execution_space.host

    add_host_device = funcs_by_name["add_host_device"]
    assert add_host_device.return_type.name == "int"
    assert [param.name for param in add_host_device.params] == ["a", "b"]
    assert [param.type_.name for param in add_host_device.params] == [
        "int",
        "int",
    ]
    assert add_host_device.exec_space == execution_space.host_device

    for func in functions:
        assert func.parse_entry_point == host_function_source
