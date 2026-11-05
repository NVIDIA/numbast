# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pickle

import pytest

from ast_canopy import parse_declarations_from_source


@pytest.fixture(scope="module")
def sample_macro_source(data_folder):
    return data_folder / "sample_macro.cu"


def test_macro_expansions(sample_macro_source):
    srcstr = str(sample_macro_source)

    decls = parse_declarations_from_source(
        srcstr,
        [srcstr],
        "sm_80",
    )
    functions = decls.functions

    assert len(functions) == 3

    forty_two_int, forty_two_float, forty_two_double = functions

    assert forty_two_int.name == "forty_two_int"
    assert forty_two_float.name == "forty_two_float"
    assert forty_two_double.name == "forty_two_double"

    assert forty_two_int.return_type.name == "int"
    assert forty_two_float.return_type.name == "float"
    assert forty_two_double.return_type.name == "double"


def test_macro_defines(data_folder):
    srcstr = str(data_folder / "sample_macro_defines.cu")

    decls = parse_declarations_from_source(
        srcstr,
        [srcstr],
        "sm_80",
    )

    assert decls.macro_defines["foo"].replacement_text == "123"
    assert decls.macro_defines["FLAG"].replacement_text == ""
    assert decls.macro_defines["FLAG"].evaluated_constant is None
    assert decls.macro_defines["foo"].name == "foo"
    assert decls.macro_defines["foo"].evaluated_constant.value == 123
    assert "MAKE_VALUE" not in decls.macro_defines
    assert "INCLUDED_VALUE" not in decls.macro_defines


def test_macro_defines_retained_include(data_folder):
    srcstr = str(data_folder / "sample_macro_defines.cu")
    include = str(data_folder / "sample_macro_defines_include.cuh")

    decls = parse_declarations_from_source(
        srcstr,
        [srcstr, include],
        "sm_80",
    )

    assert decls.macro_defines["foo"].replacement_text == "123"
    assert decls.macro_defines["INCLUDED_VALUE"].replacement_text == "456"


@pytest.fixture(scope="module")
def macro_declarations(data_folder):
    source = str(data_folder / "sample_macro_constants.cu")
    return parse_declarations_from_source(source, [source], "sm_80")


def test_macro_constant_values_and_types(macro_declarations):
    constants = {
        name: definition.evaluated_constant
        for name, definition in macro_declarations.macro_defines.items()
        if definition.evaluated_constant is not None
    }
    expected = {
        "MACRO_INTEGER": 42,
        "MACRO_UNSIGNED": 2**64 - 1,
        "MACRO_NEGATIVE": -23,
        "MACRO_MASK": 1 << 40,
        "MACRO_ALIAS": 1 << 40,
        "MACRO_COMBINED": (1 << 40) | 3,
        "MACRO_SIZE": 16,
        "MACRO_VERSION": (2 << 16) + 16,
        "MACRO_CAST": 65535,
        "MACRO_BOOL": True,
        "MACRO_FLOAT": 1.2345678901234567,
        "MACRO_ENUM": 7,
        "MACRO_CONSTEXPR_CALL": 81,
        "MACRO_CONSTANT_VARIABLE": 11,
        "MACRO_FUNCTION_EXPANSION": 42,
        "MACRO_INCLUDED": 457,
        "MACRO_REDEFINED": 2,
        "Z_MACRO_AFTER_ERRORS": 99,
    }
    assert {name: item.value for name, item in constants.items()} == expected
    assert constants["MACRO_UNSIGNED"].value_serialized == str(2**64 - 1)
    assert constants["MACRO_UNSIGNED"].type_.name == "unsigned long long"
    assert constants["MACRO_BOOL"].type_.name == "bool"
    assert constants["MACRO_BOOL"].value is True
    for name, item in constants.items():
        assert item.name == item.qual_name == name
        assert not item.is_constexpr  # A macro is not a constexpr variable.


def test_macro_replacement_text_is_preserved(macro_declarations):
    definitions = macro_declarations.macro_defines
    assert definitions["MACRO_INTEGER"].replacement_text == "42"
    assert definitions["MACRO_ALIAS"].replacement_text == "MACRO_MASK"
    assert (
        definitions["MACRO_VERSION"].replacement_text
        == "( ( 2 << 16 ) + sizeof ( MacroLayout ) )"
    )
    assert definitions["MACRO_STRING"].replacement_text == '"text"'
    assert definitions["MACRO_EMPTY"].replacement_text == ""
    assert "MACRO_TWICE" not in definitions
    assert "MACRO_REMOVED" not in definitions
    assert "INCLUDED_VALUE" not in definitions


def test_unevaluable_macros_keep_text_without_values(macro_declarations):
    for name in (
        "MACRO_EMPTY",
        "MACRO_STRING",
        "MACRO_TYPE",
        "MACRO_UNKNOWN",
        "MACRO_RUNTIME",
        "MACRO_RUNTIME_CALL",
        "MACRO_SIDE_EFFECT",
        "MACRO_DIV_ZERO",
        "MACRO_TRAILING",
        "MACRO_BROKEN",
        "MACRO_POINTER",
        "MACRO_AGGREGATE",
        "MACRO_RECURSIVE",
    ):
        assert name in macro_declarations.macro_defines
        assert macro_declarations.macro_defines[name].evaluated_constant is None
    # Probes must not leak synthetic declarations into the public AST output.
    assert [item.name for item in macro_declarations.global_constants] == [
        "macro_constant"
    ]


def test_macro_constant_target_and_defines(data_folder):
    source = str(data_folder / "sample_macro_constants.cu")
    decls = parse_declarations_from_source(
        source,
        [source, str(data_folder / "sample_macro_defines_include.cuh")],
        "sm_80",
        defines=["MACRO_BUILD_SETTING=37", "__CUDA_ARCH__=800"],
    )
    assert decls.macro_defines["INCLUDED_VALUE"].evaluated_constant.value == 456
    assert decls.macro_defines["MACRO_INCLUDED"].evaluated_constant.value == 457
    assert decls.macro_defines["MACRO_PROFILE"].evaluated_constant.value == 837
    assert "MACRO_BUILD_SETTING" not in decls.macro_defines
    assert "__CUDA_ARCH__" not in decls.macro_defines


def test_macro_values_pickle(macro_declarations):
    definitions = macro_declarations.macro_defines
    restored = pickle.loads(pickle.dumps(definitions))
    assert restored.keys() == definitions.keys()
    for name, original in definitions.items():
        item = restored[name]
        assert item.name == original.name == name
        assert item.replacement_text == original.replacement_text
        constant = item.evaluated_constant
        expected = original.evaluated_constant
        if expected is None:
            assert constant is None
        else:
            assert constant.value == expected.value
            assert constant.type_.name == expected.type_.name
            assert constant.value_serialized == expected.value_serialized
