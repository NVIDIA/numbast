# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from ast_canopy import parse_declarations_from_source
from ast_canopy.pylibastcanopy import ParseError


DATA_DIR = Path(__file__).parent / "data"


def _parse_enums(source: Path, *, cxx_standard: str = "gnu++17"):
    declarations = parse_declarations_from_source(
        str(source),
        [str(source)],
        "sm_80",
        cxx_standard=cxx_standard,
    )
    return {enum.name: enum for enum in declarations.enums}


@pytest.fixture(scope="module")
def enums():
    return _parse_enums(DATA_DIR / "sample_enum_values.cu")


@pytest.mark.parametrize(
    ("enum_name", "expected_values"),
    [
        (
            "IntegerLiteralValues",
            {
                "Decimal": "42",
                "Binary": "42",
                "Octal": "42",
                "Hexadecimal": "42",
                "DigitSeparators": "1000000",
                "UnsignedLongLongSuffix": "42",
                "Maximum": "18446744073709551615",
            },
        ),
        (
            "SignedIntegerLiteralValues",
            {
                "Negative": "-42",
                "Minimum": "-9223372036854775808",
            },
        ),
        (
            "BooleanLiteralValues",
            {"False": "0", "True": "1"},
        ),
        (
            "CharacterLiteralValues",
            {
                "Basic": "65",
                "NewlineEscape": "10",
                "OctalEscape": "65",
                "HexadecimalEscape": "65",
                "Wide": "65",
                "Utf8": "65",
                "Utf16": "65",
                "Utf32": "65",
            },
        ),
        (
            "IntegralConstantExpressionValues",
            {
                "Arithmetic": "14",
                "Parenthesized": "20",
                "LeftShift": "16",
                "BitwiseOr": "7",
                "Comparison": "1",
                "Logical": "1",
                "Conditional": "9",
                "PreviousEnumerator": "10",
                "SizeOf": "4",
                "ExplicitlyConvertedFloat": "1",
            },
        ),
    ],
)
def test_enum_values_are_clang_evaluated(enums, enum_name, expected_values):
    enum = enums[enum_name]
    actual_values = dict(
        zip(enum.enumerators, enum.enumerator_values, strict=True)
    )

    assert actual_values == expected_values


@pytest.mark.parametrize(
    ("enum_name", "expected_type"),
    [
        ("BoolUnderlying", "bool"),
        ("CharUnderlying", "char"),
        ("SignedCharUnderlying", "signed char"),
        ("UnsignedCharUnderlying", "unsigned char"),
        ("ShortUnderlying", "short"),
        ("UnsignedShortUnderlying", "unsigned short"),
        ("IntUnderlying", "int"),
        ("UnsignedIntUnderlying", "unsigned int"),
        ("LongUnderlying", "long"),
        ("UnsignedLongUnderlying", "unsigned long"),
        ("LongLongUnderlying", "long long"),
        ("UnsignedLongLongUnderlying", "unsigned long long"),
        ("WCharUnderlying", "wchar_t"),
        ("Char16Underlying", "char16_t"),
        ("Char32Underlying", "char32_t"),
        ("Int8AliasUnderlying", "int8_t"),
        ("Uint8AliasUnderlying", "uint8_t"),
        ("Int16AliasUnderlying", "int16_t"),
        ("Uint16AliasUnderlying", "uint16_t"),
        ("Int32AliasUnderlying", "int32_t"),
        ("Uint32AliasUnderlying", "uint32_t"),
        ("Int64AliasUnderlying", "int64_t"),
        ("Uint64AliasUnderlying", "uint64_t"),
        ("UserDefinedAliasUnderlying", "unsigned int"),
    ],
)
def test_enum_reports_integral_underlying_type(enums, enum_name, expected_type):
    assert enums[enum_name].underlying_type.name == expected_type


def test_enum_supports_cpp20_char8_underlying_type():
    enums = _parse_enums(
        DATA_DIR / "sample_enum_char8.cu", cxx_standard="gnu++20"
    )

    assert enums["Char8Underlying"].underlying_type.name == "char8_t"
    assert enums["Char8Underlying"].enumerator_values == ["65"]


@pytest.mark.parametrize(
    "declaration",
    [
        "enum class FloatUnderlying : float { Value = 0 };",
        "enum class FloatingValue { Value = 1.5 };",
        'enum class StringValue { Value = "abc" };',
    ],
)
def test_enum_rejects_non_integral_types(tmp_path, declaration):
    source = tmp_path / "invalid_enum.cu"
    source.write_text(declaration)

    with pytest.raises(ParseError):
        _parse_enums(source)
