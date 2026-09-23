// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#include <stdint.h>

enum class IntegerLiteralValues : unsigned long long {
  Decimal = 42,
  Binary = 0b101010,
  Octal = 052,
  Hexadecimal = 0x2a,
  DigitSeparators = 1'000'000,
  UnsignedLongLongSuffix = 42ULL,
  Maximum = 0xffffffffffffffffULL,
};

enum class SignedIntegerLiteralValues : long long {
  Negative = -42,
  Minimum = (-9223372036854775807LL - 1),
};

enum class BooleanLiteralValues : bool {
  False = false,
  True = true,
};

enum class CharacterLiteralValues : char32_t {
  Basic = 'A',
  NewlineEscape = '\n',
  OctalEscape = '\101',
  HexadecimalEscape = '\x41',
  Wide = L'A',
  Utf8 = u8'A',
  Utf16 = u'A',
  Utf32 = U'A',
};

enum class IntegralConstantExpressionValues : int {
  Arithmetic = 2 + 3 * 4,
  Parenthesized = (2 + 3) * 4,
  LeftShift = 1 << 4,
  BitwiseOr = 0x3 | 0x4,
  Comparison = 3 > 2,
  Logical = false || true,
  Conditional = true ? 9 : 10,
  PreviousEnumerator = Conditional + 1,
  SizeOf = sizeof(int),
  ExplicitlyConvertedFloat = static_cast<int>(1.5),
};

enum class BoolUnderlying : bool { Value = false };
enum class CharUnderlying : char { Value = 0 };
enum class SignedCharUnderlying : signed char { Value = 0 };
enum class UnsignedCharUnderlying : unsigned char { Value = 0 };
enum class ShortUnderlying : short { Value = 0 };
enum class UnsignedShortUnderlying : unsigned short { Value = 0 };
enum class IntUnderlying : int { Value = 0 };
enum class UnsignedIntUnderlying : unsigned int { Value = 0 };
enum class LongUnderlying : long { Value = 0 };
enum class UnsignedLongUnderlying : unsigned long { Value = 0 };
enum class LongLongUnderlying : long long { Value = 0 };
enum class UnsignedLongLongUnderlying : unsigned long long { Value = 0 };
enum class WCharUnderlying : wchar_t { Value = 0 };
enum class Char16Underlying : char16_t { Value = 0 };
enum class Char32Underlying : char32_t { Value = 0 };

enum class Int8AliasUnderlying : int8_t { Value = 0 };
enum class Uint8AliasUnderlying : uint8_t { Value = 0 };
enum class Int16AliasUnderlying : int16_t { Value = 0 };
enum class Uint16AliasUnderlying : uint16_t { Value = 0 };
enum class Int32AliasUnderlying : int32_t { Value = 0 };
enum class Uint32AliasUnderlying : uint32_t { Value = 0 };
enum class Int64AliasUnderlying : int64_t { Value = 0 };
enum class Uint64AliasUnderlying : uint64_t { Value = 0 };

using user_defined_enum_storage_t = unsigned int;
enum class UserDefinedAliasUnderlying : user_defined_enum_storage_t {
  Value = 0,
};
