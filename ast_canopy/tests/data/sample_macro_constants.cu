// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: Apache-2.0

#include "sample_macro_defines_include.cuh"

struct MacroLayout {
  int value;
  double other;
};
enum class MacroStatus : unsigned { ready = 7 };
constexpr int macro_square(int x) { return x * x; }
constexpr int macro_constant = 11;
extern int macro_runtime;
int macro_runtime_call();

#define MACRO_INTEGER 42
#define MACRO_UNSIGNED 18446744073709551615ULL
#define MACRO_NEGATIVE (-23)
#define MACRO_MASK (1ULL << 40)
#define MACRO_ALIAS MACRO_MASK
#define MACRO_COMBINED (MACRO_ALIAS | 3)
#define MACRO_SIZE sizeof(MacroLayout)
#define MACRO_VERSION ((2 << 16) + sizeof(MacroLayout))
#define MACRO_CAST static_cast<unsigned short>(65535)
#define MACRO_BOOL (MACRO_INTEGER == 42)
#define MACRO_FLOAT 1.2345678901234567
#define MACRO_ENUM MacroStatus::ready
#define MACRO_CONSTEXPR_CALL macro_square(9)
#define MACRO_CONSTANT_VARIABLE macro_constant
#define MACRO_TWICE(x) ((x) * 2)
#define MACRO_FUNCTION_EXPANSION MACRO_TWICE(21)
#define MACRO_INCLUDED (INCLUDED_VALUE + 1)
#define MACRO_EMPTY
#define MACRO_STRING "text"
#define MACRO_TYPE unsigned long
#define MACRO_UNKNOWN undefined_macro_identifier
#define MACRO_RUNTIME macro_runtime
#define MACRO_RUNTIME_CALL macro_runtime_call()
#define MACRO_SIDE_EFFECT (++macro_runtime, 42)
#define MACRO_DIV_ZERO (1 / 0)
#define MACRO_TRAILING 42 trailing_tokens
#define MACRO_BROKEN (1 +)
#define MACRO_POINTER nullptr
#define MACRO_AGGREGATE MacroLayout{1, 2.0}
#define MACRO_RECURSIVE MACRO_RECURSIVE
#define MACRO_REMOVED 12
#undef MACRO_REMOVED
#define MACRO_REDEFINED 1
#undef MACRO_REDEFINED
#define MACRO_REDEFINED 2
#ifdef MACRO_BUILD_SETTING
#define MACRO_PROFILE (MACRO_BUILD_SETTING + __CUDA_ARCH__)
#endif
// An invalid definition must not prevent later valid expressions evaluating.
#define Z_MACRO_AFTER_ERRORS 99
