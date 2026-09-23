// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

// This is separate from sample_enum_values.cu because AST Canopy defaults to
// GNU C++17, where char8_t is not a built-in type. The other enum cases remain
// valid in C++20; keeping them separate preserves coverage of the default mode.
enum class Char8Underlying : char8_t { Value = u8'A' };
