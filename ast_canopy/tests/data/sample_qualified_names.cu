// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#include <stdint.h>

// Global function
void __device__ global_func(uint64_t a) {}

namespace outer {
// Function in outer namespace
void __device__ outer_func(uint64_t a) {}

namespace inner {
// Function in nested namespace
void __device__ inner_func(uint64_t a) {}

// Struct with method in nested namespace
struct NestedStruct {
  void __device__ struct_method(uint64_t a) {}
};
} // namespace inner
} // namespace outer

// Struct with method in global namespace
struct GlobalStruct {
  void __device__ struct_method(uint64_t a) {}
};
