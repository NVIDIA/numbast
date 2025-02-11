// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

// Overloaded functions
int __device__ add(int a, int b) { return a + b; }
float __device__ add(float a, float b) { return a + b; }

// Different types
int __device__ minus_i32_f32(int a, float b) { return a - static_cast<int>(b); }

// void return type
void __device__ set_42(int *val) {
  *val = 42;
  return;
}
