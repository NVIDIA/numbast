// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

int __device__ add(int a, int b) { return a + b; }
float __device__ add(float a, float b) { return a + b; }

int __device__ minus_i32_f32(int a, float b) { return a - static_cast<int>(b); }

void __device__ set_42(int *val) {
  *val = 42;
  return;
}

float2 __device__ operator+(const float2 &a, const float2 &b) {
  return {a.x + b.x, a.y + b.y};
}
