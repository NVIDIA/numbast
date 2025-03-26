// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#pragma once

// Overloaded functions
int __device__ add(int a, int b);
float __device__ add(float a, float b);

// Different types
int __device__ minus_i32_f32(int a, float b);

// void return type
void __device__ set_42(int *val);

// operator overload on numba-cuda specific type (float2)
float2 __device__ operator+(const float2 &a, const float2 &b);
