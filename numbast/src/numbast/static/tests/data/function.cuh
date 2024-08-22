// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

// 2 Overloaded function
int __device__ add(int a, int b) { return a + b; }
float __device__ add(float a, float b) { return a + b; }
