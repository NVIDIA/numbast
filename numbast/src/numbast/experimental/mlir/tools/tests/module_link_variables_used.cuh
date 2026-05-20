// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

extern "C" {
__device__ int retained_global = 7;
__device__ int another_retained_global = 11;
}

int __device__ test_function(int a, int b) { return a + b; }
