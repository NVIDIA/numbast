// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

// Simple out-parameter device functions for testing function bindings.

__device__ void add_out(int &out, int x) { out = x + 1; }

__device__ int add_out_ret(int &out, int x) {
  out = x + 2;
  return x + 3;
}

__device__ int add_in_ref(int &x) { return x + 5; }

static __device__ const float4 transform_rows[3] = {
    {1.0f, 2.0f, 3.0f, 4.0f},
    {5.0f, 6.0f, 7.0f, 8.0f},
    {9.0f, 10.0f, 11.0f, 12.0f},
};

__device__ const float4 *get_transform(int handle) {
  (void)handle;
  return transform_rows;
}
