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
