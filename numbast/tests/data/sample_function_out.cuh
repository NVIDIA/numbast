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

__device__ void get_matrix(float out[12]) {
  for (int i = 0; i < 12; ++i) {
    out[i] = static_cast<float>(i) + 0.5f;
  }
}

__device__ void get_matrix_3x4(float out[3][4]) {
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 4; ++col) {
      out[row][col] = static_cast<float>(row * 4 + col) + 1.25f;
    }
  }
}

__device__ void get_data(float4 out[3]) {
  out[0] = make_float4(1.0f, 2.0f, 3.0f, 4.0f);
  out[1] = make_float4(5.0f, 6.0f, 7.0f, 8.0f);
  out[2] = make_float4(9.0f, 10.0f, 11.0f, 12.0f);
}
