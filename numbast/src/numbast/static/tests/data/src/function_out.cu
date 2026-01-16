// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

void __device__ add_out(int &out, int x) { out = x + 1; }

int __device__ add_out_ret(int &out, int x) {
  out = x + 2;
  return x + 3;
}
