// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

template <typename T, int BLOCK_DIM_X> struct BlockScan {
  __device__ BlockScan() { printf("BlockScan constructor called\n"); }

  void __device__ InclusiveSum(T input, T &output) {
    printf("BlockScan InclusiveSum called\n");
  }
};

void __device__ foo() { BlockScan<int, 128> block_scan; }
