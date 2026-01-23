// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#pragma once

template <typename T, int BLOCK_DIM_X> struct BlockScan {
  __device__ BlockScan() { printf("BlockScan constructor called\n"); }

  void __device__ InclusiveSum(T input, T *output) {
    printf("BlockScan InclusiveSum called\n");
  }

  __device__ void AddToRef(T input, T &out) { out = static_cast<T>(input + 1); }
};

template <int N, typename T> class Foo {
public:
  T t;
  __device__ Foo(T t) : t(t), t2(static_cast<T>(N)) {}

  __device__ T get_t() { return t; }
  __device__ T get_t2() { return t2; }

private:
  T t2;
};
