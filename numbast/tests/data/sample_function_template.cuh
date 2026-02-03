// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

// Simple templated device functions for function-template bindings.

template <typename T> __device__ T add(T a, T b) { return a + b; }

template <> __device__ int add<int>(int a, int b) { return a + b + 100; }

template <typename T> __device__ T add(T a, T b, T c) { return a + b + c; }

template <typename T> __device__ void add_out(T &out, T x) { out = x + 1; }

template <typename T> __device__ T add_out_ret(T &out, T x) {
  out = x + 2;
  return x + 3;
}

template <typename T, int N = 7> __device__ T add_default(T x) { return x + N; }

template <typename T, typename U> __device__ T add_cast(T a, U b) {
  return a + static_cast<T>(b);
}

template <typename T, int N = 5> __device__ T add_with_non_type(T a, T b) {
  return a + b + static_cast<T>(N);
}

template <typename T, typename U = int> __device__ T add_default_type(T a) {
  return a + static_cast<T>(sizeof(U));
}
