// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

int __device__ prefix_foo(int a, int b) { return a + b; }

struct __internal__Foo {
  int x;
  __device__ __internal__Foo() : x(0) {}
  __device__ int get_x() { return x; }
};

enum __internal__Bar {
  BAR_A,
  BAR_B,
};
