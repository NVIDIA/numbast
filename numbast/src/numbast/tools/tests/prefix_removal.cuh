// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

enum __internal__Bar {
  BAR_A,
  BAR_B,
};

int __device__ prefix_foo(int a, int b) { return a + b; }

int __device__ baz(__internal__Bar bar) { return bar == BAR_A ? 1 : 0; }

struct __internal__Foo {
  int x;
  __device__ __internal__Foo() : x(0) {}
  __device__ int get_x() { return x; }
};
