// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

// #include "operator.cuh" <-- this line is added at test time

__device__ Foo::Foo() {}
__device__ Foo::Foo(int x) : x(x) {}

Foo __device__ operator+(const Foo &lhs, const Foo &rhs) {
  return Foo(lhs.x + rhs.x);
}
