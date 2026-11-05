// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

struct Foo {
  __device__ Foo() = default;
  __device__ Foo(int a) : a(a) {}
  __device__ int variadic_member(int value, ...);
  int a;
};

struct Bar {
  __device__ Bar() = default;
  __device__ Bar(int a) : a(a) {}
  int a;
};

__device__ Foo operator+(const Foo &a, const Foo &b) { return Foo(); }
__device__ Bar operator+(const Bar &a, const Bar &b) { return Bar(); }

namespace ns1 {
__device__ int inner_func(Foo a, Bar b) { return 0; }
} // namespace ns1

namespace ns2 {
__device__ int inner_func(Foo a, Bar b) { return 0; }
} // namespace ns2

extern "C" __device__ int c_device_func(int value) { return value; }
extern "C" int c_variadic_func(int value, ...);
