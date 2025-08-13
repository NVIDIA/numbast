// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

// Allow implicit conversion from int
struct Foo {
  int x;
  __device__ Foo(int y) : x(y) {}
};

// Dis-allow implicit conversion from int
struct Bar {
  int x;
  __device__ explicit Bar(int y) : x(y) {}
};

__device__ bool useFoo(Foo f) {
  return true;
}

__device__ bool useBar(Bar b) {
  return true;
}
