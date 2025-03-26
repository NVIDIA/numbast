// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

struct Foo {
public:
  int x;

  __device__ Foo();
  __device__ Foo(int x);
};

// Overloaded functions
Foo __device__ operator+(const Foo &lhs, const Foo &rhs);
