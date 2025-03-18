// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#pragma once

struct Foo {
public:
  int x;

  __device__ Foo() : x(0) {}
  __device__ Foo(int x) : x(x) {}

private:
  int y;

protected:
  int z;
};

struct Bar {
public:
  float x;

  __device__ Bar(int x) : x(x) {}
  __device__ Bar(float x) : x(x) {}
};

struct MyInt {
public:
  int x;

  __device__ MyInt(int x) : x(x) {}
  __device__ operator int() { return x; }
};
