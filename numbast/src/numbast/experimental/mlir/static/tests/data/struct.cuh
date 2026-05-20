// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#pragma once

struct Foo {
public:
  int x;

  __device__ Foo();
  __device__ Foo(int x);
  __device__ int get_x();
  __device__ float add_one(float x);
  __device__ void print();

private:
  int y;

protected:
  int z;
};

struct Bar {
public:
  float x;

  __device__ Bar(int x);
  __device__ Bar(float x);
};

struct MyInt {
public:
  int x;

  __device__ MyInt(int x);
  __device__ operator int();
};
