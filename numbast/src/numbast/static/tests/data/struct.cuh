// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

struct Foo {
public:
  int x;

  __device__ Foo() {}
  __device__ Foo(int x) : x(x) {}

private:
  int y;

protected:
  int z;
};

struct Bar {
public:
  float x;

  __device__ Bar(float x) : x(x) {}
};
