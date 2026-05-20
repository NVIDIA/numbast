// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

struct Foo {
public:
  int x;

  __device__ Foo() : x(42) {}

  __device__ int get_x() { return x; }

  __device__ float add_one(float x) { return x + 1; }

  __device__ void print() { printf("Foo: %d\n", x); }

private:
  int y;

protected:
  int z;
};
