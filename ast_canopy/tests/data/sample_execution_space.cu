// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on
void __device__ dfoo() {}
void __host__ hfoo() {}
void __global__ gfoo() {}
void __device__ __host__ dhfoo() {}
void foo() {}

struct Bar {
  void __device__ dfoo() {}
  void __host__ hfoo() {}
  void __device__ __host__ dhfoo() {}
  void foo() {}
};
