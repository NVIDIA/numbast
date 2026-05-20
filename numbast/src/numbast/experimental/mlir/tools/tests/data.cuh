// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

struct Foo {
public:
  int x;
  __device__ Foo() : x(0) {}
};

typedef Foo Bar;

int __device__ add(int a, int b) { return a + b; }

#if (defined(__CUDACC__) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 860)))

int __device__ mul(int a, int b) { return a * b; }

#endif
