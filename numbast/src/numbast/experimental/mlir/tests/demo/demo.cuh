// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#pragma once

#include <cuda_fp16.h>

// demo.cuh
struct __attribute__((aligned(2))) __myfloat16 {
private:
  half data;

public:
  __host__ __device__ __myfloat16();

  __host__ __device__ __myfloat16(double val);

  __host__ __device__ operator double() const;

  __host__ __device__ void print();
};

__host__ __device__ __myfloat16 operator+(const __myfloat16 &lh,
                                          const __myfloat16 &rh);

__device__ __myfloat16 hsqrt(const __myfloat16 a);

__host__ __device__ __myfloat16::__myfloat16() { data = 0x0; }

__host__ __device__ __myfloat16::__myfloat16(double val) {
  data = static_cast<half>(val);
}

__host__ __device__ __myfloat16::operator double() const {
  return static_cast<double>(data);
}

__host__ __device__ __myfloat16 operator+(const __myfloat16 &lh,
                                          const __myfloat16 &rh) {
  return __myfloat16(static_cast<double>(lh) + static_cast<double>(rh));
}

__device__ __myfloat16 hsqrt(const __myfloat16 a) {
  return __myfloat16(sqrt(static_cast<double>(a)));
}

__host__ __device__ void __myfloat16::print() {
  printf("__myfloat16: %f\n", static_cast<double>(data));
}
