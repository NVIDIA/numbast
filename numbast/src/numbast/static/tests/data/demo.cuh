// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#include <cuda_fp16.h>

struct __attribute__((aligned(2))) __myfloat16 {
private:
  half data;

public:
  __host__ __device__ __myfloat16() : data(0) {}
  __host__ __device__ __myfloat16(double val) : data(static_cast<half>(val)) {}
  __host__ __device__ operator double() const {
    return static_cast<half>(data);
  }
};
