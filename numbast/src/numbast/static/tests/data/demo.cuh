// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

struct __attribute__((aligned(2))) myfloat8 {
private:
  unsigned char data;

public:
  __host__ __device__ myfloat8() {}
  __host__ __device__ myfloat8(double val) {
    this->data = static_cast<unsigned char>(val);
  }
  __host__ __device__ operator double() const {
    return static_cast<double>(this->data);
  }
};
