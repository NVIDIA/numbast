// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: Apache-2.0

#ifndef BFLOAT16_CUH
#define BFLOAT16_CUH

#include <cuda_bf16.h>

nv_bfloat16 inline __device__ add(nv_bfloat16 a, nv_bfloat16 b) {
  return a + b;
}

__nv_bfloat16 inline __device__ add2(__nv_bfloat16 a, __nv_bfloat16 b) {
  return a + b;
}

__nv_bfloat16_raw inline __device__ bf16_to_raw(nv_bfloat16 a) {
  return __nv_bfloat16_raw(a);
}

nv_bfloat16 inline __device__ bf16_from_raw(__nv_bfloat16_raw a) {
  return __nv_bfloat16(a);
}

#endif
