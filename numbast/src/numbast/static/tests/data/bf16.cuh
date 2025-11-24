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

#endif
