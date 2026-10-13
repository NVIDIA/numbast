// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#pragma once

__device__ int acmeCompute(int x);

__device__ int acmeExcluded(int x);

__device__ int acmeCooperativeReduce(int x);

// Matched by a `skip_prefix` of "internal".
__device__ int internalHelper(int x);

// Host-only: no execution space annotation.
extern int acmeHostOnly(int x);

// Both host-only and matched by a `skip_prefix` of "internal", so the two
// reasons compete.
extern int internalHostOnly(int x);
