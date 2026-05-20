// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#pragma once

void __device__ __forceinline__ foo(int a, int b);

void __device__ __forceinline__ foo(int a, int b);

void __device__ __forceinline__ foo(int a, int b) { return; }
