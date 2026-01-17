// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#pragma once

// Out-parameter functions for static binding tests.
void __device__ add_out(int &out, int x);
int __device__ add_out_ret(int &out, int x);
int __device__ add_in_ref(int &x);
void __device__ add_inout_ref(int &x, int delta);
