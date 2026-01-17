// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

// Simple mutative device functions for testing function bindings.

__device__ void add_one_inplace(int &x) { x += 1; }

__device__ void set_42(int &x) { x = 42; }
