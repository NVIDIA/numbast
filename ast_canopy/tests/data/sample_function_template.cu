// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on
enum class E { A, B, C };

// Type Template, Non-type template
template <typename T, int N, E e> void __device__ foo(T t) {}

// Min required arg == 0?
template <typename T = int> void __device__ bar(T t) {}
