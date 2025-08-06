// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: Apache-2.0

// clang-format off

void __device__ __forceinline__ foo() {} // line 6

struct Bar {    // line 8
    Bar() {}    // line 9
};

template <typename T>
void __device__ __forceinline__ baz() {} // line 13

template <typename T>
struct Bax {}; // line 16

// clang-format on
