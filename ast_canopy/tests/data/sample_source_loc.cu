// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: Apache-2.0

// clang-format off

void __device__ __forceinline__ foo() {} // line 6

struct Watermelon {    // line 8
    Watermelon() {}    // line 9
};

template <typename T>                    // line 12
void __device__ __forceinline__ baz() {} // line 13

template <typename T>   // line 15
struct Bax {};          // line 16

enum class Fruit {}; // line 18

typedef Watermelon Suika; // line 20

// clang-format on
