// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

// #include "struct.cuh" <-- this line is added at test time

__device__ Foo::Foo() : x(0) {}
__device__ Foo::Foo(int x) : x(x) {}

__device__ Bar::Bar(int x) : x(x) {}
__device__ Bar::Bar(float x) : x(x) {}

__device__ MyInt::MyInt(int x) : x(x) {}
__device__ MyInt::operator int() { return x; }
