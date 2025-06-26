// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

struct Foo {
  Foo() = default;
};

Foo __device__ __forceinline__ operator+(const Foo &a, const Foo &b) {
  return Foo();
}
