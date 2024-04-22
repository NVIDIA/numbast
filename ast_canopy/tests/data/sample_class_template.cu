// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on
enum class E { A, B, C };

template <typename T, E e> struct Foo {
  Foo() {}

  template <typename U> E bar(T t, U u) { return e; }

  void baz() {}

  T t;
};
