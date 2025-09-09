// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

// Functions.
[[maybe_unused]] int __attribute__((noinline)) func() { return 0; }

int func_noattr() { return 0; }

// Function templates.
template <typename> [[maybe_unused]] int __attribute__((noinline)) functempl() {
  return 0;
}

template <typename> int functempl_noattr() { return 0; }

struct A {
  // Methods.
  [[maybe_unused]] int __attribute__((noinline)) meth() { return 0; }

  int meth_noattr() { return 0; }

  // Method templates.
  template <typename> [[maybe_unused]] int __attribute__((noinline)) tmeth() {
    return 0;
  }

  template <typename> int tmeth_noattr() { return 0; }
};

template <typename> struct B {
  // Class template methods.
  [[maybe_unused]] int __attribute__((noinline)) meth() { return 0; }

  int meth_noattr() { return 0; }

  // Class template method templates.
  template <typename> [[maybe_unused]] int __attribute__((noinline)) tmeth() {
    return 0;
  }

  template <typename> int tmeth_noattr() { return 0; }
};
