// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#define DECLARE_ATTR(Name)                                                     \
  [[maybe_unused]] int __attribute__((noinline)) Name() { return 0; }
#define DECLARE_NOATTR(Name)                                                   \
  int Name##_noattr() { return 0; } // Note: name ends with "noattr"

// Functions.
DECLARE_ATTR(func)
DECLARE_NOATTR(func)

// Function templates.
template <typename>
DECLARE_ATTR(functempl)
template <typename>
DECLARE_NOATTR(func)

struct A {
  // Methods.
  DECLARE_ATTR(meth)
  DECLARE_NOATTR(meth)
  // Method templates.
  template <typename>
  DECLARE_ATTR(tmeth)
  template <typename>
  DECLARE_NOATTR(tmeth)
};

template <typename> struct B {
  // Class template methods.
  DECLARE_ATTR(meth)
  DECLARE_NOATTR(meth)
  // Class template method templates.
  template <typename>
  DECLARE_ATTR(tmeth)
  template <typename>
  DECLARE_NOATTR(tmeth)
};
