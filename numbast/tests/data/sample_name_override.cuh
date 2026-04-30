// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#pragma once

// A trivial class template whose specialization we want to bind with a
// custom C++ name. struct_decl.name for the specialization is just the
// unqualified template name ("Vec"), but in real shim code we need the
// fully qualified specialization (e.g. "demo::Vec<float, 3>").

namespace demo {
template <typename T, int N> struct Vec {
  T data[N];
  // A conversion operator forces the bind path that composes a shim
  // symbol from struct_name — that symbol must remain a valid C
  // identifier even when struct_name contains template syntax.
  __device__ operator T() const { return data[0]; }
};
} // namespace demo

// Force instantiation so the class_template_specialization shows up in
// the parsed declarations.
struct ForceInstantiation {
  demo::Vec<float, 3> v3f;
};
