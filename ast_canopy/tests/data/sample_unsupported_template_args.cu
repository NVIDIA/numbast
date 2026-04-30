// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: Apache-2.0

// Minimal repro for handling Pack-kind template arguments in
// class_template_specialization.cpp.
//
// Parameter packs instantiate to a Pack-kind TemplateArgument on the
// specialization's getTemplateArgs() list. Prior to the fix, only Type
// and Integral kinds were handled, so any specialization that contained
// a Pack argument caused a std::runtime_error to propagate and abort parsing
// of the whole header.

#pragma once

// Variadic template -- instantiations produce a Pack-kind template
// argument that the old code did not handle.
template <typename... Ts> struct Variadic {
  static constexpr int count = sizeof...(Ts);
};

// Force an instantiation so a ClassTemplateSpecializationDecl exists
// in the parsed AST.
struct ForceInstantiation {
  Variadic<int, float, double> v;
};

// A plain, fully-supported specialization should still be parsed.
template <typename T, int N> struct Simple {
  T data[N];
};

struct ForceSimple {
  Simple<float, 3> s;
};
