// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

namespace ns1 {
namespace ns2 {

enum E { A = 1 };

struct S {
  int m(int x) { return x; }
};

typedef S Alias;

int f(int a) { return a; }

template <typename T> T tf(T x) { return x; }

template <typename T> struct Tpl {
  T v;
};

} // namespace ns2
} // namespace ns1
