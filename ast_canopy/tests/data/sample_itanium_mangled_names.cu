// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

struct Foo {
  Foo() = default;
};

struct Bar {
  Bar() = default;
};

Foo operator+(const Foo &a, const Foo &b) { return Foo(); }
Bar operator+(const Bar &a, const Bar &b) { return Bar(); }

namespace ns1 {
int inner_func(Foo a, Bar b) { return 0; }
} // namespace ns1

namespace ns2 {
int inner_func(Foo a, Bar b) { return 0; }
} // namespace ns2
