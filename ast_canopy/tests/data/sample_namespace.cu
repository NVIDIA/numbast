// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

// 1. No namespace
struct NoNamespaceStruct {
  int x;
  __device__ void foo() {}
};

template <typename T> __device__ void no_namespace_func(T x) {}

// 2. Flat namespace
namespace flat {
struct FlatStruct {
  int y;
  __device__ void bar() {}
};

template <typename T> __device__ void flat_func(T x) {}

enum FlatEnum { A = 1, B = 2 };
} // namespace flat

// 3. Nested namespace
namespace outer {
namespace inner {
struct NestedStruct {
  int z;
  __device__ void baz() {}
};

template <typename T> __device__ void nested_func(T x) {}

} // namespace inner
} // namespace outer

// 4. Nested namespace with anonymous namespace
namespace {
struct AnonymousStruct {
  int w;
  __device__ void qux() {}
};
} // namespace

namespace outer {
namespace {
struct OuterAnonymousStruct {
  int v;
  __device__ void quux() {}
};
} // namespace
} // namespace outer
