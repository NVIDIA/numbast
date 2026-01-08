// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#pragma once

// A minimal class-template + templated-method playground for unit tests.
//
// Goals:
// - Exercise templated member functions where:
//   - template args are required (non-deducible, no defaults)
//   - template args can be partially/fully specified (future Python API)
//   - template args are not fully specified but defaults make the call valid
//
// NOTE: Templated-method bindings currently assume `void` return types.

template <typename T, int N = 5> struct TMix {
  __device__ TMix() {}

  // Requires explicit template arg M. (M is non-type and not deducible from
  // args.)
  template <int M, typename U> __device__ void AddConst(U x, U *out) const {
    *out = static_cast<U>(x + static_cast<U>(M));
  }

  // Same shape, but does *not* require explicit specialization:
  // - M defaults to N
  // - U defaults to T
  template <int M = N, typename U = T>
  __device__ void AddConstDefault(U x, U *out) const {
    *out = static_cast<U>(x + static_cast<U>(M));
  }
};
