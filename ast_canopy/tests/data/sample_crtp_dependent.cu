// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: Apache-2.0

// Minimal repro for the mangleName segfault in
// ast_canopy/cpp/src/function.cpp.
//
// The Itanium name mangler segfaults when handed a CXXMethodDecl whose
// signature is still template-dependent (parameter or return types that
// refer to template parameters not yet bound to concrete types). This
// happens for any method declared inside a class template, e.g. CRTP
// base classes or Eigen expression templates.
//
// The fix detects dependent signatures up-front with
// has_dependent_signature() and skips Itanium mangling, using an explicit
// dependent-signature fallback. It also plugs a leak by wrapping the
// ItaniumMangleContext in a unique_ptr.

#pragma once

// Case A: CRTP base with a method whose return type depends on the
// derived type. Parsing CRTPBase<T> was enough to segfault the mangler.
template <typename Derived> struct CRTPBase {
  __device__ Derived &derived() { return static_cast<Derived &>(*this); }
  __device__ const Derived &derived() const {
    return static_cast<const Derived &>(*this);
  }
  __device__ Derived add(const Derived &other) const { return derived(); }
};

template <typename Scalar> struct Vec3 : public CRTPBase<Vec3<Scalar>> {
  Scalar x, y, z;
  __host__ __device__ Vec3() : x(0), y(0), z(0) {}
};

// Case B: Function template whose return type depends on T::value_type.
template <typename T>
__device__ typename T::value_type extract_value(const T &container);

// A concrete device function that should still be parsed with a
// computed mangled name. Proves the non-dependent path is unchanged.
__device__ float vec3_dot(Vec3<float> a, Vec3<float> b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
