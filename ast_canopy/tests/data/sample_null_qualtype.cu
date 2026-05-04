// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: Apache-2.0

// Minimal repro for the null-QualType crash in ast_canopy/cpp/src/type.cpp.
//
// When a field's type refers to an uninstantiated dependent member of a
// template parameter (e.g. `typename T::nested`), Clang may hand the
// ast_canopy Type constructor a null QualType during error recovery /
// bypass_parse_error parsing. Before the fix, the subsequent call to
// qualtype.getAsString() / qualtype.getCanonicalType() segfaulted.
//
// The fix adds an `if (qualtype.isNull()) { ... return; }` guard that
// substitutes a "<null-type>" placeholder so parsing can continue.

#pragma once

// A trait-style template whose member field depends on T::nested --
// deeply dependent types that aren't concretely instantiated here push
// Clang's type resolution into edge cases that historically produced
// null QualTypes under bypass_parse_error.
template <typename T> struct DependentFieldTrait {
  typename T::nested field_a;
  typename T::template rebind<int>::other field_b;
};

// A plain concrete struct alongside, to assert the rest of the header
// still parses.
struct Plain {
  int x;
  float y;
};
