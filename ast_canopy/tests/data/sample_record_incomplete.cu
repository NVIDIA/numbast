// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: Apache-2.0

// Minimal repro for the dependent/incomplete type crash in
// ast_canopy/cpp/src/record.cpp.
//
// The Record constructor calls ctx.getTypeSize(type) /
// ctx.getTypeAlign(type) on class template specialisations that reach
// the matcher with ANCESTOR_IS_NOT_TEMPLATE. When the specialisation is
// still dependent or incomplete, Clang aborts inside getTypeSize.
// The fix guards on isDependentType()/isIncompleteType() and emits
// INVALID_SIZE_OF / INVALID_ALIGN_OF sentinels when layout cannot be
// computed.

#pragma once

// A forward declaration: `Fwd<int>` becomes an incomplete
// ClassTemplateSpecialization if anything forces it into the parsed AST.
template <typename T> struct Fwd;

// Wrapper that uses Fwd via pointer -- fine at C++ level but forces
// Clang to produce an (incomplete) Fwd<int> specialization node.
struct UsesIncomplete {
  Fwd<int> *ptr;
};

// A complete template instantiation in the same header, to prove the
// guard does not break the normal path.
template <typename T> struct Complete {
  T data;
};

struct UsesComplete {
  Complete<float> c;
};
