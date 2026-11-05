// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#include <ast_canopy/ast_canopy.hpp>

namespace ast_canopy {

namespace {

/// Name of ``ED``, falling back to the typedef that names it.
///
/// ``typedef enum { ... } Foo;`` declares an enum with no identifier of its
/// own, so ``getNameAsString`` is empty and the declaration arrives unnamed and
/// unusable. The typedef is the only name such a type will ever have, and the
/// association is defined by the language rather than guessed: the typedef
/// name becomes the tag's name for linkage purposes, which is what
/// ``getTypedefNameForAnonDecl`` reports.
std::string enum_name(const clang::EnumDecl *ED) {
  std::string name = ED->getNameAsString();
  if (name.empty()) {
    if (const clang::TypedefNameDecl *TND = ED->getTypedefNameForAnonDecl()) {
      name = TND->getNameAsString();
    }
  }
  return name;
}

} // namespace

Enum::Enum(const clang::EnumDecl *ED)
    : name(enum_name(ED)),
      underlying_type(ED->getIntegerType(), ED->getASTContext()),
      qual_name(ED->getQualifiedNameAsString()) {
  if (qual_name.empty()) {
    qual_name = name;
  }
  for (const auto *enumerator : ED->enumerators()) {
    enumerators.push_back(enumerator->getNameAsString());

    auto const val = enumerator->getInitVal();
    llvm::SmallVector<char> buf;
    val.toString(buf);
    std::string s(buf.begin(), buf.end());

    enumerator_values.push_back(s);
  }
}

} // namespace ast_canopy
