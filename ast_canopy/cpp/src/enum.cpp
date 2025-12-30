// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#include <ast_canopy/ast_canopy.hpp>

namespace ast_canopy {

Enum::Enum(const clang::EnumDecl *ED)
    : name(ED->getNameAsString()),
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
