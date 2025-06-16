// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#include <ast_canopy/ast_canopy.hpp>

namespace ast_canopy {

Enum::Enum(const std::string &name, const std::vector<std::string> &enumerators,
           const std::vector<std::string> &enumerator_values,
           const std::vector<std::string> &namespace_stack)
    : Declaration(namespace_stack), name(name), enumerators(enumerators),
      enumerator_values(enumerator_values) {}

Enum::Enum(const clang::EnumDecl *ED)
    : Declaration(ED), name(ED->getNameAsString()) {
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
