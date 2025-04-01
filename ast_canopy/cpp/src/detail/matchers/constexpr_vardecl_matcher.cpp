// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#include "matchers.hpp"

#ifndef NDEBUG
#include <iostream>
#endif

#include <ast_canopy/ast_canopy.hpp>

namespace ast_canopy {

namespace detail {

ConstexprVarDeclCallback::ConstexprVarDeclCallback(
    vardecl_matcher_payload *payload)
    : payload(payload) {}

void ConstexprVarDeclCallback::run(const MatchFinder::MatchResult &Result) {
  const VarDecl *VD =
      Result.Nodes.getNodeAs<clang::VarDecl>("constexpr_vardecl");

#ifndef NDEBUG
  std::cout << "ConstexprVarDeclCallback::run: " << VD->getNameAsString()
            << std::endl;
#endif

  std::string name = VD->getNameAsString();
  if (name == payload->name_to_match)
    payload->var = std::move(ConstExprVar(VD));
}

} // namespace detail

} // namespace ast_canopy
