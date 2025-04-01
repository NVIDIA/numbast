// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#include <ast_canopy/ast_canopy.hpp>

#include <iostream>
namespace ast_canopy {

ConstExprVar::ConstExprVar(const clang::VarDecl *VD)
    : type_(VD->getType(), VD->getASTContext()), name(VD->getNameAsString()) {
  if (!VD->isConstexpr()) {
    throw std::runtime_error("Not a constexpr variable");
  }

  clang::APValue *ap_value = VD->getEvaluatedValue();
  std::cout << ap_value->getKind() << std::endl;

  if (ap_value->isInt()) {
    value = std::to_string(ap_value->getInt().getExtValue());

  } else if (ap_value->isFloat()) {
    value = std::to_string(ap_value->getFloat().convertToDouble());
  } else {
    throw std::runtime_error("Unsupported constexpr vardecl value type.");
  }
}

} // namespace ast_canopy
