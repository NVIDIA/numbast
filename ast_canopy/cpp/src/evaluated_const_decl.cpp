// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#include <ast_canopy/ast_canopy.hpp>

#include <llvm/ADT/SmallString.h>
namespace ast_canopy {

EvaluatedConstDecl::EvaluatedConstDecl(const clang::VarDecl *VD)
    : is_constexpr(VD->isConstexpr()),
      type_(VD->getType(), VD->getASTContext()), name(VD->getNameAsString()),
      qual_name(VD->getQualifiedNameAsString()) {
  if (qual_name.empty()) {
    qual_name = name;
  }
  if (!VD->getType().isConstQualified()) {
    throw std::runtime_error("Not a constant variable");
  }

  clang::APValue *ap_value = VD->evaluateValue();

  if (ap_value == nullptr) {
    throw std::runtime_error("Evaluation is not successful.");
  }

  llvm::SmallString<32> serialized;
  if (ap_value->isInt()) {
    ap_value->getInt().toString(serialized, 10);
  } else if (ap_value->isFloat()) {
    ap_value->getFloat().toString(serialized);
  } else {
    throw std::runtime_error("Unsupported evaluated constant value type.");
  }
  value = std::string(serialized);
}

} // namespace ast_canopy
