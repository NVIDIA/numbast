// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#include "matchers.hpp"

#include <algorithm>

#ifndef NDEBUG
#include <iostream>
#endif

#include <ast_canopy/ast_canopy.hpp>

namespace ast_canopy {

namespace detail {

GlobalConstantCallback::GlobalConstantCallback(traverse_ast_payload *payload)
    : payload(payload) {}

void GlobalConstantCallback::run(const MatchFinder::MatchResult &Result) {
  const auto *VD = Result.Nodes.getNodeAs<clang::VarDecl>("global_constant");
  const std::string file_name = source_filename_from_decl(VD);
  if (std::find(payload->files_to_retain->begin(),
                payload->files_to_retain->end(),
                file_name) == payload->files_to_retain->end()) {
    return;
  }
  // Specialization names omit template arguments in EvaluatedConstDecl.
  if (llvm::isa<clang::VarTemplateSpecializationDecl>(VD)) {
    return;
  }
  if (!VD->hasInit() || VD->isStaticDataMember() ||
      !VD->getType().isConstQualified() ||
      VD->getType().isVolatileQualified() || VD->getType()->isDependentType() ||
      VD->getInit()->isValueDependent() ||
      VD->getDeclContext()->isDependentContext()) {
    return;
  }
  // Mutable globals, runtime initializers, pointers and aggregate objects must
  // not be turned into literal constants in a consuming language.
  const auto *value = VD->evaluateValue();
  // Clang also evaluates extension types such as _Float16. Only collect
  // floating-point types in the supported NVIDIA CUDA C++ language surface.
  if (value != nullptr && value->isFloat()) {
    const auto type = VD->getType();
    if (!type->isSpecificBuiltinType(clang::BuiltinType::Float) &&
        !type->isSpecificBuiltinType(clang::BuiltinType::Double) &&
        !type->isSpecificBuiltinType(clang::BuiltinType::LongDouble)) {
      return;
    }
  }
  if (value != nullptr && (value->isInt() || value->isFloat())) {
    payload->decls->global_constants.emplace_back(VD);
  }
}

ConstexprVarDeclCallback::ConstexprVarDeclCallback(
    vardecl_matcher_payload *payload)
    : payload(payload) {}

void ConstexprVarDeclCallback::run(const MatchFinder::MatchResult &Result) {
  const VarDecl *VD =
      Result.Nodes.getNodeAs<clang::VarDecl>("constexpr_vardecl");

  std::string name = VD->getNameAsString();
  if (name == payload->name_to_match) {
#ifndef NDEBUG
    std::cout << "Matched constexpr vardecl: " << name << std::endl;
#endif
    payload->var = std::move(EvaluatedConstDecl(VD));
  }
}

} // namespace detail

} // namespace ast_canopy
