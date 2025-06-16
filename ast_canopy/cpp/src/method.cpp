// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#include <ast_canopy/ast_canopy.hpp>

#include <clang/AST/DeclCXX.h>

namespace ast_canopy {

Method::Method(const std::string &name, const Type &return_type,
               const std::vector<ParamVar> &params,
               const execution_space &exec_space, const method_kind &kind,
               const std::vector<std::string> &namespace_stack)
    : Function(name, return_type, params, exec_space, namespace_stack),
      kind(kind) {}

Method::Method(const clang::CXXMethodDecl *MD) : Function(MD) {
  if (const clang::CXXConstructorDecl *CD =
          clang::dyn_cast<clang::CXXConstructorDecl>(MD)) {
    if (CD->isDefaultConstructor())
      kind = method_kind::default_constructor;
    else if (CD->isCopyConstructor())
      kind = method_kind::copy_constructor;
    else if (CD->isMoveConstructor())
      kind = method_kind::move_constructor;
    else
      kind = method_kind::other_constructor;
  } else if (clang::dyn_cast<clang::CXXDestructorDecl>(MD)) {
    kind = method_kind::destructor;
  } else if (clang::dyn_cast<clang::CXXConversionDecl>(MD)) {
    kind = method_kind::conversion_function;
  } else {
    kind = method_kind::other;
  }
}

bool Method::is_move_constructor() {
  return kind == method_kind::move_constructor;
}

} // namespace ast_canopy
