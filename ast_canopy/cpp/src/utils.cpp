// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#include <ast_canopy/utils.hpp>
#include <clang/AST/Decl.h>

namespace ast_canopy {

std::vector<std::string> extract_namespace_stack(const clang::Decl *decl) {
  std::vector<std::string> namespace_stack;

  // Get the declaration context
  const clang::DeclContext *context = decl->getDeclContext();

  // Traverse up the context chain to collect namespaces
  while (context) {
    if (const clang::NamespaceDecl *ns =
            clang::dyn_cast<clang::NamespaceDecl>(context)) {
      // For anonymous namespaces, add an empty string
      if (ns->isAnonymousNamespace()) {
        namespace_stack.push_back("");
      } else {
        namespace_stack.push_back(ns->getNameAsString());
      }
    }
    context = context->getParent();
  }

  return namespace_stack;
}

} // namespace ast_canopy
