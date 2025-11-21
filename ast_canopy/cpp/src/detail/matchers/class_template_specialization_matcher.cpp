// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#include "matchers.hpp"

#include <clang/AST/ASTContext.h>
#include <clang/AST/RecursiveASTVisitor.h>

#ifndef NDEBUG
#include <iostream>
#endif

namespace ast_canopy {

namespace detail {

using namespace clang;

ClassTemplateSpecializationCallback::ClassTemplateSpecializationCallback(
    traverse_ast_payload *payload)
    : payload(payload) {}

void ClassTemplateSpecializationCallback::run(
    const MatchFinder::MatchResult &Result) {

  const ClassTemplateSpecializationDecl *CTSD =
      Result.Nodes.getNodeAs<clang::ClassTemplateSpecializationDecl>("ctsd");

  if (llvm::isa<clang::ClassTemplatePartialSpecializationDecl>(CTSD)) {
    // Notice that CTPSD is-a CTSD, we cannot materialize it, otherwise infinite
    // recursion will happen with getTypeInfo().
    return;
  }

  std::string file_name = source_filename_from_decl(CTSD);

  if (std::any_of(payload->files_to_retain->begin(),
                  payload->files_to_retain->end(),
                  [&file_name](const std::string &file_to_retain) {
                    return file_name == file_to_retain;
                  }))

  {

    if (!CTSD->isImplicit())
      payload->decls->class_template_specializations.push_back(
          ClassTemplateSpecialization(CTSD));
  }
}

} // namespace detail

} // namespace ast_canopy
