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

class CXXRecordDeclInCTPSDVisitor
    : public RecursiveASTVisitor<CXXRecordDeclInCTPSDVisitor> {
public:
  explicit CXXRecordDeclInCTPSDVisitor(
      ASTContext *Context,
      std::unordered_set<int64_t> *record_id_with_ctpsd_ancestor)
      : Context(Context),
        record_id_with_ctpsd_ancestor(record_id_with_ctpsd_ancestor) {}

  bool VisitCXXRecordDecl(CXXRecordDecl *Decl) {
    if (Decl->isThisDeclarationADefinition()) {
      int64_t ID = Decl->getID();
      record_id_with_ctpsd_ancestor->insert(ID);
    }
    return true; // Continue traversal
  }

private:
  ASTContext *Context;
  std::unordered_set<int64_t> *record_id_with_ctpsd_ancestor;
};

ClassTemplatePartialSpecializationCallback::
    ClassTemplatePartialSpecializationCallback(traverse_ast_payload *payload)
    : payload(payload) {}

void ClassTemplatePartialSpecializationCallback::run(
    const MatchFinder::MatchResult &Result) {
  const ClassTemplatePartialSpecializationDecl *CTPSD =
      Result.Nodes.getNodeAs<clang::ClassTemplatePartialSpecializationDecl>(
          "ctpsd");
  std::string file_name = source_filename_from_decl(CTPSD);

  if (std::any_of(payload->files_to_retain->begin(),
                  payload->files_to_retain->end(),
                  [&file_name](const std::string &file_to_retain) {
                    return file_name == file_to_retain;
                  }))

  {
    ASTContext &ctx = CTPSD->getASTContext();
    CXXRecordDeclInCTPSDVisitor visitor(&ctx,
                                        payload->record_id_with_ctpsd_ancestor);
    ClassTemplatePartialSpecializationDecl *CTPSD_ =
        const_cast<ClassTemplatePartialSpecializationDecl *>(CTPSD);
    visitor.TraverseDecl(CTPSD_);
  }
}

} // namespace detail

} // namespace ast_canopy
