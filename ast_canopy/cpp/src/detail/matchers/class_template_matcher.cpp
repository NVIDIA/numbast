// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#include "matchers.hpp"

#ifndef NDEBUG
#include <iostream>
#endif

namespace ast_canopy {

namespace detail {

ClassTemplateCallback::ClassTemplateCallback(traverse_ast_payload *payload)
    : payload(payload) {
  payload->decls->function_templates.clear();
}

void ClassTemplateCallback::run(const MatchFinder::MatchResult &Result) {
  const ClassTemplateDecl *CTD =
      Result.Nodes.getNodeAs<clang::ClassTemplateDecl>("class_template");
  std::string file_name = source_filename_from_decl(CTD);

  if (std::any_of(payload->files_to_retain->begin(),
                  payload->files_to_retain->end(),
                  [&file_name](const std::string &file_to_retain) {
                    return file_name == file_to_retain;
                  }))

  {

#ifndef NDEBUG
    std::string source_range =
        CTD->getSourceRange().printToString(*Result.SourceManager);
    std::cout << source_range << std::endl;
    std::cout << CTD->getNameAsString() << std::endl;
#endif

    if (!CTD->isImplicit()) {
      std::vector<std::string> parent_record_names_stack;
      payload->decls->class_templates.push_back(
          ClassTemplate(CTD, parent_record_names_stack));
    }
  }
}

} // namespace detail

} // namespace ast_canopy
