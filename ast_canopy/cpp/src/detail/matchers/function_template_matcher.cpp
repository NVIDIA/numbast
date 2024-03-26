// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#include "matchers.hpp"

#ifndef NDEBUG
#include <iostream>
#endif

namespace canopy {

namespace detail {

FunctionTemplateCallback::FunctionTemplateCallback(
    traverse_ast_payload *payload)
    : payload(payload) {
  payload->decls->function_templates.clear();
}

void FunctionTemplateCallback::run(const MatchFinder::MatchResult &Result) {
  const FunctionTemplateDecl *FTD =
      Result.Nodes.getNodeAs<clang::FunctionTemplateDecl>("function_template");
  std::string file_name = source_filename_from_decl(FTD);

  if (std::any_of(payload->files_to_retain->begin(),
                  payload->files_to_retain->end(),
                  [&file_name](const std::string &file_to_retain) {
                    return file_name == file_to_retain;
                  }))

  {

#ifndef NDEBUG
    std::string source_range =
        FTD->getSourceRange().printToString(*Result.SourceManager);
    std::cout << source_range << std::endl;
    std::cout << FTD->getNameAsString() << std::endl;
#endif

    if (!FTD->isImplicit())
      payload->decls->function_templates.push_back(FunctionTemplate(FTD));
  }
}

} // namespace detail

} // namespace canopy
