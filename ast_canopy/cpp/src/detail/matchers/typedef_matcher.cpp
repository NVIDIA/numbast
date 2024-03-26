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

TypedefMatcher::TypedefMatcher(traverse_ast_payload *payload)
    : payload(payload) {}

void TypedefMatcher::run(const MatchFinder::MatchResult &Result) {
  const TypedefDecl *TD = Result.Nodes.getNodeAs<clang::TypedefDecl>("typedef");
  std::string file_name = source_filename_from_decl(TD);

  if (std::any_of(payload->files_to_retain->begin(),
                  payload->files_to_retain->end(),
                  [&file_name](const std::string &file_to_retain) {
                    return file_name == file_to_retain;
                  }))

  {
    std::string name = TD->getNameAsString();
    TypeSourceInfo *TSI = TD->getTypeSourceInfo();
    QualType underlying = TD->getUnderlyingType();

    if (underlying->isStructureOrClassType()) {
#ifndef NDEBUG
      std::string source_range =
          TD->getSourceRange().printToString(*Result.SourceManager);
      std::cout << source_range << std::endl;
      std::cout << std::endl;
#endif
      payload->decls->typedefs.push_back(
          Typedef(TD, payload->record_id_to_name));
    }
  }
}

} // namespace detail
} // namespace canopy
