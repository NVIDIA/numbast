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

RecordCallback::RecordCallback(traverse_ast_payload *payload)
    : payload(payload) {
  payload->decls->records.clear();
}

void RecordCallback::run(const MatchFinder::MatchResult &Result) {
  const CXXRecordDecl *RD =
      Result.Nodes.getNodeAs<clang::CXXRecordDecl>("record");
  std::string file_name = source_filename_from_decl(RD);

  if (std::any_of(payload->files_to_retain->begin(),
                  payload->files_to_retain->end(),
                  [&file_name](const std::string &file_to_retain) {
                    return file_name == file_to_retain;
                  }))

    if (RD->isThisDeclarationADefinition() && RD->isCompleteDefinition()) {
      // This is a complete definition, not a forward declaration.
      // An incomplete record definition can throw when we try to get its
      // size.
      auto id = RD->getID();

      // WAR: since
      // unless(hasAncestor(classTemplatePartialSpecializationDecl())) is not
      // functioning, we implemented a custom traversal in CTPSD matcher to log
      // all recordDecl. For any recordDecl that showed up inside CTPSD, we skip
      // them, as they are not top level decl.
      auto skip_set = payload->record_id_with_ctpsd_ancestor;
      if (skip_set->find(id) != skip_set->end())
        return;

      // Anonymous records may have an empty name from Clang
      // (`getNameAsString()` can return ""). In that case we assign a
      // placeholder `unnamed<ID>` so downstream always has something printable.
      // This placeholder is not stable across runs because it is derived from
      // Clang's internal Decl ID.
      std::string name = RD->getNameAsString();
      std::string reanme_for_unnamed =
          name.empty() ? "unnamed" + std::to_string(id) : name;

      auto &record_id_map = *payload->record_id_to_name;
      record_id_map[id] = reanme_for_unnamed;
      payload->decls->records.push_back(Record(
          RD, RecordAncestor::ANCESTOR_IS_NOT_TEMPLATE, reanme_for_unnamed));

#ifndef NDEBUG
      std::string source_range =
          RD->getSourceRange().printToString(*Result.SourceManager);
      std::cout << source_range << std::endl;
#endif
    }
}

} // namespace detail

} // namespace ast_canopy
