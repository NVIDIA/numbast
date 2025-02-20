// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#include "matchers.hpp"

namespace ast_canopy {

namespace detail {

FunctionCallback::FunctionCallback(traverse_ast_payload *payload)
    : payload(payload) {
  payload->decls->functions.clear();
}

void FunctionCallback::run(const MatchFinder::MatchResult &Result) {
  const FunctionDecl *FD =
      Result.Nodes.getNodeAs<clang::FunctionDecl>("function");
  std::string file_name = source_filename_from_decl(FD);

  bool should_add = false;

  // If file name is empty, assume this is a macro expanded function,
  // test if its prefix is in the allowlist.
  if (file_name.empty()) {
    std::string function_name = FD->getNameAsString();
    if (std::any_of(payload->prefixes_to_whitelist->begin(),
                    payload->prefixes_to_whitelist->end(),
                    [&function_name,
                     &should_add](const std::string &prefix_to_whitelist) {
                      return function_name.starts_with(prefix_to_whitelist);
                    })) {
      should_add = true;
    }
  } else {
    // If file name is not empty, test if the filename name is in the allowlist.
    if (std::any_of(payload->files_to_retain->begin(),
                    payload->files_to_retain->end(),
                    [&file_name](const std::string &file_to_retain) {
                      return file_name == file_to_retain;
                    }))

    {
      should_add = true;
    }
  }

  if (should_add) {
    if (!FD->isImplicit())
      payload->decls->functions.push_back(Function(FD));
  }
}

} // namespace detail

} // namespace ast_canopy
