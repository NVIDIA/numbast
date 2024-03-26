// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#include "matchers.hpp"

namespace canopy {

namespace detail {

FunctionCallback::FunctionCallback(traverse_ast_payload *payload)
    : payload(payload) {
  payload->decls->functions.clear();
}

void FunctionCallback::run(const MatchFinder::MatchResult &Result) {
  const FunctionDecl *FD =
      Result.Nodes.getNodeAs<clang::FunctionDecl>("function");
  std::string file_name = source_filename_from_decl(FD);

  if (std::any_of(payload->files_to_retain->begin(),
                  payload->files_to_retain->end(),
                  [&file_name](const std::string &file_to_retain) {
                    return file_name == file_to_retain;
                  }))

  {
    if (!FD->isImplicit())
      payload->decls->functions.push_back(Function(FD));
  }
}

} // namespace detail

} // namespace canopy
