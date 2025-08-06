// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#include <clang/AST/Decl.h>
#include <clang/Basic/SourceManager.h>

#include <ast_canopy/ast_canopy.hpp>
#include <ast_canopy/source_location.hpp>

#include <iostream>

namespace ast_canopy {

Decl::Decl() : source_location(SourceLocation()) {}

Decl::Decl(const clang::Decl *decl) {
  clang::SourceLocation loc = clang::SourceLocation(decl->getLocation());
  bool is_valid = loc.isValid();
  std::cout << "is_valid: " << is_valid << std::endl;
  if (is_valid) {
    clang::SourceManager &SM = decl->getASTContext().getSourceManager();
    unsigned int row = SM.getSpellingLineNumber(loc);
    unsigned int col = SM.getSpellingColumnNumber(loc);
    llvm::StringRef file_name = SM.getFilename(loc);
    source_location = SourceLocation(file_name.str(), row, col, is_valid);
  } else {
    source_location = SourceLocation("", 0, 0, is_valid);
  }
}

} // namespace ast_canopy
