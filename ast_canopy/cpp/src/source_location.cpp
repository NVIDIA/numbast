// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#include <ast_canopy/source_location.hpp>

#include <clang/AST/ASTContext.h>
#include <clang/Basic/SourceManager.h>

namespace ast_canopy {

SourceLocation::SourceLocation() : _is_valid(false) {}

SourceLocation::SourceLocation(const std::string &file_name,
                               const unsigned int line,
                               const unsigned int column, const bool is_valid)
    : _file_name(file_name), _line(line), _column(column), _is_valid(is_valid) {
}

namespace detail {

SourceLocation location_from_decl(const clang::Decl *decl) {
  ast_canopy::SourceLocation res;
  clang::SourceLocation loc = clang::SourceLocation(decl->getLocation());
  bool is_valid = loc.isValid();
  if (is_valid) {
    clang::SourceManager &SM = decl->getASTContext().getSourceManager();
    unsigned int row = SM.getSpellingLineNumber(loc);
    unsigned int col = SM.getSpellingColumnNumber(loc);
    llvm::StringRef file_name = SM.getFilename(loc);
    res = SourceLocation(file_name.str(), row, col, is_valid);
  } else {
    res = SourceLocation("", 0, 0, is_valid);
  }
  return res;
}

} // namespace detail

} // namespace ast_canopy
