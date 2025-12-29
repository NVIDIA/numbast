// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#include <ast_canopy/ast_canopy.hpp>

#include <clang/AST/DeclTemplate.h>

#include <algorithm>

#include <iostream>

namespace ast_canopy {
ClassTemplate::ClassTemplate(const clang::ClassTemplateDecl *CTD)
    : Template(CTD->getTemplateParameters()),
      qual_name(CTD->getQualifiedNameAsString()),
      record(CTD->getTemplatedDecl(), RecordAncestor::ANCESTOR_IS_TEMPLATE) {
  if (qual_name.empty()) {
    qual_name = record.qual_name;
  }
}
} // namespace ast_canopy
