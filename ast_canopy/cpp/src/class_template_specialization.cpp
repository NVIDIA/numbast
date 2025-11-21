// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#include <ast_canopy/ast_canopy.hpp>

#include <clang/AST/DeclTemplate.h>
#include <llvm/ADT/APSInt.h>
#include <llvm/Support/raw_ostream.h>

namespace ast_canopy {

ClassTemplateSpecialization::ClassTemplateSpecialization(
    const clang::ClassTemplateSpecializationDecl *CTSD)
    : Record(llvm::cast<clang::CXXRecordDecl>(CTSD),
             RecordAncestor::ANCESTOR_IS_NOT_TEMPLATE),
      class_template(CTSD->getSpecializedTemplate()) {

  // Get the actual template arguments
  const auto &tparam_list = CTSD->getTemplateArgs();
  actual_template_arguments.reserve(tparam_list.size());

  for (auto i = 0; i < tparam_list.size(); i++) {
    const auto &targ = tparam_list[i];
    clang::TemplateArgument::ArgKind kind = targ.getKind();
    switch (kind) {
    case clang::TemplateArgument::ArgKind::Type:
      actual_template_arguments.push_back(targ.getAsType().getAsString());
      break;
    case clang::TemplateArgument::ArgKind::Integral: {
      llvm::APSInt integer = targ.getAsIntegral();
      llvm::SmallString<32> str; // -uint64_t::max() is 21 digits (with sign).
                                 // This should be enough.
      integer.toString(str);
      actual_template_arguments.push_back(str.c_str());
      break;
    }
    default:
      throw std::runtime_error("Unsupported template argument kind");
    }
  }
}

} // namespace ast_canopy
