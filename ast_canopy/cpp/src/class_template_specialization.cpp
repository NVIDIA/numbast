// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#include <ast_canopy/ast_canopy.hpp>

#include <clang/AST/DeclTemplate.h>
#include <llvm/ADT/APSInt.h>
#include <llvm/ADT/SmallString.h>

namespace ast_canopy {

static std::string
integral_template_argument_to_string(const clang::TemplateArgument &targ) {
  clang::QualType T = targ.getIntegralType();

  if (T->isEnumeralType()) {
    const auto *ET = T->getAs<clang::EnumType>();
    const clang::EnumDecl *ED = ET->getDecl();
    const llvm::APSInt &Val = targ.getAsIntegral();

    for (const auto *ECD : ED->enumerators()) {
      if (ECD->getInitVal() == Val) {
        return ECD->getQualifiedNameAsString();
      }
    }
  }

  llvm::APSInt integer = targ.getAsIntegral();
  llvm::SmallString<32> str; // -uint64_t::max() is 21 digits (with sign).
  integer.toString(str);
  return std::string(str.c_str());
}

static std::string
template_argument_to_string(const clang::TemplateArgument &targ) {
  switch (targ.getKind()) {
  case clang::TemplateArgument::ArgKind::Type:
    return targ.getAsType().getAsString();
  case clang::TemplateArgument::ArgKind::Integral:
    return integral_template_argument_to_string(targ);
  case clang::TemplateArgument::ArgKind::Pack: {
    std::string result;
    bool first = true;
    for (const auto &packed_arg : targ.pack_elements()) {
      if (!first)
        result += ", ";
      first = false;
      result += template_argument_to_string(packed_arg);
    }
    return result;
  }
  default:
    // Gracefully handle template argument kinds we don't yet support
    // (e.g. Template, Expression, NullPtr, StructuralValue).
    return "<unsupported>";
  }
}

ClassTemplateSpecialization::ClassTemplateSpecialization(
    const clang::ClassTemplateSpecializationDecl *CTSD)
    : Record(llvm::cast<clang::CXXRecordDecl>(CTSD),
             RecordAncestor::ANCESTOR_IS_NOT_TEMPLATE),
      class_template(CTSD->getSpecializedTemplate()) {

  // Get the actual template arguments
  const auto &tparam_list = CTSD->getTemplateArgs();
  actual_template_arguments.reserve(tparam_list.size());

  for (unsigned i = 0; i < tparam_list.size(); i++) {
    const auto &targ = tparam_list[i];
    actual_template_arguments.push_back(template_argument_to_string(targ));
  }
}

} // namespace ast_canopy
