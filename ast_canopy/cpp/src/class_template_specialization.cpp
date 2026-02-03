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
      clang::QualType T = targ.getIntegralType();

      // If this integral NTTP is actually an enum, try to recover the
      // enumerator name from the value.
      if (T->isEnumeralType()) {
        const auto *ET = T->getAs<clang::EnumType>();
        const clang::EnumDecl *ED = ET->getDecl();
        const llvm::APSInt &Val = targ.getAsIntegral();

        const clang::EnumConstantDecl *Matched = nullptr;
        for (const auto *ECD : ED->enumerators()) {
          if (ECD->getInitVal() == Val) {
            Matched = ECD;
            break;
          }
        }

        if (Matched) {
          // Use the fully-qualified enumerator name if you like,
          // or just ECD->getNameAsString() for the bare name.
          actual_template_arguments.push_back(
              Matched->getQualifiedNameAsString());
        } else {
          // No enumerator found with that value; fall back to integer.
          llvm::APSInt integer = Val;
          llvm::SmallString<32> str;
          integer.toString(str);
          actual_template_arguments.push_back(str.c_str());
        }
      } else {
        // Plain integral (not an enum type) — keep your existing behavior.
        llvm::APSInt integer = targ.getAsIntegral();
        llvm::SmallString<32> str; // -uint64_t::max() is 21 digits (with sign).
        integer.toString(str);
        actual_template_arguments.push_back(str.c_str());
      }

      break;
    }
    default:
      throw std::runtime_error("Unsupported template argument kind");
    }
  }
}

} // namespace ast_canopy
