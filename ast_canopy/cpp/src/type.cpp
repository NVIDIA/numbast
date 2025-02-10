// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#include <ast_canopy/ast_canopy.hpp>

#include <iostream>

namespace ast_canopy {

std::string
remove_qualifier_recursive_to_name(clang::QualType const &ty,
                                   const clang::PrintingPolicy &pp) {
  const clang::QualType unqualified =
      ty.getNonReferenceType().getUnqualifiedType();
  const clang::Type *underlying = unqualified.getTypePtrOrNull();

  if (underlying == nullptr) {
    std::cout << "Warning: empty pointee type pointer encountered."
              << std::endl;
    return "<error-type>";
  }

  if (!underlying->isPointerType()) {
    return unqualified.getAsString(pp);
  }

  const clang::QualType pointee_type = underlying->getPointeeType();
  std::string underlying_removed =
      remove_qualifier_recursive_to_name(pointee_type, pp);
  return underlying_removed + " *";
}

Type::Type(std::string name, std::string unqualified_non_ref_type_name,
           bool is_right_reference, bool is_left_reference)
    : name(std::move(name)),
      unqualified_non_ref_type_name(std::move(unqualified_non_ref_type_name)),
      _is_right_reference(is_right_reference),
      _is_left_reference(is_left_reference) {}

Type::Type(const clang::QualType &qualtype, const clang::ASTContext &context) {

  clang::QualType canonical_type = qualtype.getCanonicalType();
  clang::PrintingPolicy pp{context.getLangOpts()};

  name = canonical_type.getAsString(pp);

  unqualified_non_ref_type_name =
      remove_qualifier_recursive_to_name(canonical_type, pp);

  _is_right_reference = canonical_type->isRValueReferenceType();
  _is_left_reference = canonical_type->isLValueReferenceType();
}

} // namespace ast_canopy
