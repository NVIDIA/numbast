// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#include <ast_canopy/ast_canopy.hpp>

#include <iostream>

namespace ast_canopy {

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
      canonical_type.getNonReferenceType().getUnqualifiedType().getAsString(pp);

  _is_right_reference = canonical_type->isRValueReferenceType();
  _is_left_reference = canonical_type->isLValueReferenceType();
}

} // namespace ast_canopy
