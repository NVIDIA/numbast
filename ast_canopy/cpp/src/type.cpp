// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#include <ast_canopy/ast_canopy.hpp>

#include <iostream>

namespace ast_canopy {

namespace detail {

/**
 * @brief Check if a type name contains a stdint type
 *
 * @param type_name The type name to check
 * @return true If the type name contains a type
 * @return false If the type name does not contain a type
 */
bool contains_stdint_type(const std::string &type_name) {
  static const std::vector<std::string> stdintTypes = {
      "int8_t",        "uint8_t",        "int16_t",       "uint16_t",
      "int32_t",       "uint32_t",       "int64_t",       "uint64_t",
      "intptr_t",      "uintptr_t",      "intmax_t",      "uintmax_t",
      "int_fast8_t",   "uint_fast8_t",   "int_fast16_t",  "uint_fast16_t",
      "int_fast32_t",  "uint_fast32_t",  "int_fast64_t",  "uint_fast64_t",
      "int_least8_t",  "uint_least8_t",  "int_least16_t", "uint_least16_t",
      "int_least32_t", "uint_least32_t", "int_least64_t", "uint_least64_t"};

  return std::any_of(stdintTypes.begin(), stdintTypes.end(),
                     [&](std::string_view s) {
                       return type_name.find(s) != std::string::npos;
                     });
}
} // namespace detail

/**
 * @brief Remove qualifiers from a type name recursively
 *
 * Removes qualifiers as well as references recursively from a type name.
 * "Recursively" refers to qualifiers that exists in pointer-pointee nested
 * structure of the clangAST. For instance, `const int*` is a pointer type to a
 * const int. The top level type is a pointer type, but the const qualifier
 * exists in the pointee type, which is one level lower. This function removes
 * the const qualifier from the pointee type, so that `const int*` becomes
 * `int*`.
 *
 * @param ty The type to remove qualifiers from
 * @param pp The printing policy to use
 * @return std::string The type name with qualifiers removed
 */
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
  // If the type is a stdint type (uint64_t, e.g.), we maintain the name of the
  // type itself for portability. Downstream consumer of this type information
  // should remember to include <stdint.h> or <cstdint> in their code.

  std::string printed_name = qualtype.getAsString();
  clang::QualType ty = detail::contains_stdint_type(printed_name)
                           ? qualtype
                           : qualtype.getCanonicalType();

  clang::PrintingPolicy pp{context.getLangOpts()};

  name = ty.getAsString(pp);
  unqualified_non_ref_type_name = remove_qualifier_recursive_to_name(ty, pp);

  _is_right_reference = ty->isRValueReferenceType();
  _is_left_reference = ty->isLValueReferenceType();
}

bool Type::is_right_reference() const { return _is_right_reference; }
bool Type::is_left_reference() const { return _is_left_reference; }

} // namespace ast_canopy
