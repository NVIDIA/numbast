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

Type::Type(std::string name, std::string unqualified_non_ref_type_name,
           bool is_right_reference, bool is_left_reference, type_kind kind,
           bool is_const_qualified, bool is_volatile_qualified,
           bool is_restrict_qualified, std::vector<Type> inner_types,
           std::optional<std::uint64_t> array_size, std::string type_name)
    : name(std::move(name)),
      unqualified_non_ref_type_name(std::move(unqualified_non_ref_type_name)),
      kind(kind), array_size(array_size), type_name(std::move(type_name)),
      _inner_types(std::move(inner_types)),
      _is_right_reference(is_right_reference),
      _is_left_reference(is_left_reference),
      _is_const_qualified(is_const_qualified),
      _is_volatile_qualified(is_volatile_qualified),
      _is_restrict_qualified(is_restrict_qualified) {}

Type::Type(const clang::QualType &qualtype, const clang::ASTContext &context) {
  // If the type is a stdint type (uint64_t, e.g.), we maintain the name of the
  // type itself for portability. Downstream consumer of this type information
  // should remember to include <stdint.h> or <cstdint> in their code.

  // Guard: if the QualType is null, produce a placeholder.
  if (qualtype.isNull()) {
    name = "<null-type>";
    unqualified_non_ref_type_name = "<null-type>";
    _is_right_reference = false;
    _is_left_reference = false;
    return;
  }

  std::string printed_name = qualtype.getAsString();

  clang::QualType ty = detail::contains_stdint_type(printed_name)
                           ? qualtype
                           : qualtype.getCanonicalType();

  clang::PrintingPolicy pp{context.getLangOpts()};

  name = ty.getAsString(pp);
  unqualified_non_ref_type_name = remove_qualifier_recursive_to_name(ty, pp);

  _is_right_reference = ty->isRValueReferenceType();
  _is_left_reference = ty->isLValueReferenceType();
  _is_const_qualified = qualtype.isConstQualified();
  _is_volatile_qualified = qualtype.isVolatileQualified();
  _is_restrict_qualified = qualtype.isRestrictQualified();

  const clang::Type *type = qualtype.getTypePtrOrNull();
  if (type == nullptr)
    return;

  auto add_inner_type = [&](clang::QualType inner) {
    _inner_types.emplace_back(inner, context);
  };

  if (const auto *builtin = llvm::dyn_cast<clang::BuiltinType>(type)) {
    kind = type_kind::builtin;
    type_name = builtin->getName(pp).str();
  } else if (const auto *pointer = llvm::dyn_cast<clang::PointerType>(type)) {
    kind = type_kind::pointer;
    add_inner_type(pointer->getPointeeType());
  } else if (const auto *reference =
                 llvm::dyn_cast<clang::LValueReferenceType>(type)) {
    kind = type_kind::lvalue_reference;
    add_inner_type(reference->getPointeeType());
  } else if (const auto *reference =
                 llvm::dyn_cast<clang::RValueReferenceType>(type)) {
    kind = type_kind::rvalue_reference;
    add_inner_type(reference->getPointeeType());
  } else if (const auto *array =
                 llvm::dyn_cast<clang::ConstantArrayType>(type)) {
    kind = type_kind::constant_array;
    array_size = array->getSize().getLimitedValue();
    add_inner_type(array->getElementType());
  } else if (const auto *array =
                 llvm::dyn_cast<clang::IncompleteArrayType>(type)) {
    kind = type_kind::incomplete_array;
    add_inner_type(array->getElementType());
  } else if (const auto *array =
                 llvm::dyn_cast<clang::VariableArrayType>(type)) {
    kind = type_kind::variable_array;
    add_inner_type(array->getElementType());
  } else if (const auto *array =
                 llvm::dyn_cast<clang::DependentSizedArrayType>(type)) {
    kind = type_kind::dependent_sized_array;
    add_inner_type(array->getElementType());
  } else if (const auto *record = llvm::dyn_cast<clang::RecordType>(type)) {
    kind = type_kind::record;
    type_name = record->getDecl()->getNameAsString();
  } else if (const auto *enum_type = llvm::dyn_cast<clang::EnumType>(type)) {
    kind = type_kind::enum_;
    type_name = enum_type->getDecl()->getNameAsString();
  } else if (const auto *typedef_type =
                 llvm::dyn_cast<clang::TypedefType>(type)) {
    kind = type_kind::typedef_;
    type_name = typedef_type->getDecl()->getNameAsString();
    add_inner_type(typedef_type->desugar());
  } else if (llvm::isa<clang::FunctionType>(type)) {
    kind = type_kind::function;
  } else if (const auto *member_pointer =
                 llvm::dyn_cast<clang::MemberPointerType>(type)) {
    kind = type_kind::member_pointer;
    add_inner_type(member_pointer->getPointeeType());
  } else if (const auto *adjusted = llvm::dyn_cast<clang::AdjustedType>(type)) {
    kind = type_kind::adjusted;
    add_inner_type(adjusted->getAdjustedType());
  } else if (const clang::QualType desugared =
                 qualtype.getSingleStepDesugaredType(context);
             desugared != qualtype) {
    // Clang's individual transparent sugar classes are not a stable API. For
    // example, ElaboratedType was removed in Clang 22. Preserve the structural
    // layer through QualType's version-stable, one-step desugaring API instead.
    kind = type_kind::sugar;
    add_inner_type(desugared);
  } else {
    kind = type_kind::other;
  }
}

} // namespace ast_canopy
