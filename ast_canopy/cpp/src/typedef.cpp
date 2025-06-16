// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#include <ast_canopy/ast_canopy.hpp>

#ifndef NDEBUG
#include <iostream>
#endif

#include <unordered_map>

namespace ast_canopy {

Typedef::Typedef(const std::string &name, const std::string &underlying_name,
                 const std::vector<std::string> &namespace_stack)
    : Declaration(namespace_stack), name(name),
      underlying_name(underlying_name) {}

Typedef::Typedef(const clang::TypedefDecl *TD,
                 std::unordered_map<int64_t, std::string> *record_id_to_name)
    : Declaration(TD), name(TD->getNameAsString()) {
  const clang::Type *type = TD->getUnderlyingType().getTypePtr();
  if (const clang::RecordType *RT = type->getAs<clang::RecordType>()) {
    if (const clang::CXXRecordDecl *RD = RT->getAsCXXRecordDecl()) {
      int64_t id = RD->getID();
      underlying_name = record_id_to_name->contains(id)
                            ? (*record_id_to_name)[id]
                            : RD->getNameAsString();
    }
  } else {
    underlying_name = TD->getUnderlyingType().getAsString();
  }

#ifndef NDEBUG

  std::cout << "TYPEDEF: "
            << "name: " << name << " underlying_name: " << underlying_name
            << std::endl;

#endif
}

} // namespace ast_canopy
