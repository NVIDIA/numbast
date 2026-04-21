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

Typedef::Typedef(const clang::TypedefDecl *TD,
                 std::unordered_map<int64_t, std::string> *record_id_to_name) {
  name = TD->getNameAsString();
  qual_name = TD->getQualifiedNameAsString();
  if (qual_name.empty()) {
    qual_name = name;
  }
  clang::QualType qd = TD->getUnderlyingType();
  clang::RecordDecl *underlying_record_decl = qd->getAsCXXRecordDecl();

  if (underlying_record_decl) {
    auto it = record_id_to_name->find(underlying_record_decl->getID());
    if (it != record_id_to_name->end()) {
      underlying_name = it->second;
    } else {
      // The underlying record was not captured by the record matcher (e.g. it
      // is a class template instantiation, comes from a non-retained file, or
      // lives inside a partial specialization).  Fall back to the name Clang
      // gives us directly.
      underlying_name = underlying_record_decl->getNameAsString();
      if (underlying_name.empty()) {
        underlying_name = underlying_record_decl->getQualifiedNameAsString();
      }
    }
  } else {
    // The underlying type is not a CXXRecordDecl (e.g. a built-in or
    // dependent type).  Use the printed type name as a fallback.
    underlying_name = qd.getAsString();
  }

#ifndef NDEBUG

  if (underlying_record_decl) {
    std::cout << name << std::endl;
    std::cout << underlying_record_decl->getNameAsString() << std::endl;
    std::cout << underlying_record_decl->getID() << std::endl;
  }
  std::cout << "TYPEDEF: "
            << "name: " << name << " underlying_name: " << underlying_name
            << std::endl;

#endif
}

} // namespace ast_canopy
