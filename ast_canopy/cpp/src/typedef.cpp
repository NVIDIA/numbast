// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#include "ast_canopy.hpp"

#ifndef NDEBUG
#include <iostream>
#endif

#include <unordered_map>

namespace ast_canopy {

Typedef::Typedef(const clang::TypedefDecl *TD,
                 std::unordered_map<int64_t, std::string> *record_id_to_name) {
  name = TD->getNameAsString();
  clang::QualType qd = TD->getUnderlyingType();
  clang::RecordDecl *underlying_record_decl = qd->getAsCXXRecordDecl();

  underlying_name = record_id_to_name->at(underlying_record_decl->getID());

#ifndef NDEBUG

  std::cout << name << std::endl;
  std::cout << underlying_record_decl->getNameAsString() << std::endl;
  std::cout << underlying_record_decl->getID() << std::endl;
  std::cout << "TYPEDEF: "
            << "name: " << name << " underlying_name: " << underlying_name
            << std::endl;

#endif
}

} // namespace ast_canopy
