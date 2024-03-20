// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#include "canopy.hpp"

namespace canopy {

Enum::Enum(const clang::EnumDecl *ED) : name(ED->getNameAsString()) {
  for (const auto *enumerator : ED->enumerators()) {
    enumerators.push_back(enumerator->getNameAsString());

    auto const val = enumerator->getInitVal();
    llvm::SmallVector<char> buf;
    val.toString(buf);
    std::string s(buf.begin(), buf.end());

    enumerator_values.push_back(s);
  }
}

} // namespace canopy
