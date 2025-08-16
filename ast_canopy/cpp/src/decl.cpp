// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#include <clang/AST/Decl.h>
#include <clang/Basic/SourceManager.h>

#include <ast_canopy/ast_canopy.hpp>
#include <ast_canopy/source_location.hpp>

#include <iostream>

namespace ast_canopy {

Decl::Decl() : source_location(SourceLocation()) {}

Decl::Decl(const clang::Decl *decl) {
  source_location = detail::location_from_decl(decl);
}

} // namespace ast_canopy
