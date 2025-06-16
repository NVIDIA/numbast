// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#include <ast_canopy/ast_canopy.hpp>
#include <ast_canopy/utils.hpp>

namespace ast_canopy {

Declaration::Declaration(const std::vector<std::string> &ns_stack)
    : namespace_stack(ns_stack) {}

Declaration::Declaration(const clang::Decl *decl)
    : namespace_stack(extract_namespace_stack(decl)) {}

} // namespace ast_canopy
