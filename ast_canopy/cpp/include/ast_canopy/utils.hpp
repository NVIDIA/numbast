// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#pragma once

#include <string>
#include <vector>

namespace clang {
class Decl;
}

namespace ast_canopy {

// Extracts namespace names from a Clang declaration in bottom-up order
// (deepest to top-level). For example, for a declaration in
// "outer::middle::inner", the vector would contain ["inner", "middle", "outer"]
std::vector<std::string> extract_namespace_stack(const clang::Decl *decl);

} // namespace ast_canopy
