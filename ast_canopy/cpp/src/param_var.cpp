// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#include "ast_canopy.hpp"

namespace ast_canopy {
ParamVar::ParamVar(const clang::ParmVarDecl *PVD)
    : name(PVD->getNameAsString()), type(PVD->getType(), PVD->getASTContext()) {
}

} // namespace ast_canopy
