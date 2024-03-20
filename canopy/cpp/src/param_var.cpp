// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#include "canopy.hpp"

namespace canopy {
ParamVar::ParamVar(const clang::ParmVarDecl *PVD)
    : name(PVD->getNameAsString()), type(PVD->getType(), PVD->getASTContext()) {
}

} // namespace canopy
