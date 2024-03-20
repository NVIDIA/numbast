// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#include "canopy.hpp"

namespace canopy {

Field::Field(const clang::FieldDecl *FD)
    : name(FD->getNameAsString()), type(FD->getType(), FD->getASTContext()) {}
} // namespace canopy
