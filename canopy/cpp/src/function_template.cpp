// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#include "canopy.hpp"

#include <clang/AST/DeclTemplate.h>

#include <algorithm>

#ifndef NDEBUG
#include <iostream>
#endif

namespace canopy {

FunctionTemplate::FunctionTemplate(const clang::FunctionTemplateDecl *FTD)
    : Template(FTD->getTemplateParameters()),
      function(FTD->getTemplatedDecl()) {}
} // namespace canopy
