// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#include <ast_canopy/ast_canopy.hpp>

#include <clang/AST/DeclTemplate.h>

#include <algorithm>

#ifndef NDEBUG
#include <iostream>
#endif

namespace ast_canopy {

FunctionTemplate::FunctionTemplate(
    const std::vector<TemplateParam> &template_parameters,
    const std::size_t &num_min_required_args, const Function &function)
    : Template(std::move(template_parameters), num_min_required_args),
      function(std::move(function)) {}

FunctionTemplate::FunctionTemplate(const clang::FunctionTemplateDecl *FTD)
    : Template(FTD->getTemplateParameters()),
      function(FTD->getTemplatedDecl()) {}
} // namespace ast_canopy
