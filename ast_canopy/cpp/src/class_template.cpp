// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#include <ast_canopy/ast_canopy.hpp>

#include <clang/AST/DeclTemplate.h>

#include <algorithm>

#include <iostream>

namespace ast_canopy {

ClassTemplate::ClassTemplate(
    const std::vector<TemplateParam> &template_parameters,
    const std::size_t &num_min_required_args, const Record &record,
    const std::vector<std::string> &namespace_stack)
    : Template(template_parameters, num_min_required_args),
      Declaration(namespace_stack), record(record) {}

ClassTemplate::ClassTemplate(const clang::ClassTemplateDecl *CTD)
    : Template(CTD->getTemplateParameters()), Declaration(CTD),
      record(CTD->getTemplatedDecl(), RecordAncestor::ANCESTOR_IS_TEMPLATE) {}

} // namespace ast_canopy
