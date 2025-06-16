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

Template::Template(const std::vector<TemplateParam> &template_parameters,
                   const std::size_t &num_min_required_args)
    : template_parameters(template_parameters),
      num_min_required_args(num_min_required_args) {}

Template::Template(const clang::TemplateParameterList *TPL) {
  template_parameters.reserve(TPL->size());
  for (const auto *param : *TPL) {
    if (const auto *TTPD =
            clang::dyn_cast<clang::TemplateTypeParmDecl>(param)) {
      template_parameters.emplace_back(TemplateParam(TTPD));
    } else if (const auto *NTTPD =
                   clang::dyn_cast<clang::NonTypeTemplateParmDecl>(param)) {
      template_parameters.emplace_back(TemplateParam(NTTPD));
    } else if (const auto *TTPD =
                   clang::dyn_cast<clang::TemplateTemplateParmDecl>(param)) {
      template_parameters.emplace_back(TemplateParam(TTPD));
    }
  }
  num_min_required_args = TPL->getMinRequiredArguments();
}
} // namespace ast_canopy
