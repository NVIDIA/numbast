// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#include "ast_canopy.hpp"

#include <clang/AST/DeclTemplate.h>

#include <algorithm>

#ifndef NDEBUG
#include <iostream>
#endif

namespace ast_canopy {

Template::Template(const clang::TemplateParameterList *TPL)
    : num_min_required_args(TPL->getMinRequiredArguments()) {

  std::transform(
      TPL->begin(), TPL->end(), std::back_inserter(template_parameters),
      [](const clang::NamedDecl *ND) {
        if (const clang::TemplateTypeParmDecl *TPD =
                clang::dyn_cast<clang::TemplateTypeParmDecl>(ND)) {
          return TemplateParam(TPD);
        } else if (const clang::NonTypeTemplateParmDecl *TPD =
                       clang::dyn_cast<clang::NonTypeTemplateParmDecl>(ND)) {
          return TemplateParam(TPD);
        } else if (const clang::TemplateDecl *TD =
                       clang::dyn_cast<clang::TemplateDecl>(ND)) {
          if (const clang::TemplateTemplateParmDecl *TPD =
                  clang::dyn_cast<clang::TemplateTemplateParmDecl>(TD)) {
            return TemplateParam(TPD);
          }
        }

        // Shouldn't fall through
        throw std::runtime_error(ND->getNameAsString() +
                                 " is unknown template parameter type");
      });
}
} // namespace ast_canopy
