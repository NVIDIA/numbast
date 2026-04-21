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

Template::Template(const clang::TemplateParameterList *TPL)
    : num_min_required_args(TPL->getMinRequiredArguments()) {

  for (const clang::NamedDecl *ND : *TPL) {
    if (const clang::TemplateTypeParmDecl *TPD =
            clang::dyn_cast<clang::TemplateTypeParmDecl>(ND)) {
      template_parameters.push_back(TemplateParam(TPD));
    } else if (const clang::NonTypeTemplateParmDecl *TPD =
                   clang::dyn_cast<clang::NonTypeTemplateParmDecl>(ND)) {
      template_parameters.push_back(TemplateParam(TPD));
    } else if (const clang::TemplateTemplateParmDecl *TPD =
                   clang::dyn_cast<clang::TemplateTemplateParmDecl>(ND)) {
      // Template template parameters (e.g. template<template<class> class X>).
      // Use the TemplateTemplateParmDecl overload which now handles this
      // gracefully instead of throwing.
      template_parameters.push_back(TemplateParam(TPD));
    } else {
      // Unknown template parameter kind -- skip it rather than crashing.
      // This can happen with future Clang additions or unusual AST nodes.
    }
  }
}
} // namespace ast_canopy
