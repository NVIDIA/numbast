// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#include "ast_canopy.hpp"

#include <clang/AST/DeclTemplate.h>

#include <algorithm>

namespace ast_canopy {
TemplateParam::TemplateParam(const clang::TemplateTypeParmDecl *TPD) {
  name = TPD->getNameAsString();
  type = Type(TPD->getASTContext().getTypeDeclType(TPD), TPD->getASTContext());
  kind = template_param_kind::type;
}

TemplateParam::TemplateParam(const clang::NonTypeTemplateParmDecl *TPD) {
  name = TPD->getNameAsString();
  type = Type(TPD->getType(), TPD->getASTContext());
  kind = template_param_kind::non_type;
}

TemplateParam::TemplateParam(const clang::TemplateTemplateParmDecl *TPD) {
  throw std::runtime_error("TemplateTemplateParmDecl not implemented");
}

} // namespace ast_canopy
