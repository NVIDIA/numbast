// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#include <clang/AST/Attr.h>
#include <clang/AST/DeclCXX.h>
#include <clang/AST/Mangle.h>

#include <ast_canopy/ast_canopy.hpp>

#include <algorithm>
#include <iostream>

namespace ast_canopy {

execution_space get_execution_space(const clang::FunctionDecl *FD) {
  using namespace clang;
  if (FD->hasAttr<CUDAGlobalAttr>()) {
    return execution_space::global_;
  } else if (FD->hasAttr<CUDAHostAttr>() && FD->hasAttr<CUDADeviceAttr>()) {
    return execution_space::host_device;
  } else if (FD->hasAttr<CUDAHostAttr>()) {
    return execution_space::host;
  } else if (FD->hasAttr<CUDADeviceAttr>()) {
    return execution_space::device;
  } else {
    return execution_space::undefined;
  }
}

Function::Function(const clang::FunctionDecl *FD)
    : name(FD->getNameAsString()), qual_name(FD->getQualifiedNameAsString()),
      return_type(FD->getReturnType(), FD->getASTContext()),
      is_constexpr(FD->isConstexpr()) {
  params.reserve(FD->getNumParams());
  std::transform(FD->param_begin(), FD->param_end(), std::back_inserter(params),
                 [](const clang::ParmVarDecl *PVD) { return ParamVar(PVD); });
  exec_space = get_execution_space(FD);

  // Get the attributes.
  for (const clang::Attr *attr : FD->attrs()) {
    this->attributes.insert(attr->getSpelling());
  }

  // Compute itanium mangled name
  auto &context = FD->getASTContext();
  auto &diag = context.getDiagnostics();
  clang::ItaniumMangleContext *MC =
      clang::ItaniumMangleContext::create(context, diag);
  llvm::raw_string_ostream OS(mangled_name);

  clang::GlobalDecl GD;
  if (const clang::CXXConstructorDecl *CCD =
          clang::dyn_cast<clang::CXXConstructorDecl>(FD)) {
    GD = clang::GlobalDecl(CCD, clang::Ctor_Complete);
  } else if (const clang::CXXDestructorDecl *CDD =
                 clang::dyn_cast<clang::CXXDestructorDecl>(FD)) {
    GD = clang::GlobalDecl(CDD, clang::Dtor_Complete);
  } else {
    GD = clang::GlobalDecl(FD);
  }

  MC->mangleName(GD, OS);
  OS.flush();
}
} // namespace ast_canopy
