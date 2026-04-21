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
#include <memory>

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

/**
 * @brief Return true if the function has a dependent (unresolved) signature.
 *
 * Mangling or querying canonical types on such a function is undefined and
 * can crash the Clang mangler (segfault).  We detect this upfront so callers
 * can skip or guard the operation.
 */
static bool has_dependent_signature(const clang::FunctionDecl *FD) {
  if (FD->getReturnType()->isDependentType())
    return true;
  for (const auto *P : FD->parameters()) {
    if (P->getType()->isDependentType())
      return true;
  }
  // Also flag methods whose parent class is dependent.
  if (const auto *MD = clang::dyn_cast<clang::CXXMethodDecl>(FD)) {
    if (const auto *Parent = MD->getParent()) {
      if (Parent->isDependentContext())
        return true;
    }
  }
  return false;
}

Function::Function(const clang::FunctionDecl *FD)
    : name(FD->getNameAsString()), qual_name(FD->getQualifiedNameAsString()),
      return_type(FD->getReturnType(), FD->getASTContext()),
      is_constexpr(FD->isConstexpr()) {
  if (qual_name.empty()) {
    qual_name = name;
  }
  params.reserve(FD->getNumParams());
  std::transform(FD->param_begin(), FD->param_end(), std::back_inserter(params),
                 [](const clang::ParmVarDecl *PVD) { return ParamVar(PVD); });
  exec_space = get_execution_space(FD);

  // Get the attributes.
  for (const clang::Attr *attr : FD->attrs()) {
    this->attributes.insert(attr->getSpelling());
  }

  // Compute itanium mangled name.
  // Skip mangling for functions with dependent (unresolved) signatures --
  // the Clang Itanium mangler can segfault when asked to mangle types that
  // are still template-dependent (e.g. methods of uninstantiated class
  // templates such as Eigen::Matrix<Scalar_,...>).
  if (has_dependent_signature(FD)) {
    mangled_name = qual_name; // best-effort fallback
  } else {
    auto &context = FD->getASTContext();
    auto &diag = context.getDiagnostics();
    std::unique_ptr<clang::ItaniumMangleContext> MC(
        clang::ItaniumMangleContext::create(context, diag));
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
}
} // namespace ast_canopy
