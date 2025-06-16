// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#include <clang/AST/Attr.h>
#include <clang/AST/DeclCXX.h>

#include <ast_canopy/ast_canopy.hpp>

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
    : Declaration(FD), name(FD->getNameAsString()),
      return_type(FD->getReturnType(), FD->getASTContext()),
      exec_space(get_execution_space(FD)), is_constexpr(FD->isConstexpr()) {
  std::transform(FD->param_begin(), FD->param_end(), std::back_inserter(params),
                 [](const clang::ParmVarDecl *PVD) { return ParamVar(PVD); });
}

Function::Function(const std::string &name, const Type &return_type,
                   const std::vector<ParamVar> &params,
                   const execution_space &exec_space,
                   const std::vector<std::string> &namespace_stack)
    : Declaration(namespace_stack), name(name), return_type(return_type),
      params(params), exec_space(exec_space), is_constexpr(false) {}
} // namespace ast_canopy
