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
#include <string>

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

static bool is_identifier_alnum(unsigned char c) {
  return ('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z') ||
         ('0' <= c && c <= '9');
}

static std::string encode_identifier_fragment(const std::string &input) {
  static constexpr char hex[] = "0123456789ABCDEF";
  std::string output;
  output.reserve(input.size());
  for (unsigned char c : input) {
    if (is_identifier_alnum(c)) {
      output.push_back(static_cast<char>(c));
    } else {
      output += "_x";
      output.push_back(hex[c >> 4]);
      output.push_back(hex[c & 0x0F]);
    }
  }
  return output;
}

/**
 * @brief Build an overload-stable fallback name for dependent signatures.
 *
 * This is not an Itanium-mangled symbol.  It includes byte-encoded printed
 * signature fragments to keep dependent overloads distinct without asking
 * Clang's mangler to process unresolved template-dependent types.
 */
static std::string
make_dependent_signature_fallback(const clang::FunctionDecl *FD,
                                  const std::string &base_name) {
  const clang::PrintingPolicy &policy = FD->getASTContext().getPrintingPolicy();

  std::string fallback = encode_identifier_fragment(base_name);
  fallback += "__dependent_signature__returns__";
  fallback +=
      encode_identifier_fragment(FD->getReturnType().getAsString(policy));
  fallback += "__params__";

  if (FD->parameters().empty()) {
    fallback += "void";
  } else {
    bool first = true;
    for (const auto *P : FD->parameters()) {
      if (!first)
        fallback += "__";
      first = false;
      fallback += encode_identifier_fragment(P->getType().getAsString(policy));
    }
  }

  if (const auto *FPT = FD->getType()->getAs<clang::FunctionProtoType>()) {
    std::string method_quals = FPT->getMethodQuals().getAsString(policy);
    if (!method_quals.empty()) {
      fallback += "__quals__";
      fallback += encode_identifier_fragment(method_quals);
    }

    switch (FPT->getRefQualifier()) {
    case clang::RQ_LValue:
      fallback += "__ref__lvalue";
      break;
    case clang::RQ_RValue:
      fallback += "__ref__rvalue";
      break;
    case clang::RQ_None:
      break;
    }
  }

  return fallback;
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
    mangled_name = make_dependent_signature_fallback(FD, qual_name);
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
