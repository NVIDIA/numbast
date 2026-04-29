// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#include "matchers.hpp"

#include <clang/AST/ASTContext.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <llvm/ADT/APSInt.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/Support/raw_ostream.h>

#ifndef NDEBUG
#include <iostream>
#endif

namespace ast_canopy {

namespace detail {

using namespace clang;

static std::string
template_argument_to_string(const clang::TemplateArgument &arg,
                            const clang::PrintingPolicy &policy) {
  switch (arg.getKind()) {
  case clang::TemplateArgument::ArgKind::Type:
    return arg.getAsType().getAsString(policy);
  case clang::TemplateArgument::ArgKind::Integral: {
    clang::QualType type = arg.getIntegralType();
    if (type->isEnumeralType()) {
      const auto *enum_type = type->getAs<clang::EnumType>();
      const clang::EnumDecl *enum_decl = enum_type->getDecl();
      const llvm::APSInt &value = arg.getAsIntegral();

      for (const auto *enum_constant : enum_decl->enumerators()) {
        if (enum_constant->getInitVal() == value) {
          return enum_constant->getQualifiedNameAsString();
        }
      }
    }

    llvm::SmallString<32> str;
    arg.getAsIntegral().toString(str);
    return std::string(str.c_str());
  }
  case clang::TemplateArgument::ArgKind::Pack: {
    std::string result;
    bool first = true;
    for (const auto &packed_arg : arg.pack_elements()) {
      if (!first)
        result += ", ";
      first = false;
      result += template_argument_to_string(packed_arg, policy);
    }
    return result;
  }
  default: {
    std::string result;
    llvm::raw_string_ostream os(result);
    arg.print(policy, os, false);
    os.flush();
    return result;
  }
  }
}

static std::string class_template_specialization_name(
    const clang::ClassTemplateSpecializationDecl *CTSD) {
  auto name = CTSD->getNameAsString();
  if (name.empty()) {
    name = CTSD->getQualifiedNameAsString();
  }
  if (name.empty()) {
    name = "unnamed" + std::to_string(CTSD->getID());
  }

  const auto &template_args = CTSD->getTemplateArgs();
  if (template_args.size() == 0) {
    return name;
  }

  const clang::PrintingPolicy &policy =
      CTSD->getASTContext().getPrintingPolicy();
  std::string specialization_name = name + "<";
  for (unsigned i = 0; i < template_args.size(); ++i) {
    if (i > 0)
      specialization_name += ", ";
    specialization_name +=
        template_argument_to_string(template_args[i], policy);
  }
  specialization_name += ">";
  return specialization_name;
}

ClassTemplateSpecializationCallback::ClassTemplateSpecializationCallback(
    traverse_ast_payload *payload)
    : payload(payload) {}

void ClassTemplateSpecializationCallback::run(
    const MatchFinder::MatchResult &Result) {

  const ClassTemplateSpecializationDecl *CTSD =
      Result.Nodes.getNodeAs<clang::ClassTemplateSpecializationDecl>("ctsd");

  if (llvm::isa<clang::ClassTemplatePartialSpecializationDecl>(CTSD)) {
    // Notice that CTPSD is-a CTSD, we cannot materialize it, otherwise infinite
    // recursion will happen with getTypeInfo().
    return;
  }

  std::string file_name = source_filename_from_decl(CTSD);

  if (std::any_of(payload->files_to_retain->begin(),
                  payload->files_to_retain->end(),
                  [&file_name](const std::string &file_to_retain) {
                    return file_name == file_to_retain;
                  }))

  {
    auto id = CTSD->getID();
    (*payload->record_id_to_name)[id] =
        class_template_specialization_name(CTSD);

    if (!CTSD->isImplicit() && CTSD->isCompleteDefinition())
      payload->decls->class_template_specializations.push_back(
          ClassTemplateSpecialization(CTSD));
  }
}

} // namespace detail

} // namespace ast_canopy
