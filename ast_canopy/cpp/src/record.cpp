// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#include <ast_canopy/ast_canopy.hpp>

#include <clang/AST/ASTContext.h>
#include <clang/AST/DeclCXX.h>

#include <algorithm>

#ifndef NDEBUG
#include <iostream>
#endif

namespace ast_canopy {

std::size_t constexpr INVALID_SIZE_OF = std::numeric_limits<std::size_t>::max();
std::size_t constexpr INVALID_ALIGN_OF =
    std::numeric_limits<std::size_t>::max();

Record::Record(const clang::CXXRecordDecl *RD, RecordAncestor rp) : Decl(RD) {
  using AS = clang::AccessSpecifier;

#ifndef NDEBUG
  source_range = RD->getSourceRange().printToString(
      RD->getASTContext().getSourceManager());
#endif

  name = RD->getNameAsString();

  // Class default access specifier is private, struct is public.
  AS access = RD->isClass() ? AS::AS_private : AS::AS_public;

  fields.reserve(std::distance(RD->field_begin(), RD->field_end()));
  methods.reserve(std::distance(RD->method_begin(), RD->method_end()));
  auto DC = static_cast<clang::DeclContext>(*RD);

  for (auto const *D : DC.decls()) {
    // Scan all declarations, if the declaration is a access specifier,
    // update access for all following declarations.
    if (auto const *ASD = clang::dyn_cast<clang::AccessSpecDecl>(D)) {
      access = ASD->getAccess();
      continue;
    }

    // Include all fields regardless of access specifier, downstream
    // tools needs all fields to create type with proper size and alignment.
    if (auto const *FD = clang::dyn_cast<clang::FieldDecl>(D)) {
      fields.emplace_back(Field(FD, access));
    }

    // Skip Non-public function, nested class and template declarations
    if (access == AS::AS_public) {

      if (auto const *MD = clang::dyn_cast<clang::CXXMethodDecl>(D)) {
        if (MD->isImplicit())
          continue;
        methods.emplace_back(Method(MD));
      }

      if (auto const *FTD = clang::dyn_cast<clang::FunctionTemplateDecl>(D)) {
        templated_methods.emplace_back(FunctionTemplate(FTD));
      }

      if (auto const *CTD = clang::dyn_cast<clang::ClassTemplateDecl>(D)) {
        nested_class_templates.emplace_back(ClassTemplate(CTD));
      }

      if (auto const *R = clang::dyn_cast<clang::CXXRecordDecl>(D)) {
        nested_records.emplace_back(Record(R, rp));
      }
    }
  }

  if (rp == RecordAncestor::ANCESTOR_IS_NOT_TEMPLATE) {
    clang::QualType type = RD->getASTContext().getTypeDeclType(RD);
    clang::ASTContext &ctx = RD->getASTContext();
    sizeof_ = ctx.getTypeSize(type) / ctx.getCharWidth();
    alignof_ = ctx.getTypeAlign(type) / ctx.getCharWidth();
  } else {
    // A record with class template parent is not instantiated, and thus
    // computing size and alignment of such a record is not possible.
    sizeof_ = INVALID_SIZE_OF;
    alignof_ = INVALID_ALIGN_OF;
  }

#ifndef NDEBUG
  print(0);
#endif
}

Record::Record(const clang::CXXRecordDecl *RD, RecordAncestor rp,
               std::string name)
    : Record(RD, rp) {
  this->name = name;
}

void Record::print(int level) const {
#ifndef NDEBUG
  std::string indentation = std::string(level * 2, ' ');
  std::cout << indentation << "Record: " << name << std::endl;
  std::cout << indentation << source_range << std::endl;
  std::cout << indentation << "  fields: " << fields.size() << std::endl;
  std::cout << indentation << "  methods: " << methods.size() << std::endl;
  for (const auto &method : methods) {
    std::cout << indentation << "    " << method.name << std::endl;
  }
  std::cout << indentation
            << "  templated_methods: " << templated_methods.size() << std::endl;
  for (const auto &templated_method : templated_methods) {
    std::cout << indentation << "    " << templated_method.function.name
              << std::endl;
  }
  std::cout << indentation << "  nested_records: " << nested_records.size()
            << std::endl;
  for (const auto &nested_record : nested_records) {
    nested_record.print(level + 1);
  }
  std::cout << indentation
            << "  nested_class_templates: " << nested_class_templates.size()
            << std::endl;
  for (const auto &nested_class_template : nested_class_templates) {
    nested_class_template.record.print(level + 1);
  }
  std::cout << indentation << "  sizeof: " << sizeof_ << std::endl;
  std::cout << indentation << "  alignof: " << alignof_ << std::endl;
#endif
}

} // namespace ast_canopy
