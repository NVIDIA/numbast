// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <clang/ASTMatchers/ASTMatchers.h>
#include <clang/Frontend/ASTUnit.h>
#include <clang/Frontend/CompilerInstance.h>

#include <filesystem>
#include <utility>

#ifndef NDEBUG
#include <iostream>
#endif

#include "canopy.hpp"
#include "detail/matchers.hpp"

using namespace clang;
using namespace clang::ast_matchers;

namespace canopy {

namespace detail {

/**
 * @brief Return the source filename of the declaration.
 */
std::string source_filename_from_decl(const Decl *D) {
  const ASTContext &ast_context = D->getASTContext();
  const SourceManager &sm = ast_context.getSourceManager();
  const StringRef file_name_ref = sm.getFilename(D->getLocation());
  std::string file_name = file_name_ref.str();
  return file_name;
}

} // namespace detail

Declarations
parse_declarations_from_ast(std::string_view ast_file_path,
                            std::vector<std::string> files_to_retain) {
  if (!std::filesystem::exists(ast_file_path)) {
    throw std::invalid_argument("AST file does not exist.");
  }

  clang::CompilerInstance CI;
  CI.createDiagnostics();

  std::unique_ptr<ASTUnit> ast = ASTUnit::LoadFromASTFile(
      ast_file_path.data(), CI.getPCHContainerReader(), ASTUnit::LoadEverything,
      &CI.getDiagnostics(), CI.getFileSystemOpts(), nullptr);

  Declarations decls;
  std::unordered_map<int64_t, std::string> record_id_to_name;
  detail::traverse_ast_payload payload{&decls, &record_id_to_name,
                                       &files_to_retain};

  detail::FunctionCallback func_callback(&payload);
  detail::RecordCallback record_callback(&payload);
  detail::TypedefMatcher typedef_callback(&payload);
  detail::FunctionTemplateCallback func_template_callback(&payload);
  detail::ClassTemplateCallback class_template_callback(&payload);
  detail::EnumCallback enum_callback(&payload);

  MatchFinder finder;

  // Match all free standing, non template functions.
  finder.addMatcher(
      functionDecl(allOf(unless(hasAncestor(recordDecl())),
                         unless(hasAncestor(functionTemplateDecl()))))
          .bind("function"),
      &func_callback);

  // Match all non-template struct/class/union declarations.
  // Use `hasAncestor` because record declaration can be nested. A recordDecl
  // can still be a templated declaration if its declared inside a templated
  // record even if itself doesn't have a classTemplateDecl parent.
  // Nested Records are also ignored.

  finder.addMatcher(recordDecl(allOf(unless(hasAncestor(classTemplateDecl())),
                                     unless(hasParent(recordDecl()))))
                        .bind("record"),
                    &record_callback);
  finder.addMatcher(typedefDecl().bind("typedef"), &typedef_callback);

  // Match free function templates.
  finder.addMatcher(functionTemplateDecl(unless(hasAncestor(cxxRecordDecl())))
                        .bind("function_template"),
                    &func_template_callback);

  finder.addMatcher(classTemplateDecl().bind("class_template"),
                    &class_template_callback);

  finder.addMatcher(enumDecl().bind("enum"), &enum_callback);

  finder.matchAST(ast->getASTContext());

#ifndef NDEBUG
  std::cout << "Records: " << decls.records.size() << std::endl;
  std::cout << "Functions: " << decls.functions.size() << std::endl;
  std::cout << "Function templates: " << decls.function_templates.size()
            << std::endl;
  std::cout << "Class templates: " << decls.class_templates.size() << std::endl;
  std::cout << "Typedefs: " << decls.typedefs.size() << std::endl;
  std::cout << "Enums: " << decls.enums.size() << std::endl;
#endif
  std::cout << "Finished parsing declarations from AST file." << std::endl;

  return decls;
}

Declarations
parse_declarations_from_command_line(std::vector<std::string> options,
                                     std::vector<std::string> files_to_retain) {

  std::vector<const char *> option_ptrs;
  for (auto &opt : options) {
    option_ptrs.push_back(opt.c_str());
  }
  const char **argstart = &(*option_ptrs.begin());
  const char **argend = &(*option_ptrs.end());

  auto PCHContainerOps = std::make_shared<PCHContainerOperations>();
  auto Diags = CompilerInstance::createDiagnostics(new DiagnosticOptions());

  std::unique_ptr<ASTUnit> ast = ASTUnit::LoadFromCommandLine(
      argstart, argend, PCHContainerOps, Diags, "");

  Declarations decls;
  std::unordered_map<int64_t, std::string> record_id_to_name;
  detail::traverse_ast_payload payload{&decls, &record_id_to_name,
                                       &files_to_retain};

  detail::FunctionCallback func_callback(&payload);
  detail::RecordCallback record_callback(&payload);
  detail::TypedefMatcher typedef_callback(&payload);
  detail::FunctionTemplateCallback func_template_callback(&payload);
  detail::ClassTemplateCallback class_template_callback(&payload);
  detail::EnumCallback enum_callback(&payload);

  MatchFinder finder;

  // Match all free standing, non template functions.
  finder.addMatcher(
      functionDecl(allOf(unless(hasAncestor(recordDecl())),
                         unless(hasAncestor(functionTemplateDecl()))))
          .bind("function"),
      &func_callback);

  // Match all non-template struct/class/union declarations.
  // Use `hasAncestor` because record declaration can be nested. A recordDecl
  // can still be a templated declaration if its declared inside a templated
  // record even if itself doesn't have a classTemplateDecl parent.
  // Nested Records are also ignored.

  finder.addMatcher(recordDecl(allOf(unless(hasAncestor(classTemplateDecl())),
                                     unless(hasParent(recordDecl()))))
                        .bind("record"),
                    &record_callback);
  finder.addMatcher(typedefDecl().bind("typedef"), &typedef_callback);

  // Match free function templates.
  finder.addMatcher(functionTemplateDecl(unless(hasAncestor(cxxRecordDecl())))
                        .bind("function_template"),
                    &func_template_callback);

  finder.addMatcher(classTemplateDecl().bind("class_template"),
                    &class_template_callback);

  finder.addMatcher(enumDecl().bind("enum"), &enum_callback);

  finder.matchAST(ast->getASTContext());

#ifndef NDEBUG
  std::cout << "Records: " << decls.records.size() << std::endl;
  std::cout << "Functions: " << decls.functions.size() << std::endl;
  std::cout << "Function templates: " << decls.function_templates.size()
            << std::endl;
  std::cout << "Class templates: " << decls.class_templates.size() << std::endl;
  std::cout << "Typedefs: " << decls.typedefs.size() << std::endl;
  std::cout << "Enums: " << decls.enums.size() << std::endl;
#endif
  std::cout << "Finished parsing declarations from AST file." << std::endl;

  return decls;
}

} // namespace canopy
