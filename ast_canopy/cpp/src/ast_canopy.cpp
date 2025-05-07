// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <clang/ASTMatchers/ASTMatchers.h>
#include <clang/Basic/Version.h>
#include <clang/Frontend/ASTUnit.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>

#include <filesystem>
#include <utility>

#ifndef NDEBUG
#include <iostream>
#endif

#include "detail/matchers.hpp"
#include <ast_canopy/ast_canopy.hpp>

using namespace clang;
using namespace clang::ast_matchers;

namespace ast_canopy {

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

std::unique_ptr<ASTUnit>
default_ast_unit_from_command_line(const std::vector<std::string> &options) {

  std::vector<const char *> option_ptrs;
  for (auto &opt : options) {
    option_ptrs.push_back(opt.c_str());
  }
  const char **argstart = &(*option_ptrs.begin());
  const char **argend = &(*option_ptrs.end());

  auto PCHContainerOps = std::make_shared<PCHContainerOperations>();

  // Create a diagnostics engine that prints to llvm::errs(). llvm::errs()
  // is a buffered err stream. By default, llvm::errs() writes directly to
  // STDERR_FILENO, and is unbuffered. To capture the output, we need to
  // manually redirect the stderr file descriptor (2).
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
  DiagnosticConsumer *DiagConsumer =
      new TextDiagnosticPrinter(llvm::errs(), &*DiagOpts);

#if CLANG_VERSION_MAJOR >= 20
  IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS = llvm::vfs::getRealFileSystem();
  auto Diags = CompilerInstance::createDiagnostics(*FS, &*DiagOpts,
                                                   &*DiagConsumer, false);
#else
  auto Diags =
      CompilerInstance::createDiagnostics(&*DiagOpts, &*DiagConsumer, false);
#endif

  std::unique_ptr<ASTUnit> ast(ASTUnit::LoadFromCommandLine(
      argstart, argend, PCHContainerOps, Diags, ""));

  return ast;
}

} // namespace detail

Declarations parse_declarations_from_command_line(
    std::vector<std::string> options, std::vector<std::string> files_to_retain,
    std::vector<std::string> whitelist_prefixes) {

  auto ast = detail::default_ast_unit_from_command_line(options);

  Declarations decls;
  std::unordered_map<int64_t, std::string> record_id_to_name;
  detail::traverse_ast_payload payload{&decls, &record_id_to_name,
                                       &files_to_retain, &whitelist_prefixes};

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
  std::cout << "Finished parsing declarations from AST file." << std::endl;
#endif

  return decls;
}

std::optional<ConstExprVar>
value_from_constexpr_vardecl(std::vector<std::string> options,
                             std::string vardecl_name) {

  auto ast = detail::default_ast_unit_from_command_line(options);

  detail::vardecl_matcher_payload payload{vardecl_name, std::nullopt};

  MatchFinder finder;

  detail::ConstexprVarDeclCallback constexpr_vardecl_callback(&payload);

  // Match all VarDecl declared with constexpr.
  finder.addMatcher(varDecl(isConstexpr()).bind("constexpr_vardecl"),
                    &constexpr_vardecl_callback);

  finder.matchAST(ast->getASTContext());

  return std::move(payload.var);
}

} // namespace ast_canopy
