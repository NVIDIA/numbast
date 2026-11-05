// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <clang/ASTMatchers/ASTMatchers.h>
#include <clang/Basic/FileManager.h>
#include <clang/Basic/FileSystemOptions.h>
#include <clang/Basic/TargetInfo.h>
#include <clang/Basic/Version.h>
#include <clang/Frontend/ASTUnit.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/FrontendAction.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>
#if CLANG_VERSION_MAJOR >= 22
#include <clang/Driver/CreateInvocationFromArgs.h>
#else
#include <clang/Frontend/Utils.h>
#endif
#include <clang/Lex/MacroInfo.h>
#include <clang/Lex/Preprocessor.h>
#include <clang/Parse/Parser.h>
#include <llvm/ADT/SmallString.h>

#include <algorithm>
#include <filesystem>
#include <utility>

#ifndef NDEBUG
#include <iostream>
#endif

#include "detail/matchers.hpp"
#include <ast_canopy/ast_canopy.hpp>

using namespace clang;
using namespace clang::ast_matchers;

static_assert(CLANG_VERSION_MAJOR >= 18 && CLANG_VERSION_MAJOR <= 22 &&
              "ASTCanopy Only Supports Building with CLANG 18 to 22");
namespace ast_canopy {

namespace detail {

class AstCanopyDiagnosticsConsumer : public DiagnosticConsumer {
public:
  std::vector<std::string> error_messages;
  void HandleDiagnostic(clang::DiagnosticsEngine::Level DiagLevel,
                        const clang::Diagnostic &Info) override {
    llvm::SmallVector<char, 1024> buffer;
    Info.FormatDiagnostic(buffer);
    if (DiagLevel == DiagnosticsEngine::Error ||
        DiagLevel == DiagnosticsEngine::Fatal) {
      // Create verbose error message with location and level information
      std::string verbose_message;

      // Add diagnostic level
      const char *level_str =
          (DiagLevel == DiagnosticsEngine::Error) ? "Error" : "Fatal";
      verbose_message += "[" + std::string(level_str) + "] ";

      // Add source location if available
      if (Info.hasSourceManager() && Info.getLocation().isValid()) {
        const SourceManager &SM = Info.getSourceManager();
        SourceLocation loc = Info.getLocation();
        PresumedLoc PLoc = SM.getPresumedLoc(loc);
        if (PLoc.isValid()) {
          verbose_message += PLoc.getFilename();
          verbose_message += ":" + std::to_string(PLoc.getLine());
          verbose_message += ":" + std::to_string(PLoc.getColumn()) + ": ";
        }
      }

      // Add the original diagnostic message
      verbose_message += std::string(buffer.begin(), buffer.end());

      error_messages.push_back(verbose_message);
    }
  }
};

/**
 * @brief Return whether a source filename is present in the retain list.
 *
 * Macro extraction follows the same source-retention policy as declaration
 * extraction: only entities whose spelling location comes from a retained file
 * should be surfaced to Python callers.
 *
 * @param file_name The source filename reported by Clang's SourceManager.
 * @param files_to_retain Exact filenames whose declarations and macros should
 * be included in the parse result.
 * @return true when file_name exactly matches an entry in files_to_retain.
 */
bool filename_is_retained(const std::string &file_name,
                          const std::vector<std::string> &files_to_retain) {
  return std::any_of(files_to_retain.begin(), files_to_retain.end(),
                     [&file_name](const std::string &file_to_retain) {
                       return file_name == file_to_retain;
                     });
}

/**
 * @brief Convert an object-like macro's replacement tokens to source text.
 *
 * Clang stores macro replacement lists as tokens. This helper asks the
 * preprocessor for each token's spelling and joins the spellings with a single
 * space to produce MacroDefinition::replacement_text. Tokens that cannot be
 * spelled are skipped.
 *
 * @param macro_info Clang metadata for an object-like macro definition.
 * @param preprocessor The preprocessor that owns the source manager and
 * language options needed to spell tokens.
 * @return The macro replacement text, or an empty string for flag-style macros
 * such as `#define FLAG`.
 */
std::string replacement_text_from_macro_info(const MacroInfo &macro_info,
                                             const Preprocessor &preprocessor) {
  std::string replacement_text;
  for (const Token &token : macro_info.tokens()) {
    bool invalid = false;
    std::string spelling = preprocessor.getSpelling(token, &invalid);
    if (invalid) {
      continue;
    }

    if (!replacement_text.empty()) {
      replacement_text += " ";
    }
    replacement_text += spelling;
  }
  return replacement_text;
}

/**
 * @brief Populate object-like macro definitions from a parsed ASTUnit.
 *
 * This function reuses the preprocessor state from the ASTUnit created for the
 * declaration matcher pass; it does not invoke Clang or parse the source a
 * second time. It walks the identifier table, keeps only active object-like
 * macros, filters them by their definition spelling location, and writes the
 * resulting name-to-definition map to Declarations::macro_defines. Each
 * definition pairs its replacement text with an optional evaluated constant.
 *
 * Function-like macros are intentionally ignored for now because the public API
 * only models simple name-to-value definitions.
 *
 * @param ast Parsed ASTUnit whose preprocessor state contains the macro table.
 * @param files_to_retain Exact source files whose macro definitions should be
 * returned.
 * @param decls Output declarations object whose macro_defines map is cleared
 * and repopulated.
 */
void collect_macro_defines_from_ast(
    ASTUnit *ast, const std::vector<std::string> &files_to_retain,
    Declarations *decls) {
  decls->macro_defines.clear();

  Preprocessor &preprocessor = ast->getPreprocessor();
  const SourceManager &source_manager = ast->getSourceManager();

  std::vector<std::pair<std::string, const MacroInfo *>> macros;
  for (const auto &entry : preprocessor.getIdentifierTable()) {
    const IdentifierInfo *identifier_info = entry.second;
    if (!identifier_info) {
      continue;
    }

    clang::MacroDefinition macro_definition =
        preprocessor.getMacroDefinition(identifier_info);
    const MacroInfo *macro_info = macro_definition.getMacroInfo();
    if (!macro_info || macro_info->isFunctionLike()) {
      continue;
    }

    SourceLocation spelling_location =
        source_manager.getSpellingLoc(macro_info->getDefinitionLoc());
    if (!spelling_location.isValid()) {
      continue;
    }

    std::string file_name = source_manager.getFilename(spelling_location).str();
    if (!filename_is_retained(file_name, files_to_retain)) {
      continue;
    }

    const std::string name = identifier_info->getName().str();
    decls->macro_defines.emplace(
        name,
        MacroDefinition{
            name, replacement_text_from_macro_info(*macro_info, preprocessor),
            std::nullopt});
    if (!macro_info->tokens_empty()) {
      macros.emplace_back(identifier_info->getName().str(), macro_info);
    }
  }

  // Evaluation can intern identifiers. Finish enumerating the preprocessor's
  // table first, and use a stable order for reproducible evaluation.
  std::sort(macros.begin(), macros.end(),
            [](const auto &a, const auto &b) { return a.first < b.first; });
  auto &diagnostics = ast->getDiagnostics();
  diagnostics.setClient(new IgnoringDiagConsumer(), true);
  preprocessor.enableIncrementalProcessing();
  for (const auto &[name, macro] : macros) {
    diagnostics.Reset(true);
    std::vector<Token> tokens(macro->tokens_begin(), macro->tokens_end());
    Token end;
    end.startToken();
    end.setKind(tok::eof);
    end.setLocation(macro->getDefinitionLoc());
    tokens.push_back(end);
    preprocessor.EnterTokenStream(tokens, false, false);
    Parser parser(preprocessor, ast->getSema(), false);
    parser.Initialize();
    ExprResult expression = parser.ParseExpression();
    if (expression.isInvalid() || diagnostics.hasErrorOccurred() ||
        !parser.getCurToken().is(tok::eof)) {
      // Discard the remainder so a malformed definition cannot contaminate
      // the next probe. Macro definitions need not be C++ expressions.
      while (!parser.getCurToken().is(tok::eof)) {
        parser.ConsumeAnyToken();
      }
      continue;
    }
    expression = ast->getSema().DefaultLvalueConversion(expression.get());
    if (expression.isInvalid()) {
      continue;
    }
    const Expr *expr = expression.get();
    const auto type = expr->getType();
    if (!type->isIntegralOrEnumerationType() &&
        !type->isSpecificBuiltinType(BuiltinType::Float) &&
        !type->isSpecificBuiltinType(BuiltinType::Double) &&
        !type->isSpecificBuiltinType(BuiltinType::LongDouble)) {
      continue;
    }
    APValue value;
    if (!expr->isCXX11ConstantExpr(ast->getASTContext(), &value) ||
        (!value.isInt() && !value.isFloat())) {
      continue;
    }
    llvm::SmallString<32> serialized;
    if (value.isInt()) {
      value.getInt().toString(serialized, 10);
    } else {
      value.getFloat().toString(serialized);
    }
    EvaluatedConstDecl constant;
    constant.name = constant.qual_name = name;
    constant.type_ = Type(type, ast->getASTContext());
    constant.value = std::string(serialized);
    decls->macro_defines.at(name).evaluated_constant = std::move(constant);
  }
}

/**
 * @brief Return the source filename of the declaration.
 */
std::string source_filename_from_decl(const Decl *D) {
  const ASTContext &ast_context = D->getASTContext();
  const SourceManager &sm = ast_context.getSourceManager();
  SourceLocation ELoc = sm.getExpansionLoc(D->getLocation());
  const StringRef file_name_ref = sm.getFilename(ELoc);
  std::string file_name = file_name_ref.str();
  return file_name;
}

// ASTUnit retains the main target, but not the auxiliary CUDA target that
// ASTContext/Sema also reference. Keep both alive for post-parse evaluation.
struct ParsedAST {
  IntrusiveRefCntPtr<TargetInfo> auxiliary_target;
  std::unique_ptr<ASTUnit> ast;
  ASTUnit *operator->() const { return ast.get(); }
  ASTUnit *get() const { return ast.get(); }
};

class RetainAuxiliaryTargetAction : public ASTFrontendAction {
  IntrusiveRefCntPtr<TargetInfo> &target;

public:
  explicit RetainAuxiliaryTargetAction(IntrusiveRefCntPtr<TargetInfo> &target)
      : target(target) {}

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &compiler,
                                                 StringRef) override {
    target = compiler.getAuxTarget();
    return std::make_unique<ASTConsumer>();
  }
};

ParsedAST
default_ast_unit_from_command_line(const std::vector<std::string> &options,
                                   bool bypass_parse_error) {

  std::vector<const char *> option_ptrs;
  for (auto &opt : options) {
    option_ptrs.push_back(opt.c_str());
  }
  const char **argstart = option_ptrs.data();
  const char **argend = argstart + option_ptrs.size();

  auto PCHContainerOps = std::make_shared<PCHContainerOperations>();

  // Create a diagnostics engine that captures errors. Writes error and fatal
  // errors to the diagnostics_consumer. diagnostics_consumer stores the result
  // to a vector of strings.
#if CLANG_VERSION_MAJOR >= 21
  auto DiagOpts = std::make_shared<DiagnosticOptions>();
#elif CLANG_VERSION_MAJOR >= 18 && CLANG_VERSION_MAJOR < 21
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
#else
#error Clang version not supported.
#endif
  auto *diagnostics_consumer = new AstCanopyDiagnosticsConsumer();
  IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS = llvm::vfs::getRealFileSystem();

#if CLANG_VERSION_MAJOR >= 21
  auto Diags = CompilerInstance::createDiagnostics(*FS, *DiagOpts,
                                                   diagnostics_consumer, true);
#elif CLANG_VERSION_MAJOR == 20
  auto Diags = CompilerInstance::createDiagnostics(*FS, &*DiagOpts,
                                                   diagnostics_consumer, true,
                                                   /*CodeGenOptions*/ nullptr);
#elif CLANG_VERSION_MAJOR >= 18 && CLANG_VERSION_MAJOR < 20
  auto Diags = CompilerInstance::createDiagnostics(DiagOpts.get(),
                                                   diagnostics_consumer, true);
#else
#error Clang version not supported.
#endif

  CreateInvocationOptions CIOpts;
  CIOpts.Diags = Diags;
  CIOpts.VFS = FS;
  std::shared_ptr<CompilerInvocation> Invocation =
      createInvocation(llvm::ArrayRef<const char *>(argstart, argend), CIOpts);
  if (!Invocation) {
    throw std::runtime_error("Failed to create a CompilerInvocation.");
  }
  IntrusiveRefCntPtr<TargetInfo> auxiliary_target;
  RetainAuxiliaryTargetAction action(auxiliary_target);
#if CLANG_VERSION_MAJOR >= 21
  std::unique_ptr<ASTUnit> ast(ASTUnit::LoadFromCompilerInvocationAction(
      Invocation, PCHContainerOps, DiagOpts, Diags, &action));
#else
  std::unique_ptr<ASTUnit> ast(ASTUnit::LoadFromCompilerInvocationAction(
      Invocation, PCHContainerOps, Diags, &action));
#endif

  if (!ast) {
    throw std::runtime_error("Failed to create an ASTUnit pointer.");
  } else {
    bool has_error =
        Diags->hasErrorOccurred() || Diags->hasFatalErrorOccurred();

    if (has_error) {
      std::string error_message;
      for (const auto &msg : diagnostics_consumer->error_messages) {
        error_message += "\n" + msg;
      }
      if (!bypass_parse_error) {
        throw ParseError(error_message);
      }
    }
  }

  return {std::move(auxiliary_target), std::move(ast)};
}

} // namespace detail

Declarations
parse_declarations_from_command_line(std::vector<std::string> options,
                                     std::vector<std::string> files_to_retain,
                                     bool bypass_parse_error) {

  auto ast =
      detail::default_ast_unit_from_command_line(options, bypass_parse_error);

  Declarations decls;
  std::unordered_map<int64_t, std::string> record_id_to_name;
  std::unordered_set<int64_t> record_id_with_ctpsd_ancestor;
  detail::traverse_ast_payload payload{&decls, &record_id_to_name,
                                       &files_to_retain,
                                       &record_id_with_ctpsd_ancestor};

  detail::FunctionCallback func_callback(&payload);
  detail::RecordCallback record_callback(&payload);
  detail::TypedefMatcher typedef_callback(&payload);
  detail::FunctionTemplateCallback func_template_callback(&payload);
  detail::ClassTemplateCallback class_template_callback(&payload);
  detail::EnumCallback enum_callback(&payload);
  detail::GlobalConstantCallback constant_callback(&payload);
  detail::ClassTemplatePartialSpecializationCallback ctpsd_callback(&payload);
  detail::ClassTemplateSpecializationCallback ctsd_callback(&payload);

  MatchFinder finder;

  finder.addMatcher(classTemplatePartialSpecializationDecl().bind("ctpsd"),
                    &ctpsd_callback);

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

  finder.addMatcher(varDecl(hasGlobalStorage(),
                            unless(hasAncestor(functionDecl())),
                            unless(hasAncestor(recordDecl())))
                        .bind("global_constant"),
                    &constant_callback);

  finder.addMatcher(classTemplateSpecializationDecl(
                        hasSpecializedTemplate(classTemplateDecl()))
                        .bind("ctsd"),
                    &ctsd_callback);

  finder.matchAST(ast->getASTContext());
  detail::collect_macro_defines_from_ast(ast.get(), files_to_retain, &decls);

#ifndef NDEBUG
  std::cout << "Records: " << decls.records.size() << std::endl;
  std::cout << "Functions: " << decls.functions.size() << std::endl;
  std::cout << "Function templates: " << decls.function_templates.size()
            << std::endl;
  std::cout << "Class templates: " << decls.class_templates.size() << std::endl;
  std::cout << "Class template specializations: "
            << decls.class_template_specializations.size() << std::endl;
  std::cout << "Typedefs: " << decls.typedefs.size() << std::endl;
  std::cout << "Enums: " << decls.enums.size() << std::endl;
  std::cout << "Finished parsing declarations from AST file." << std::endl;
#endif

  return decls;
}

std::optional<EvaluatedConstDecl>
value_from_constexpr_vardecl(std::vector<std::string> options,
                             std::string vardecl_name) {

  auto ast = detail::default_ast_unit_from_command_line(options, false);

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
