// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <clang/ASTMatchers/ASTMatchers.h>

#include <ast_canopy/ast_canopy.hpp>

using namespace clang;
using namespace clang::ast_matchers;

namespace ast_canopy {

namespace detail {

struct traverse_ast_payload {
  Declarations *decls;
  std::unordered_map<int64_t, std::string> *record_id_to_name;
  std::vector<std::string> *files_to_retain;
  // The set of record IDs that has ClassTemplatePartialSpecializationDecl in
  // ancestor
  std::unordered_set<int64_t> *record_id_with_ctpsd_ancestor;
};

struct vardecl_matcher_payload {
  std::string_view name_to_match;
  std::optional<ConstExprVar> var;
};

std::string source_filename_from_decl(const Decl *);

class FunctionCallback : public MatchFinder::MatchCallback {
public:
  FunctionCallback(traverse_ast_payload *);
  void run(const MatchFinder::MatchResult &) override;

private:
  traverse_ast_payload *payload;
};

class FunctionTemplateCallback : public MatchFinder::MatchCallback {
public:
  FunctionTemplateCallback(traverse_ast_payload *);
  void run(const MatchFinder::MatchResult &) override;

private:
  traverse_ast_payload *payload;
};

class RecordCallback : public MatchFinder::MatchCallback {
public:
  RecordCallback(traverse_ast_payload *);
  void run(const MatchFinder::MatchResult &) override;

private:
  traverse_ast_payload *payload;
};

class TypedefMatcher : public MatchFinder::MatchCallback {
public:
  TypedefMatcher(traverse_ast_payload *);
  void run(const MatchFinder::MatchResult &) override;

private:
  traverse_ast_payload *payload;
};

class ClassTemplateCallback : public MatchFinder::MatchCallback {
public:
  ClassTemplateCallback(traverse_ast_payload *);
  void run(const MatchFinder::MatchResult &) override;

private:
  traverse_ast_payload *payload;
};

class EnumCallback : public MatchFinder::MatchCallback {
public:
  EnumCallback(traverse_ast_payload *);
  void run(const MatchFinder::MatchResult &) override;

private:
  traverse_ast_payload *payload;
};

class ConstexprVarDeclCallback : public MatchFinder::MatchCallback {
public:
  ConstexprVarDeclCallback(vardecl_matcher_payload *);
  void run(const MatchFinder::MatchResult &) override;

private:
  vardecl_matcher_payload *payload;
};

class ClassTemplateSpecializationCallback : public MatchFinder::MatchCallback {
public:
  ClassTemplateSpecializationCallback(traverse_ast_payload *);
  void run(const MatchFinder::MatchResult &) override;

private:
  traverse_ast_payload *payload;
};

class ClassTemplatePartialSpecializationCallback
    : public MatchFinder::MatchCallback {
public:
  ClassTemplatePartialSpecializationCallback(traverse_ast_payload *);
  void run(const MatchFinder::MatchResult &) override;

private:
  traverse_ast_payload *payload;
};

} // namespace detail

} // namespace ast_canopy
