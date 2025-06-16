// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#pragma once

#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <clang/AST/Decl.h>
#include <clang/AST/DeclTemplate.h>
#include <clang/AST/Type.h>

namespace ast_canopy {

struct TemplateParam;
struct ClassTemplate;

enum class execution_space { undefined, host, device, host_device, global_ };

enum class method_kind {
  default_constructor, // Default constructor (C++98)
  copy_constructor,    // Copy constructor (C++98)
  move_constructor,    // Move constructor (C++11)
  other_constructor,   // Other constructors
  destructor,          // Destructor (C++98)
  conversion_function, // Conversion function (C++11)
  other                // All other methods
};

enum class template_param_kind { type, non_type, template_ };

enum class access_kind { public_, protected_, private_ };

struct Declaration {
  // Stores nested namespace names in bottom-up order (deepest to top-level)
  // For example, for a declaration in "outer::middle::inner", the vector
  // would contain ["inner", "middle", "outer"]
  std::vector<std::string> namespace_stack;

  Declaration() = default;
  explicit Declaration(const std::vector<std::string> &ns_stack);
  explicit Declaration(const clang::Decl *decl);
  virtual ~Declaration() = default;
};

struct Enum : public Declaration {
  Enum(const std::string &name, const std::vector<std::string> &enumerators,
       const std::vector<std::string> &enumerator_values,
       const std::vector<std::string> &namespace_stack);
  Enum(const clang::EnumDecl *);

  std::string name;
  std::vector<std::string> enumerators;
  std::vector<std::string> enumerator_values;
};

struct Type {
  Type() = default;
  Type(std::string name, std::string unqualified_non_ref_type_name,
       bool is_right_reference, bool is_left_reference);
  Type(const clang::QualType &, const clang::ASTContext &context);

  std::string name;
  std::string unqualified_non_ref_type_name;

  bool is_right_reference() const;
  bool is_left_reference() const;

private:
  bool _is_right_reference;
  bool _is_left_reference;
};

struct ConstExprVar {
  ConstExprVar() = default;
  ConstExprVar(const clang::VarDecl *VD);

  Type type_;
  std::string name;
  std::string value;
};

struct Field {
  Field(const clang::FieldDecl *FD, const clang::AccessSpecifier &AS);
  Field(const std::string &name, const Type &type, const access_kind &access);

  std::string name;
  Type type;
  access_kind access;
};

struct ParamVar {
  ParamVar(std::string name, Type type);
  ParamVar(const clang::ParmVarDecl *PVD);

  std::string name;
  Type type;
};

struct Template {
  Template(const std::vector<TemplateParam> &template_parameters,
           const std::size_t &num_min_required_args);
  Template(const clang::TemplateParameterList *);

  std::vector<TemplateParam> template_parameters;
  std::size_t num_min_required_args;
  virtual ~Template() = default;
};

struct TemplateParam {
  TemplateParam(const std::string &name, template_param_kind kind, Type type)
      : name(name), kind(kind), type(type) {}
  TemplateParam(const clang::TemplateTypeParmDecl *);
  TemplateParam(const clang::NonTypeTemplateParmDecl *);
  TemplateParam(const clang::TemplateTemplateParmDecl *);

  std::string name;
  template_param_kind kind;
  Type type;
};

struct Function : public Declaration {
  Function(const std::string &name, const Type &return_type,
           const std::vector<ParamVar> &params,
           const execution_space &exec_space,
           const std::vector<std::string> &namespace_stack);
  Function(const clang::FunctionDecl *);

  std::string name;
  Type return_type;
  std::vector<ParamVar> params;
  execution_space exec_space;
  bool is_constexpr;
};

struct FunctionTemplate : public Template, public Declaration {
  FunctionTemplate(const std::vector<TemplateParam> &template_parameters,
                   const std::size_t &num_min_required_args,
                   const Function &function,
                   const std::vector<std::string> &namespace_stack);
  FunctionTemplate(const clang::FunctionTemplateDecl *);
  Function function;
};

struct Method : public Function {
  Method(const std::string &name, const Type &return_type,
         const std::vector<ParamVar> &params, const execution_space &exec_space,
         const method_kind &kind,
         const std::vector<std::string> &namespace_stack);
  Method(const clang::CXXMethodDecl *);
  method_kind kind;

  bool is_move_constructor();

private:
  const clang::CXXMethodDecl *_clang_method;
};

enum class RecordAncestor {
  ANCESTOR_IS_TEMPLATE,
  ANCESTOR_IS_NOT_TEMPLATE,
};

struct Record : public Declaration {
  Record(const std::string &name, const std::vector<Field> &fields,
         const std::vector<Method> &methods,
         const std::vector<FunctionTemplate> &templated_methods,
         const std::vector<Record> &nested_records,
         const std::vector<ClassTemplate> &nested_class_templates,
         const std::size_t &sizeof_, const std::size_t &alignof_,
         const std::string &source_range,
         const std::vector<std::string> &namespace_stack);
  Record(const clang::CXXRecordDecl *, RecordAncestor);
  Record(const clang::CXXRecordDecl *, RecordAncestor,
         std::string); // overrides name from RD

  std::string name;
  std::vector<Field> fields;
  std::vector<Method> methods;
  std::vector<FunctionTemplate> templated_methods;
  std::vector<Record> nested_records;
  std::vector<ClassTemplate> nested_class_templates;
  std::size_t sizeof_;
  std::size_t alignof_;

  std::string source_range;
  void print(int) const;
};

struct ClassTemplate : public Template, public Declaration {
  ClassTemplate(const clang::ClassTemplateDecl *);
  ClassTemplate(const std::vector<TemplateParam> &template_parameters,
                const std::size_t &num_min_required_args, const Record &record,
                const std::vector<std::string> &namespace_stack);
  Record record;
};

struct Typedef : public Declaration {
  Typedef(const std::string &name, const std::string &underlying_name,
          const std::vector<std::string> &namespace_stack);
  Typedef(const clang::TypedefDecl *,
          std::unordered_map<int64_t, std::string> *);

  std::string name;
  std::string underlying_name;
};

struct Declarations {
  std::vector<Record> records;
  std::vector<Function> functions;
  std::vector<FunctionTemplate> function_templates;
  std::vector<ClassTemplate> class_templates;
  std::vector<Typedef> typedefs;
  std::vector<Enum> enums;
};

Declarations parse_declarations_from_command_line(
    std::vector<std::string> options, std::vector<std::string> files_to_retain,
    std::vector<std::string> whitelist_prefixes);

std::optional<ConstExprVar>
value_from_constexpr_vardecl(std::vector<std::string> clang_options,
                             std::string vardecl_name);

} // namespace ast_canopy
