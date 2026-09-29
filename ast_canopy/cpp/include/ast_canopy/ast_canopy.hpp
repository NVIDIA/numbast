// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <clang/AST/Decl.h>
#include <clang/AST/DeclTemplate.h>
#include <clang/AST/Type.h>

#include <ast_canopy/error.hpp>

namespace ast_canopy {

inline constexpr std::size_t INVALID_SIZE_OF =
    std::numeric_limits<std::size_t>::max();
inline constexpr std::size_t INVALID_ALIGN_OF =
    std::numeric_limits<std::size_t>::max();

struct TemplateParam;
struct ClassTemplate;

enum class execution_space { undefined, host, device, host_device, global_ };

enum class method_kind {
  default_constructor,    // Default constructor (C++98)
  copy_constructor,       // Copy constructor (C++98)
  move_constructor,       // Move constructor (C++11)
  converting_constructor, // Converting constructor (C++11)
  other_constructor,      // Other constructors
  destructor,             // Destructor (C++98)
  conversion_function,    // Conversion function (C++11)
  other                   // All other methods
};

enum class template_param_kind { type, non_type, template_ };

enum class access_kind { public_, protected_, private_ };

// A stable, serializable view of the Clang type classes needed by AST Canopy
// consumers. Qualifiers are stored on every Type node, while inner_type()
// links pointer, reference, array, typedef, and transparent sugar nodes.
enum class type_kind {
  unknown,
  builtin,
  pointer,
  lvalue_reference,
  rvalue_reference,
  constant_array,
  incomplete_array,
  variable_array,
  dependent_sized_array,
  record,
  enum_,
  typedef_,
  function,
  member_pointer,
  elaborated,
  paren,
  attributed,
  adjusted,
  other
};

struct Type {
  Type() = default;
  Type(std::string name, std::string unqualified_non_ref_type_name,
       bool is_right_reference, bool is_left_reference);
  Type(std::string name, std::string unqualified_non_ref_type_name,
       bool is_right_reference, bool is_left_reference, type_kind kind,
       bool is_const_qualified, bool is_volatile_qualified,
       bool is_restrict_qualified, std::vector<Type> inner_types,
       std::optional<std::uint64_t> array_size, std::string type_name);
  Type(const clang::QualType &, const clang::ASTContext &context);

  std::string name;
  std::string unqualified_non_ref_type_name;
  type_kind kind = type_kind::unknown;
  std::optional<std::uint64_t> array_size;
  std::string type_name;

  bool is_right_reference() const { return _is_right_reference; }
  bool is_left_reference() const { return _is_left_reference; }
  bool is_const_qualified() const { return _is_const_qualified; }
  bool is_volatile_qualified() const { return _is_volatile_qualified; }
  bool is_restrict_qualified() const { return _is_restrict_qualified; }
  const Type *inner_type() const {
    return _inner_types.empty() ? nullptr : &_inner_types.front();
  }

private:
  // A vector provides recursive value storage while maintaining the invariant
  // that the currently supported structural nodes have at most one child.
  std::vector<Type> _inner_types;
  bool _is_right_reference = false;
  bool _is_left_reference = false;
  bool _is_const_qualified = false;
  bool _is_volatile_qualified = false;
  bool _is_restrict_qualified = false;
};

struct Enum {
  Enum(const std::string &name, const std::string &qual_name,
       const std::vector<std::string> &enumerators,
       const std::vector<std::string> &enumerator_values,
       const Type &underlying_type)
      : name(name), qual_name(qual_name), enumerators(enumerators),
        enumerator_values(enumerator_values), underlying_type(underlying_type) {
  }
  Enum(const clang::EnumDecl *);

  std::string name;
  std::string qual_name;
  std::vector<std::string> enumerators;
  std::vector<std::string> enumerator_values;
  Type underlying_type;
};

struct ConstExprVar {
  ConstExprVar() = default;
  ConstExprVar(const clang::VarDecl *VD);

  Type type_;
  std::string name;
  std::string qual_name;
  std::string value;
};

struct Field {
  Field(const clang::FieldDecl *FD, const clang::AccessSpecifier &AS);
  Field(const std::string &name, const Type &type, const access_kind &access)
      : name(name), type(type), access(access) {};

  std::string name;
  Type type;
  access_kind access;
};

struct ParamVar {
  ParamVar(std::string name, Type type) : name(std::move(name)), type(type) {}
  ParamVar(const clang::ParmVarDecl *PVD);

  std::string name;
  Type type;
};

struct Template {
  Template(const std::vector<TemplateParam> &template_parameters,
           const std::size_t &num_min_required_args)
      : template_parameters(template_parameters),
        num_min_required_args(num_min_required_args) {}
  Template(const clang::TemplateParameterList *);

  std::vector<TemplateParam> template_parameters;
  std::size_t num_min_required_args;
  virtual ~Template() = default;
};

struct TemplateParam {
  TemplateParam(const std::string &name, template_param_kind kind, Type type,
                bool is_pack = false)
      : name(name), kind(kind), type(type), is_pack(is_pack) {}
  TemplateParam(const clang::TemplateTypeParmDecl *);
  TemplateParam(const clang::NonTypeTemplateParmDecl *);
  TemplateParam(const clang::TemplateTemplateParmDecl *);

  std::string name;
  template_param_kind kind;
  Type type;
  bool is_pack;
};

struct Function {
  Function(const std::string &name, const Type &return_type,
           const std::vector<ParamVar> &params,
           const execution_space &exec_space)
      : name(name), qual_name(name), return_type(return_type), params(params),
        exec_space(exec_space) {}
  Function(const std::string &name, const Type &return_type,
           const std::vector<ParamVar> &params,
           const execution_space &exec_space, const std::string &qual_name)
      : name(name), qual_name(qual_name), return_type(return_type),
        params(params), exec_space(exec_space) {}
  Function(const clang::FunctionDecl *);

  std::string name;
  std::string qual_name;
  Type return_type;
  std::vector<ParamVar> params;
  execution_space exec_space;
  bool is_constexpr;
  bool is_c_linkage = false;
  bool is_variadic = false;
  std::string mangled_name;
  std::set<std::string> attributes;
};

struct FunctionTemplate : public Template {
  FunctionTemplate(const std::vector<TemplateParam> &template_parameters,
                   const std::size_t &num_min_required_args,
                   const Function &function)
      : Template(std::move(template_parameters), num_min_required_args),
        qual_name(function.qual_name), function(std::move(function)) {}
  FunctionTemplate(const std::vector<TemplateParam> &template_parameters,
                   const std::size_t &num_min_required_args,
                   const Function &function, const std::string &qual_name)
      : Template(std::move(template_parameters), num_min_required_args),
        qual_name(qual_name), function(std::move(function)) {}
  FunctionTemplate(const clang::FunctionTemplateDecl *);
  std::string qual_name;
  Function function;
};

struct Method : public Function {
  Method(const std::string &name, const Type &return_type,
         const std::vector<ParamVar> &params, const execution_space &exec_space,
         const std::string &qual_name, const method_kind &kind)
      : Function(name, return_type, params, exec_space, qual_name), kind(kind) {
  }
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

struct Record {
  Record(const std::string &name, const std::vector<Field> &fields,
         const std::vector<Method> &methods,
         const std::vector<FunctionTemplate> &templated_methods,
         const std::vector<Record> &nested_records,
         const std::vector<ClassTemplate> &nested_class_templates,
         const std::size_t &sizeof_, const std::size_t &alignof_,
         const std::string &source_range)
      : name(name), qual_name(name), fields(fields), methods(methods),
        templated_methods(templated_methods), nested_records(nested_records),
        nested_class_templates(nested_class_templates), sizeof_(sizeof_),
        alignof_(alignof_), source_range(source_range) {}
  Record(const std::string &name, const std::vector<Field> &fields,
         const std::vector<Method> &methods,
         const std::vector<FunctionTemplate> &templated_methods,
         const std::vector<Record> &nested_records,
         const std::vector<ClassTemplate> &nested_class_templates,
         const std::size_t &sizeof_, const std::size_t &alignof_,
         const std::string &source_range, const std::string &qual_name)
      : name(name), qual_name(qual_name), fields(fields), methods(methods),
        templated_methods(templated_methods), nested_records(nested_records),
        nested_class_templates(nested_class_templates), sizeof_(sizeof_),
        alignof_(alignof_), source_range(source_range) {}
  Record(const clang::CXXRecordDecl *, RecordAncestor);
  Record(const clang::CXXRecordDecl *, RecordAncestor,
         std::string); // overrides name from RD

  std::string name;
  std::string qual_name;
  std::vector<Field> fields;
  std::vector<Method> methods;
  std::vector<FunctionTemplate> templated_methods;
  std::vector<Record> nested_records;
  std::vector<ClassTemplate> nested_class_templates;
  std::size_t sizeof_;
  std::size_t alignof_;
  bool is_union = false; // true for a C/C++ union, false for a struct/class

  std::string source_range;
  void print(int) const;
};

struct ClassTemplate : public Template {
  ClassTemplate(const clang::ClassTemplateDecl *);
  std::string qual_name;
  Record record;
};

struct Typedef {
  Typedef(const std::string &name, const std::string &underlying_name)
      : name(name), qual_name(name), underlying_name(underlying_name),
        underlying_type(underlying_name, underlying_name, false, false) {}
  Typedef(const std::string &name, const std::string &underlying_name,
          const std::string &qual_name)
      : name(name), qual_name(qual_name), underlying_name(underlying_name),
        underlying_type(underlying_name, underlying_name, false, false) {}
  Typedef(const std::string &name, const std::string &underlying_name,
          const std::string &qual_name, const Type &underlying_type)
      : name(name), qual_name(qual_name), underlying_name(underlying_name),
        underlying_type(underlying_type) {}
  Typedef(const clang::TypedefDecl *,
          std::unordered_map<int64_t, std::string> *);

  std::string name;
  std::string qual_name;
  std::string underlying_name;
  Type underlying_type;
};

struct ClassTemplateSpecialization : public Record {
  ClassTemplateSpecialization(const clang::ClassTemplateSpecializationDecl *);

  ClassTemplate class_template;
  std::vector<std::string> actual_template_arguments;
};

struct Declarations {
  std::vector<Record> records;
  std::vector<Function> functions;
  std::vector<FunctionTemplate> function_templates;
  std::vector<ClassTemplate> class_templates;
  std::vector<ClassTemplateSpecialization> class_template_specializations;
  std::vector<Typedef> typedefs;
  std::vector<Enum> enums;
};

Declarations
parse_declarations_from_command_line(std::vector<std::string> options,
                                     std::vector<std::string> files_to_retain,
                                     bool bypass_parse_error);

std::optional<ConstExprVar>
value_from_constexpr_vardecl(std::vector<std::string> clang_options,
                             std::string vardecl_name);

} // namespace ast_canopy
