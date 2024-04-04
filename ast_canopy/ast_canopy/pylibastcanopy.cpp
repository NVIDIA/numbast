// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ast_canopy.hpp"

namespace py = pybind11;

using namespace ast_canopy;

PYBIND11_MODULE(pylibastcanopy, m) {
  m.doc() = "Python bindings for canopy.hpp";

  py::enum_<execution_space>(m, "execution_space")
      .value("undefined", execution_space::undefined)
      .value("host", execution_space::host)
      .value("device", execution_space::device)
      .value("host_device", execution_space::host_device)
      .value("global_", execution_space::global_);

  py::enum_<method_kind>(m, "method_kind")
      .value("default_constructor", method_kind::default_constructor)
      .value("copy_constructor", method_kind::copy_constructor)
      .value("move_constructor", method_kind::move_constructor)
      .value("other_constructor", method_kind::other_constructor)
      .value("destructor", method_kind::destructor)
      .value("conversion_function", method_kind::conversion_function)
      .value("other", method_kind::other);

  py::enum_<template_param_kind>(m, "template_param_kind")
      .value("type_", template_param_kind::type)
      .value("non_type", template_param_kind::non_type)
      .value("template_", template_param_kind::template_);

  py::class_<Enum>(m, "Enum")
      .def(py::init<const clang::EnumDecl *>())
      .def_readwrite("name", &Enum::name)
      .def_readwrite("enumerators", &Enum::enumerators)
      .def_readwrite("enumerator_values", &Enum::enumerator_values)
      .def(py::pickle(
          [](const Enum &e) {
            return py::make_tuple(e.name, e.enumerators, e.enumerator_values);
          },
          [](py::tuple t) {
            if (t.size() != 3)
              throw std::runtime_error("Invalid state!");
            return Enum{t[0].cast<std::string>(),
                        t[1].cast<std::vector<std::string>>(),
                        t[2].cast<std::vector<std::string>>()};
          }));

  py::class_<Type>(m, "Type")
      .def(py::init<>())
      .def(py::init<std::string, std::string, bool, bool>())
      .def_readwrite("name", &Type::name)
      .def_readwrite("unqualified_non_ref_type_name",
                     &Type::unqualified_non_ref_type_name)
      .def("is_right_reference", &Type::is_right_reference)
      .def("is_left_reference", &Type::is_left_reference)
      .def("__repr__", [](const Type &t) { return "<Type: " + t.name + ">"; })
      .def(py::pickle(
          [](const Type &f) {
            return py::make_tuple(f.name, f.unqualified_non_ref_type_name,
                                  f.is_right_reference(),
                                  f.is_left_reference());
          },
          [](py::tuple t) {
            if (t.size() != 4)
              throw std::runtime_error("Invalid state!");
            return Type{t[0].cast<std::string>(), t[1].cast<std::string>(),
                        t[2].cast<bool>(), t[3].cast<bool>()};
          }));

  py::class_<Field>(m, "Field")
      .def_readwrite("name", &Field::name)
      .def_readwrite("type_", &Field::type)
      .def("__repr__",
           [](const Field &f) {
             return "<Field: " + f.name + " " + f.type.name + ">";
           })
      .def(py::pickle(
          [](const Field &f) { return py::make_tuple(f.name, f.type); },
          [](py::tuple t) {
            if (t.size() != 2)
              throw std::runtime_error("Invalid state!");
            return Field{t[0].cast<std::string>(), t[1].cast<Type>()};
          }));

  py::class_<ParamVar>(m, "ParamVar")
      .def(py::init<std::string, Type>())
      .def_readwrite("name", &ParamVar::name)
      .def_readwrite("type_", &ParamVar::type)
      .def("__repr__",
           [](const ParamVar &p) {
             return "<ParamVar: " + p.name + " " + p.type.name + ">";
           })
      .def(py::pickle(
          [](const ParamVar &p) { return py::make_tuple(p.name, p.type); },
          [](py::tuple t) {
            if (t.size() != 2)
              throw std::runtime_error("Invalid state!");
            return ParamVar{t[0].cast<std::string>(), t[1].cast<Type>()};
          }));

  py::class_<TemplateParam>(m, "TemplateParam")
      .def_readwrite("name", &TemplateParam::name)
      .def_readwrite("type_", &TemplateParam::type)
      .def_readwrite("kind", &TemplateParam::kind)
      .def("__repr__",
           [](const TemplateParam &t) {
             return "<TemplateParam: " + t.name + " " + t.type.name + ">";
           })
      .def(py::pickle(
          [](const TemplateParam &t) {
            return py::make_tuple(t.name, t.kind, t.type);
          },
          [](py::tuple t) {
            if (t.size() != 3)
              throw std::runtime_error("Invalid state!");
            return TemplateParam{t[0].cast<std::string>(),
                                 t[1].cast<template_param_kind>(),
                                 t[2].cast<Type>()};
          }));

  py::class_<Function>(m, "Function")
      .def_readwrite("name", &Function::name)
      .def_readwrite("return_type", &Function::return_type)
      .def_readwrite("params", &Function::params)
      .def_readwrite("exec_space", &Function::exec_space)
      .def(py::pickle(
          [](const Function &f) {
            return py::make_tuple(f.name, f.return_type, f.params,
                                  f.exec_space);
          },
          [](py::tuple t) {
            if (t.size() != 4)
              throw std::runtime_error("Invalid state!");
            return Function{t[0].cast<std::string>(), t[1].cast<Type>(),
                            t[2].cast<std::vector<ParamVar>>(),
                            t[3].cast<execution_space>()};
          }));

  py::class_<Template>(m, "Template")
      .def_readwrite("template_parameters", &Template::template_parameters)
      .def(py::pickle(
          [](const Template &t) {
            return py::make_tuple(t.template_parameters,
                                  t.num_min_required_args);
          },
          [](py::tuple t) {
            if (t.size() != 2)
              throw std::runtime_error("Invalid state!");
            return Template{t[0].cast<std::vector<TemplateParam>>(),
                            t[1].cast<std::size_t>()};
          }));

  py::class_<FunctionTemplate, Template>(m, "FunctionTemplate")
      .def_readwrite("function", &FunctionTemplate::function)
      .def_readwrite("num_min_required_args",
                     &FunctionTemplate::num_min_required_args)
      .def(py::pickle(
          [](const FunctionTemplate &f) {
            Template t = f;
            return py::make_tuple(t, f.function);
          },
          [](py::tuple t) {
            if (t.size() != 2)
              throw std::runtime_error("Invalid state!");
            Template tmpl = t[0].cast<Template>();
            return FunctionTemplate{tmpl.template_parameters,
                                    tmpl.num_min_required_args,
                                    t[1].cast<Function>()};
          }));

  py::class_<ClassTemplate, Template>(m, "ClassTemplate")
      .def_readwrite("num_min_required_args",
                     &ClassTemplate::num_min_required_args)
      .def_readwrite("record", &ClassTemplate::record);

  py::class_<Method, Function>(m, "Method")
      .def_readwrite("kind", &Method::kind)
      .def("is_move_constructor", &Method::is_move_constructor)
      .def(py::pickle(
          [](const Method &m) {
            Function f = m;
            return py::make_tuple(f, m.kind);
          },
          [](py::tuple t) {
            if (t.size() != 2)
              throw std::runtime_error("Invalid state!");
            Function f = t[0].cast<Function>();
            return Method{f.name, f.return_type, f.params, f.exec_space,
                          t[1].cast<method_kind>()};
          }));

  py::class_<Record>(m, "Record")
      .def_readwrite("name", &Record::name)
      .def_readwrite("fields", &Record::fields)
      .def_readwrite("methods", &Record::methods)
      .def_readwrite("templated_methods", &Record::templated_methods)
      .def_readwrite("nested_records", &Record::nested_records)
      .def_readwrite("nested_class_templates", &Record::nested_class_templates)
      .def_readwrite("sizeof_", &Record::sizeof_)
      .def_readwrite("alignof_", &Record::alignof_)
      .def(py::pickle(
          [](const Record &r) {
            return py::make_tuple(r.name, r.fields, r.methods,
                                  r.templated_methods, r.nested_records,
                                  r.nested_class_templates, r.sizeof_,
                                  r.alignof_, r.source_range);
          },
          [](py::tuple t) {
            if (t.size() != 9)
              throw std::runtime_error("Invalid state!");
            return Record{t[0].cast<std::string>(),
                          t[1].cast<std::vector<Field>>(),
                          t[2].cast<std::vector<Method>>(),
                          t[3].cast<std::vector<FunctionTemplate>>(),
                          t[4].cast<std::vector<Record>>(),
                          t[5].cast<std::vector<ClassTemplate>>(),
                          t[6].cast<std::size_t>(),
                          t[7].cast<std::size_t>(),
                          t[8].cast<std::string>()};
          }));

  py::class_<Typedef>(m, "Typedef")
      .def_readwrite("name", &Typedef::name)
      .def_readwrite("underlying_name", &Typedef::underlying_name)
      .def(py::pickle(
          [](const Typedef &t) {
            return py::make_tuple(t.name, t.underlying_name);
          },
          [](py::tuple t) {
            if (t.size() != 2)
              throw std::runtime_error("Invalid state!");
            return Typedef{t[0].cast<std::string>(), t[1].cast<std::string>()};
          }));

  py::class_<Declarations>(m, "Declarations")
      .def_readwrite("records", &Declarations::records)
      .def_readwrite("functions", &Declarations::functions)
      .def_readwrite("function_templates", &Declarations::function_templates)
      .def_readwrite("class_templates", &Declarations::class_templates)
      .def_readwrite("typedefs", &Declarations::typedefs)
      .def_readwrite("enums", &Declarations::enums);

  m.def("parse_declarations_from_command_line",
        &parse_declarations_from_command_line,
        "Parse declarations from command line options.");
}
