# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import pickle
from dataclasses import astuple

import pytest

from pylibastcanopy import template_param_kind, execution_space, access_kind
from ast_canopy import parse_declarations_from_source


@pytest.fixture(scope="module")
def sample_struct_source():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_directory, "data/", "sample_struct.cu")


@pytest.fixture(scope="module")
def sample_function_source():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_directory, "data/", "sample_function.cu")


@pytest.fixture(scope="module")
def sample_typedef_source():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_directory, "data/", "sample_typedef.cu")


@pytest.fixture(scope="module")
def sample_function_template_source():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_directory, "data/", "sample_function_template.cu")


@pytest.fixture(scope="module")
def sample_class_template_source():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_directory, "data/", "sample_class_template.cu")


@pytest.fixture(scope="module")
def sample_nested_structs_source():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_directory, "data/", "sample_nested_struct.cu")


@pytest.fixture(scope="module")
def sample_access_specifier_source():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_directory, "data/", "sample_access_specifier.cu")


@pytest.fixture(scope="module")
def sample_enum_source():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_directory, "data/", "sample_enum.cu")


@pytest.fixture(scope="module")
def sample_execution_space_source():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_directory, "data/", "sample_execution_space.cu")


@pytest.fixture(scope="module")
def sample_load_by_cc_source():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_directory, "data/", "sample_diff_by_cc.cu")


@pytest.fixture(scope="module", params=[False, True], ids=["no_pickle", "pickle"])
def test_pickle(request):
    return request.param


def test_load_ast_structs(sample_struct_source, test_pickle):
    decls = parse_declarations_from_source(
        sample_struct_source, [sample_struct_source], "sm_80"
    )

    structs, _, _, _, _, _ = astuple(decls)

    if test_pickle:
        pickled = [pickle.dumps(s) for s in structs]
        structs = [pickle.loads(p) for p in pickled]

    assert len(structs) == 1
    assert structs[0].name == "foo"
    assert structs[0].fields[0].name == "a"
    assert structs[0].fields[0].type_.name == "float2"
    assert structs[0].sizeof_ == 16
    assert structs[0].alignof_ == 16

    s = structs[0]
    assert len(s.methods) == 11

    meth = s.methods[0]
    assert meth.name == "foo"
    assert meth.return_type.name == "void"
    assert [a.name for a in meth.params] == ["a_"]
    assert [a.type_.name for a in meth.params] == ["float2"]

    meth = s.methods[1]
    assert meth.name == "foo"
    assert meth.return_type.name == "void"
    assert meth.is_move_constructor
    assert [a.name for a in meth.params] == ["other"]
    assert [a.type_.name for a in meth.params] == ["foo &&"]
    assert [a.type_.is_right_reference() for a in meth.params] == [True]
    assert [a.type_.unqualified_non_ref_type_name for a in meth.params] == ["foo"]

    meth = s.methods[2]
    assert meth.name == "add"
    assert meth.return_type.name == "double"
    assert [a.name for a in meth.params] == ["a", "b"]
    assert [a.type_.name for a in meth.params] == ["double", "double"]

    meth = s.methods[3]
    assert meth.name == "operator float2"
    assert meth.return_type.name == "float2"
    assert [a.name for a in meth.params] == []
    assert [a.type_.name for a in meth.params] == []

    meth = s.methods[4]
    assert meth.name == "operator-"
    assert meth.return_type.name == "foo"
    assert [a.name for a in meth.params] == []
    assert [a.type_.name for a in meth.params] == []

    meth = s.methods[5]
    assert meth.name == "operator+"
    assert meth.return_type.name == "foo"
    assert [a.name for a in meth.params] == ["other"]
    assert [a.type_.name for a in meth.params] == ["const foo &"]
    assert [a.type_.is_left_reference() for a in meth.params] == [True]

    meth = s.methods[6]
    assert meth.name == "operator+="
    assert meth.return_type.name == "foo &"
    assert [a.name for a in meth.params] == ["other"]
    assert [a.type_.name for a in meth.params] == ["const foo &"]
    assert [a.type_.is_left_reference() for a in meth.params] == [True]

    meth = s.methods[7]
    assert meth.name == "operator=="
    assert meth.return_type.name == "bool"
    assert [a.name for a in meth.params] == ["other"]
    assert [a.type_.name for a in meth.params] == ["const foo &"]
    assert [a.type_.is_left_reference() for a in meth.params] == [True]

    meth = s.methods[8]
    assert meth.name == "operator[]"
    assert meth.return_type.name == "float &"
    assert [a.name for a in meth.params] == ["i"]
    assert [a.type_.name for a in meth.params] == ["int"]

    meth = s.methods[9]
    assert meth.name == "operator()"
    assert meth.return_type.name == "float"
    assert [a.name for a in meth.params] == ["x"]
    assert [a.type_.name for a in meth.params] == ["float"]

    meth = s.methods[10]
    assert meth.name == "operator="
    assert meth.return_type.name == "foo &"
    assert [a.name for a in meth.params] == ["other"]
    assert [a.type_.name for a in meth.params] == ["const foo &"]
    assert [a.type_.is_left_reference() for a in meth.params] == [True]

    assert len(s.templated_methods) == 1
    assert s.templated_methods[0].num_min_required_args == 1
    assert s.templated_methods[0].function.name == "bar"
    assert s.templated_methods[0].function.return_type.name == "type-parameter-0-0"
    assert len(s.templated_methods[0].function.params) == 0
    assert len(s.templated_methods[0].template_parameters) == 1
    assert s.templated_methods[0].template_parameters[0].name == "T"
    assert (
        s.templated_methods[0].template_parameters[0].type_.name == "type-parameter-0-0"
    )


def test_load_ast_functions(sample_function_source, test_pickle):
    decls = parse_declarations_from_source(
        sample_function_source, [sample_function_source], "sm_80"
    )

    _, functions, _, _, _, _ = astuple(decls)

    if test_pickle:
        pickled = [pickle.dumps(f) for f in functions]
        functions = [pickle.loads(p) for p in pickled]

    assert len(functions) == 5
    assert functions[0].name == "add"
    assert functions[0].return_type.name == "int"
    args = functions[0].params
    assert [a.name for a in args] == ["a", "b"]
    assert [a.type_.name for a in args] == ["int", "int"]

    assert functions[1].name == "mul"
    assert functions[1].return_type.name == "int"
    args = functions[1].params
    assert [a.name for a in args] == ["a", "b"]
    assert [a.type_.name for a in args] == [
        "const int &",
        "const int *",
    ]  # Note the const qualifier style is reordered
    assert [a.type_.unqualified_non_ref_type_name for a in args] == [
        "int",
        "int *",
    ]

    assert functions[2].name == "add2"
    assert functions[2].return_type.name == "int"
    args = functions[2].params
    assert [a.name for a in args] == ["a", "b"]
    assert [a.type_.name for a in args] == ["int &&", "int &&"]
    assert [a.type_.is_right_reference() for a in args] == [True, True]
    assert [a.type_.unqualified_non_ref_type_name for a in args] == ["int", "int"]

    assert functions[3].name == "add_hostdevice"
    assert functions[3].return_type.name == "int"
    args = functions[3].params
    assert [a.name for a in args] == ["a", "b"]
    assert [a.type_.name for a in args] == ["int", "int"]

    assert functions[4].name == "add_constexpr"
    assert functions[4].return_type.name == "int"
    assert functions[4].is_constexpr
    args = functions[4].params
    assert [a.name for a in args] == ["a", "b"]
    assert [a.type_.name for a in args] == ["int", "int"]


def test_load_ast_typedefs(sample_typedef_source, test_pickle):
    decls = parse_declarations_from_source(
        sample_typedef_source, [sample_typedef_source], "sm_80"
    )

    structs, _, _, _, typedefs, _ = astuple(decls)

    if test_pickle:
        pickled = [pickle.dumps(t) for t in typedefs]
        typedefs = [pickle.loads(p) for p in pickled]

        pickled = [pickle.dumps(s) for s in structs]
        structs = [pickle.loads(p) for p in pickled]

    assert len(structs) == 3
    assert len(typedefs) == 3

    assert structs[0].name == "A"
    second_struct_name = structs[1].name  # Originally unnamed, renamed with its AST ID.
    assert structs[2].name == "C"

    assert typedefs[0].name == "A1"
    assert typedefs[0].underlying_name == "A"
    assert typedefs[1].name == "B"
    assert typedefs[1].underlying_name == second_struct_name
    assert typedefs[2].name == "C1"
    assert typedefs[2].underlying_name == "C"


def test_load_ast_function_templates(sample_function_template_source, test_pickle):
    decls = parse_declarations_from_source(
        sample_function_template_source, [sample_function_template_source], "sm_80"
    )

    _, _, ft, _, _, _ = astuple(decls)

    if test_pickle:
        pickled = [pickle.dumps(f) for f in ft]
        ft = [pickle.loads(p) for p in pickled]

    assert len(ft) == 2

    # ft[0]
    assert ft[0].num_min_required_args == 3

    assert ft[0].function.name == "foo"
    assert ft[0].function.return_type.name == "void"

    assert len(ft[0].template_parameters) == 3

    assert ft[0].template_parameters[0].name == "T"
    assert (
        ft[0].template_parameters[0].type_.name == "type-parameter-0-0"
    )  # is this consistent?
    assert ft[0].template_parameters[0].kind == template_param_kind.type_

    assert ft[0].template_parameters[1].name == "N"
    assert ft[0].template_parameters[1].type_.name == "int"
    assert ft[0].template_parameters[1].kind == template_param_kind.non_type

    assert ft[0].template_parameters[2].name == "e"
    assert ft[0].template_parameters[2].type_.name == "E"
    assert ft[0].template_parameters[2].kind == template_param_kind.non_type

    # ft[1]
    assert ft[1].num_min_required_args == 0


def test_load_ast_class_templates(sample_class_template_source, test_pickle):
    decls = parse_declarations_from_source(
        sample_class_template_source, [sample_class_template_source], "sm_80"
    )

    _, _, _, ct, _, _ = astuple(decls)

    if test_pickle:
        pickled = [pickle.dumps(c) for c in ct]
        ct = [pickle.loads(p) for p in pickled]

    assert len(ct) == 1

    assert len(ct[0].template_parameters) == 2

    assert ct[0].template_parameters[0].name == "T"
    assert ct[0].template_parameters[0].type_.name == "type-parameter-0-0"

    assert ct[0].template_parameters[1].name == "e"
    assert ct[0].template_parameters[1].type_.name == "E"

    assert len(ct[0].record.fields) == 1
    assert ct[0].record.fields[0].name == "t"
    assert ct[0].record.fields[0].type_.name == "type-parameter-0-0"

    assert len(ct[0].record.methods) == 2
    assert ct[0].record.methods[0].decl_name == "Foo"
    assert ct[0].record.methods[0].return_type.name == "void"
    assert len(ct[0].record.methods[0].params) == 0

    assert ct[0].record.methods[1].decl_name == "baz"
    assert ct[0].record.methods[1].return_type.name == "void"
    assert len(ct[0].record.methods[1].params) == 0

    assert len(ct[0].record.templated_methods) == 1

    assert ct[0].record.templated_methods[0].function.name == "bar"
    assert ct[0].record.templated_methods[0].function.return_type.name == "E"
    assert len(ct[0].record.templated_methods[0].function.params) == 2
    assert ct[0].record.templated_methods[0].function.params[0].name == "t"
    assert (
        ct[0].record.templated_methods[0].function.params[0].type_.name
        == "type-parameter-0-0"
    )
    assert ct[0].record.templated_methods[0].function.params[1].name == "u"
    assert (
        ct[0].record.templated_methods[0].function.params[1].type_.name
        == "type-parameter-1-0"
    )
    assert len(ct[0].record.templated_methods[0].template_parameters) == 1
    assert ct[0].record.templated_methods[0].template_parameters[0].name == "U"
    assert (
        ct[0].record.templated_methods[0].template_parameters[0].type_.name
        == "type-parameter-1-0"
    )


def test_load_ast_nested_structs(sample_nested_structs_source, test_pickle):
    decls = parse_declarations_from_source(
        sample_nested_structs_source, [sample_nested_structs_source], "sm_80"
    )

    structs, _, _, _, _, _ = astuple(decls)

    if test_pickle:
        pickled = [pickle.dumps(s) for s in structs]
        structs = [pickle.loads(p) for p in pickled]

    assert len(structs) == 1

    assert len(structs[0].nested_records) == 1

    assert len(structs[0].nested_records[0].methods) == 1

    assert structs[0].nested_records[0].methods[0].name == "bar"


def test_load_ast_access_specifiers(sample_access_specifier_source, test_pickle):
    decls = parse_declarations_from_source(
        sample_access_specifier_source, [sample_access_specifier_source], "sm_80"
    )

    structs, _, _, _, _, _ = astuple(decls)

    if test_pickle:
        pickled = [pickle.dumps(s) for s in structs]
        structs = [pickle.loads(p) for p in pickled]

    assert len(structs) == 3

    assert len(structs[0].methods) == 1  # Only 1 public method
    assert structs[0].methods[0].name == "Bar2"

    assert len(structs[1].methods) == 2  # 2 public methods
    assert structs[1].methods[0].name == "Bar1"
    assert structs[1].methods[1].name == "Bar2"

    # All fields are visible, with access specifier information.
    assert len(structs[2].fields) == 3
    assert structs[2].fields[0].name == "x"
    assert structs[2].fields[0].type_.name == "int"
    assert structs[2].fields[0].access == access_kind.private_

    assert structs[2].fields[1].name == "y"
    assert structs[2].fields[1].type_.name == "int"
    assert structs[2].fields[1].access == access_kind.public_

    assert structs[2].fields[2].name == "z"
    assert structs[2].fields[2].type_.name == "int"
    assert structs[2].fields[2].access == access_kind.protected_


def test_load_enum(sample_enum_source, test_pickle):
    decls = parse_declarations_from_source(
        sample_enum_source, [sample_enum_source], "sm_80"
    )

    _, _, _, _, _, enums = astuple(decls)

    if test_pickle:
        pickled = [pickle.dumps(e) for e in enums]
        enums = [pickle.loads(p) for p in pickled]

    assert len(enums) == 3
    assert enums[0].name == "Foo"
    assert len(enums[0].enumerators) == 3
    assert enums[0].enumerators[0] == "A"
    assert enums[0].enumerators[1] == "B"
    assert enums[0].enumerators[2] == "C"
    assert len(enums[0].enumerator_values) == 3
    assert enums[0].enumerator_values[0] == "1"
    assert enums[0].enumerator_values[1] == "2"
    assert enums[0].enumerator_values[2] == "3"

    assert enums[1].name == "NoDefaultFoo"
    assert len(enums[1].enumerators) == 3
    assert enums[1].enumerators[0] == "D"
    assert enums[1].enumerators[1] == "E"
    assert enums[1].enumerators[2] == "F"
    assert len(enums[1].enumerator_values) == 3
    assert enums[1].enumerator_values[0] == "0"
    assert enums[1].enumerator_values[1] == "1"
    assert enums[1].enumerator_values[2] == "2"

    assert enums[2].name == "Bar"
    assert len(enums[2].enumerators) == 1
    assert enums[2].enumerators[0] == "A"
    assert len(enums[2].enumerator_values) == 1
    assert enums[2].enumerator_values[0] == "1"


def test_load_struct_function_execution_space(
    sample_execution_space_source, test_pickle
):
    decls = parse_declarations_from_source(
        sample_execution_space_source, [sample_execution_space_source], "sm_80"
    )

    structs, functions, _, _, _, _ = astuple(decls)

    if test_pickle:
        pickled = [pickle.dumps(s) for s in structs]
        structs = [pickle.loads(p) for p in pickled]

        pickled = [pickle.dumps(f) for f in functions]
        functions = [pickle.loads(p) for p in pickled]

    assert len(structs) == 1
    assert len(structs[0].methods) == 4
    assert structs[0].methods[0].name == "dfoo"
    assert structs[0].methods[0].exec_space == execution_space.device
    assert structs[0].methods[1].name == "hfoo"
    assert structs[0].methods[1].exec_space == execution_space.host
    assert structs[0].methods[2].name == "dhfoo"
    assert structs[0].methods[2].exec_space == execution_space.host_device
    assert structs[0].methods[3].name == "foo"
    assert structs[0].methods[3].exec_space == execution_space.undefined

    assert len(functions) == 5
    assert functions[0].name == "dfoo"
    assert functions[0].exec_space == execution_space.device
    assert functions[1].name == "hfoo"
    assert functions[1].exec_space == execution_space.host
    assert functions[2].name == "gfoo"
    assert functions[2].exec_space == execution_space.global_
    assert functions[3].name == "dhfoo"
    assert functions[3].exec_space == execution_space.host_device
    assert functions[4].name == "foo"
    assert functions[4].exec_space == execution_space.undefined


@pytest.mark.parametrize(
    "cc, answer",
    [
        (
            "sm_70",
            {
                "structs": [
                    {"name": "Functor", "methods": ["operator()"], "fields": ["k"]}
                ],
                "functions": [{"name": "fpi", "return_type": "float"}],
            },
        ),
        (
            "sm_86",
            {
                "structs": [
                    {
                        "name": "Functor",
                        "methods": ["operator()", "operator()"],
                        "fields": ["k", "dk"],
                    },
                    {
                        "name": "AdvancedFunctor",
                        "methods": ["operator()"],
                        "fields": ["k"],
                    },
                ],
                "functions": [
                    {"name": "fpi", "return_type": "float"},
                    {"name": "dpi", "return_type": "double"},
                ],
            },
        ),
    ],
)
def test_load_by_cc(cc, answer, sample_load_by_cc_source):
    decls = parse_declarations_from_source(
        sample_load_by_cc_source, [sample_load_by_cc_source], cc
    )

    structs, functions, _, _, _, _ = astuple(decls)

    assert len(structs) == len(answer["structs"])
    assert len(functions) == len(answer["functions"])

    # We test it with a zip iterator of answer["struct"] to also make sure that
    # the AST parsing respects the definition order by user. Similarly for functions
    # below.
    for s, answer_s in zip(structs, answer["structs"]):
        assert s.name == answer_s["name"]
        assert len(s.methods) == len(answer_s["methods"])
        assert len(s.fields) == len(answer_s["fields"])
        for m, answer_s_m_name in zip(s.methods, answer_s["methods"]):
            assert m.name == answer_s_m_name
        for f, answer_s_f_name in zip(s.fields, answer_s["fields"]):
            assert f.name == answer_s_f_name

    for f, answer_f in zip(functions, answer["functions"]):
        assert f.name == answer_f["name"]
        assert f.return_type.name == answer_f["return_type"]
