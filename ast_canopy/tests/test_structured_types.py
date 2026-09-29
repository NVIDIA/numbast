# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pickle

import pytest

from ast_canopy import parse_declarations_from_source
from ast_canopy import pylibastcanopy as bindings


@pytest.fixture(scope="module")
def declarations(data_folder):
    path = data_folder / "structured_types.cu"
    return parse_declarations_from_source(
        str(path), [str(path)], "sm_80", cuda_header_mode=True
    )


def _fields(declarations):
    record = next(
        item for item in declarations.structs if item.name == "StructuredTypes"
    )
    return {field.name: field.type_ for field in record.fields}


def _semantic_type(type_):
    transparent_kinds = {
        bindings.type_kind.adjusted,
        bindings.type_kind.sugar,
    }
    while type_.kind in transparent_kinds:
        type_ = type_.inner_type
    return type_


def test_qualifiers_are_attached_to_the_qualified_type_layer(declarations):
    fields = _fields(declarations)

    pointer_to_const = fields["pointer_to_const"]
    assert pointer_to_const.kind == bindings.type_kind.pointer
    assert not pointer_to_const.is_const_qualified()
    assert pointer_to_const.inner_type.kind == bindings.type_kind.builtin
    assert pointer_to_const.inner_type.type_name == "int"
    assert pointer_to_const.inner_type.is_const_qualified()

    const_pointer = fields["const_pointer"]
    assert const_pointer.kind == bindings.type_kind.pointer
    assert const_pointer.is_const_qualified()
    assert not const_pointer.inner_type.is_const_qualified()


def test_array_and_pointer_declarators_remain_distinct(declarations):
    fields = _fields(declarations)

    array_of_pointers = fields["array_of_pointers"]
    assert array_of_pointers.kind == bindings.type_kind.constant_array
    assert array_of_pointers.array_size == 4
    assert array_of_pointers.inner_type.kind == bindings.type_kind.pointer

    pointer_to_array = fields["pointer_to_array"]
    assert pointer_to_array.kind == bindings.type_kind.pointer
    array = _semantic_type(pointer_to_array.inner_type)
    assert array.kind == bindings.type_kind.constant_array
    assert array.array_size == 4
    assert array.inner_type.type_name == "int"


def test_function_pointers_and_references_are_structural(declarations):
    callback = _fields(declarations)["callback"]
    assert callback.kind == bindings.type_kind.pointer
    assert (
        _semantic_type(callback.inner_type).kind == bindings.type_kind.function
    )

    function = next(
        item for item in declarations.functions if item.name == "references"
    )
    assert function.params[0].type_.kind == bindings.type_kind.lvalue_reference
    assert function.params[0].type_.inner_type.type_name == "int"
    assert function.params[1].type_.kind == bindings.type_kind.rvalue_reference
    assert function.params[1].type_.inner_type.type_name == "int"


def test_named_types_and_typedef_underlying_types_are_structural(declarations):
    fields = _fields(declarations)

    alias = fields["alias"]
    assert alias.kind == bindings.type_kind.pointer
    alias_type = _semantic_type(alias.inner_type)
    assert alias_type.kind == bindings.type_kind.typedef_
    assert alias_type.type_name == "word_t"
    aliased_record = _semantic_type(alias_type.inner_type)
    assert aliased_record.kind == bindings.type_kind.record
    assert aliased_record.type_name == "WordStorage"

    enum = _semantic_type(fields["status"])
    assert enum.kind == bindings.type_kind.enum_
    assert enum.type_name == "Status"

    typedef = next(
        item for item in declarations.typedefs if item.name == "word_t"
    )
    underlying = _semantic_type(typedef.underlying_type)
    assert underlying.kind == bindings.type_kind.record
    assert underlying.type_name == "WordStorage"


def test_structured_types_survive_pickle(declarations):
    restored = pickle.loads(pickle.dumps(declarations.typedefs))
    typedef = next(item for item in restored if item.name == "word_t")
    underlying = _semantic_type(typedef.underlying_type)
    assert underlying.kind == bindings.type_kind.record
    assert underlying.type_name == "WordStorage"

    restored = pickle.loads(pickle.dumps(declarations.structs))
    record = next(item for item in restored if item.name == "StructuredTypes")
    fields = {field.name: field.type_ for field in record.fields}
    pointer_to_array = fields["pointer_to_array"]
    assert _semantic_type(pointer_to_array.inner_type).array_size == 4
    assert fields["pointer_to_const"].inner_type.is_const_qualified()
