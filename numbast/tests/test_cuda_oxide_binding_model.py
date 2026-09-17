# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from numbast.cuda_oxide_binding_model import (
    BindingModelError,
    build_c_device_model,
    modern_nvvm_required_symbols,
    translate_constant_literal,
)
from numbast.rust_types import parse_c_abi_type, rust_type


class FakeType:
    def __init__(self, name, left_reference=False, right_reference=False):
        self.name = name
        self._left_reference = left_reference
        self._right_reference = right_reference

    def is_left_reference(self):
        return self._left_reference

    def is_right_reference(self):
        return self._right_reference


def function(
    name,
    return_type="void",
    params=(),
    execution_space="device",
    c_linkage=True,
    variadic=False,
    mangled_name=None,
):
    return SimpleNamespace(
        name=name,
        return_type=FakeType(return_type),
        params=[
            SimpleNamespace(name=param_name, type_=FakeType(param_type))
            for param_name, param_type in params
        ],
        exec_space=SimpleNamespace(name=execution_space),
        is_c_linkage=c_linkage,
        is_variadic=variadic,
        mangled_name=(
            mangled_name
            if mangled_name is not None
            else name
            if c_linkage
            else f"_Z{len(name)}{name}"
        ),
    )


def declarations(**overrides):
    values = {
        "functions": [],
        "function_templates": [],
        "class_templates": [],
        "typedefs": [],
        "enums": [],
        "structs": [],
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def config(**overrides):
    values = {
        "exclude_structs": [],
        "exclude_functions": [],
        "skip_prefix": None,
        "api_prefix_removal": {"Function": ["library_"]},
        "gpu_arch": ["sm_90"],
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    ("type_", "message"),
    [
        (FakeType("int", left_reference=True), "C\\+\\+ references"),
        (FakeType("int &"), "C\\+\\+ references"),
        (FakeType("void (*)(int)"), "function/member pointers"),
        (FakeType(""), "empty type spelling"),
    ],
)
def test_type_parser_rejects_non_c_abi_types(type_, message):
    with pytest.raises(ValueError, match=message):
        parse_c_abi_type(type_)


@pytest.mark.parametrize(
    ("value", "translated"),
    [
        (True, "true"),
        (False, "false"),
        (-7, "-7"),
        ("0x2aULL", "0x2a"),
        ("077u", "077"),
    ],
)
def test_constant_literals_are_normalized(value, translated):
    assert translate_constant_literal(value) == translated


def test_constant_expressions_are_rejected():
    with pytest.raises(ValueError, match="unsupported non-literal"):
        translate_constant_literal("1 << 4")


def test_selects_c_device_surface_and_records_exclusions():
    duplicate = function(
        "library_copy",
        "int",
        (("in", "const float *"), ("output", "int *const *")),
    )
    parsed = declarations(
        functions=[
            duplicate,
            duplicate,
            function("host_only", execution_space="host"),
            function("excluded"),
            function("internal_detail"),
        ],
        function_templates=[
            SimpleNamespace(function=SimpleNamespace(name="function_template"))
        ],
        class_templates=[
            SimpleNamespace(record=SimpleNamespace(name="class_template"))
        ],
    )

    model = build_c_device_model(
        parsed,
        config(exclude_functions=["excluded"], skip_prefix="internal_"),
    )

    assert [item.native_name for item in model.functions] == ["library_copy"]
    selected = model.functions[0]
    assert selected.public_name == "copy"
    assert [parameter.rust_name for parameter in selected.parameters] == [
        "r#in",
        "output",
    ]
    assert [
        rust_type(parameter.type_, model) for parameter in selected.parameters
    ] == ["*const f32", "*const *mut i32"]
    assert {
        (item["kind"], item["name"], item["reason"])
        for item in model.exclusions
    } == {
        ("function", "excluded", "configured"),
        ("function", "host_only", "execution-space:host"),
        ("function", "internal_detail", "skip-prefix"),
        ("function", "library_copy", "duplicate-declaration"),
        ("function-template", "function_template", "round-1-c-api-only"),
        ("class-template", "class_template", "round-1-c-api-only"),
    }


@pytest.mark.parametrize(
    ("bad_function", "message"),
    [
        (function("cpp", "int", c_linkage=False), "does not have C linkage"),
        (function("variadic", "int", variadic=True), "variadic"),
        (function("unknown", "mystery_t"), "unsupported C ABI type"),
        (
            function("vector", params=(("value", "double2"),)),
            "only supports it behind a pointer",
        ),
        (
            function("mismatch", mangled_name="different_symbol"),
            "C symbol mismatch",
        ),
    ],
)
def test_strictly_rejects_out_of_contract_device_declarations(
    bad_function, message
):
    with pytest.raises(BindingModelError, match=message):
        build_c_device_model(declarations(functions=[bad_function]), config())


def test_requires_ast_canopy_linkage_metadata():
    candidate = function("old_ast", "int")
    del candidate.is_c_linkage
    with pytest.raises(BindingModelError, match="Function.is_c_linkage"):
        build_c_device_model(declarations(functions=[candidate]), config())


def test_collects_typedef_enum_and_opaque_record_abi():
    record = SimpleNamespace(
        name="handle",
        sizeof_=16,
        alignof_=8,
        fields=[SimpleNamespace(name="value", type_=FakeType("long"))],
    )
    typedef = SimpleNamespace(name="team_t", underlying_name="int")
    enum = SimpleNamespace(
        name="status",
        underlying_type=FakeType("unsigned int"),
        enumerators=["SUCCESS", "FAILURE"],
        enumerator_values=[0, "0x2u"],
    )
    model = build_c_device_model(
        declarations(
            functions=[
                function(
                    "library_apply",
                    "status",
                    (("team", "team_t"), ("handle", "handle *")),
                )
            ],
            typedefs=[typedef],
            enums=[enum],
            structs=[record],
        ),
        config(),
    )

    assert [(item.name, item.storage_type) for item in model.records] == [
        ("handle", "[u64; 2]")
    ]
    assert [(item.name, item.rust_underlying_type) for item in model.enums] == [
        ("status", "u32")
    ]
    assert model.enums[0].enumerators == (
        ("SUCCESS", "0"),
        ("FAILURE", "0x2"),
    )
    assert [item.name for item in model.typedefs] == ["team_t"]
    assert rust_type(model.typedefs[0].underlying, model) == "i32"


def test_identity_record_typedef_is_emitted_once():
    record = SimpleNamespace(
        name="record_t",
        sizeof_=4,
        alignof_=4,
        fields=[SimpleNamespace(name="value", type_=FakeType("int"))],
    )
    typedef = SimpleNamespace(
        name="record_t", underlying_name="struct record_t"
    )
    model = build_c_device_model(
        declarations(
            functions=[
                function("library_record", params=(("record", "record_t *"),))
            ],
            structs=[record],
            typedefs=[typedef],
        ),
        config(),
    )

    assert [item.name for item in model.records] == ["record_t"]
    assert model.typedefs == []


def test_public_alias_cannot_shadow_another_native_symbol():
    with pytest.raises(BindingModelError, match="conflicts with native symbol"):
        build_c_device_model(
            declarations(
                functions=[
                    function("library_foo", "int"),
                    function("library_library_foo", "int"),
                ]
            ),
            config(),
        )


def test_rust_keyword_function_names_are_preserved():
    model = build_c_device_model(
        declarations(
            functions=[
                function("match", "int"),
                function("union", "int"),
                function("library_type", "int"),
            ]
        ),
        config(),
    )

    assert [item.native_name for item in model.functions] == [
        "library_type",
        "match",
        "union",
    ]
    assert model.functions[0].public_name == "type"

    with pytest.raises(BindingModelError, match="exact CUDA-Oxide"):
        build_c_device_model(
            declarations(functions=[function("self", "int")]), config()
        )


def test_cuda_storage_aliases_and_modern_nvvm_requirements():
    parsed = declarations(
        functions=[
            function("library_half", "__half", (("value", "__half"),)),
            function(
                "library_bfloat",
                "__nv_bfloat16",
                (("value", "__nv_bfloat16"),),
            ),
            function("library_vector", params=(("values", "const double2 *"),)),
        ]
    )
    legacy = build_c_device_model(parsed, config())

    assert legacy.cuda_aliases == {
        "__half": ("u16", 2, 2),
        "__nv_bfloat16": ("u16", 2, 2),
        "double2": ("[u128; 1]", 16, 16),
    }
    assert modern_nvvm_required_symbols(legacy) == [
        "library_bfloat",
        "library_half",
    ]
    modern = build_c_device_model(parsed, config(gpu_arch=["sm_100"]))
    assert modern.cuda_aliases["__half"] == ("f16", 2, 2)


def test_invalid_record_storage_is_reported():
    record = SimpleNamespace(
        name="bad_record", sizeof_=12, alignof_=8, fields=[]
    )
    with pytest.raises(BindingModelError, match="cannot represent"):
        build_c_device_model(
            declarations(
                functions=[
                    function(
                        "library_bad",
                        params=(("record", "bad_record *"),),
                    )
                ],
                structs=[record],
            ),
            config(),
        )
