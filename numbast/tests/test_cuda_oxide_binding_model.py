# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from numbast.cuda_oxide_binding_model import (
    CudaOxideBindingPlan,
    CudaOxideEnum,
    CudaOxideFunction,
    CudaOxideParameter,
    CudaOxideStruct,
    CudaOxideTypeAlias,
)
from numbast.errors import CudaOxideBindingError
from numbast.rust_types import (
    parse_cuda_oxide_type,
    parse_cuda_oxide_type_spelling,
)


class FakeType:
    def __init__(self, name, left_reference=False, right_reference=False):
        self.name = name
        self._left_reference = left_reference
        self._right_reference = right_reference

    def is_left_reference(self):
        return self._left_reference

    def is_right_reference(self):
        return self._right_reference


class StructuredType:
    def __init__(
        self,
        kind,
        *,
        name="deliberately unparsable spelling",
        type_name="",
        inner_type=None,
        array_size=None,
        const=False,
    ):
        self.kind = kind
        self.name = name
        self.type_name = type_name
        self.inner_type = inner_type
        self.array_size = array_size
        self._const = const

    def is_left_reference(self):
        return self.kind == "lvalue_reference"

    def is_right_reference(self):
        return self.kind == "rvalue_reference"

    def is_const_qualified(self):
        return self._const


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
        return_type=(
            FakeType(return_type)
            if isinstance(return_type, str)
            else return_type
        ),
        params=[
            SimpleNamespace(
                name=param_name,
                type_=(
                    FakeType(param_type)
                    if isinstance(param_type, str)
                    else param_type
                ),
            )
            for param_name, param_type in params
        ],
        exec_space=f"execution_space.{execution_space}",
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
def test_type_parser_rejects_unsupported_cuda_oxide_types(type_, message):
    with pytest.raises(ValueError, match=message):
        parse_cuda_oxide_type(type_)


@pytest.mark.parametrize(
    "spelling",
    ["int", "const float *", "unsigned int[2][3]", "const int (*)[2]"],
)
def test_ast_type_and_spelling_parser_agree(spelling):
    assert parse_cuda_oxide_type(FakeType(spelling)) == (
        parse_cuda_oxide_type_spelling(spelling)
    )


def test_structured_ast_type_does_not_parse_its_spelling():
    integer = StructuredType("builtin", type_name="int", const=True)
    pointer = StructuredType("pointer", inner_type=integer)
    array_of_pointers = StructuredType(
        "constant_array", inner_type=pointer, array_size=4
    )
    pointer_to_array = StructuredType(
        "pointer",
        inner_type=StructuredType(
            "constant_array", inner_type=integer, array_size=4, const=True
        ),
    )

    plan = CudaOxideBindingPlan()
    assert plan.render_rust_type(parse_cuda_oxide_type(array_of_pointers)) == (
        "[*const i32; 4]"
    )
    assert plan.render_rust_type(parse_cuda_oxide_type(pointer_to_array)) == (
        "*const [i32; 4]"
    )


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

    plan = CudaOxideBindingPlan.from_declarations(
        parsed,
        config(exclude_functions=["excluded"], skip_prefix="internal_"),
    )

    assert [item.native_name for item in plan.functions] == ["library_copy"]
    selected = plan.functions[0]
    assert selected.public_name == "copy"
    assert [parameter.rust_name for parameter in selected.parameters] == [
        "r#in",
        "output",
    ]
    assert [
        plan.render_rust_type(parameter.type_)
        for parameter in selected.parameters
    ] == ["*const f32", "*const *mut i32"]
    assert {
        (item["kind"], item["name"], item["reason"]) for item in plan.exclusions
    } == {
        ("function", "excluded", "configured"),
        ("function", "host_only", "execution-space:host"),
        ("function", "internal_detail", "skip-prefix"),
        ("function", "library_copy", "duplicate-declaration"),
        ("function-template", "function_template", "round-1-c-api-only"),
        ("class-template", "class_template", "round-1-c-api-only"),
    }


def test_parameter_names_remain_unique_after_generated_suffix_collision():
    plan = CudaOxideBindingPlan.from_declarations(
        declarations(
            functions=[
                function(
                    "parameters",
                    params=(("arg2_2", "int"), ("arg2", "int"), ("", "int")),
                )
            ]
        ),
        config(),
    )

    assert [
        parameter.rust_name for parameter in plan.functions[0].parameters
    ] == [
        "arg2_2",
        "arg2",
        "arg2_3",
    ]


@pytest.mark.parametrize(
    ("bad_function", "message"),
    [
        (function("cpp", "int", c_linkage=False), "does not have C linkage"),
        (function("variadic", "int", variadic=True), "variadic"),
        (function("unknown", "mystery_t"), "unsupported C ABI type"),
        (function("bad_return", ""), "return type: empty type spelling"),
        (
            function("bad_parameter", params=(("value", ""),)),
            "parameter 0: empty type spelling",
        ),
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
    with pytest.raises(CudaOxideBindingError, match=message):
        CudaOxideBindingPlan.from_declarations(
            declarations(functions=[bad_function]), config()
        )


def test_requires_ast_canopy_linkage_metadata():
    candidate = function("old_ast", "int")
    del candidate.is_c_linkage
    with pytest.raises(CudaOxideBindingError, match="Function.is_c_linkage"):
        CudaOxideBindingPlan.from_declarations(
            declarations(functions=[candidate]), config()
        )


def test_linkage_metadata_is_only_required_for_selected_functions():
    ignored = [
        function("excluded"),
        function("internal_detail"),
        function("host_only", execution_space="host"),
    ]
    for candidate in ignored:
        del candidate.is_c_linkage

    plan = CudaOxideBindingPlan.from_declarations(
        declarations(functions=ignored),
        config(exclude_functions=["excluded"], skip_prefix="internal_"),
    )

    assert plan.functions == []
    assert {(item["name"], item["reason"]) for item in plan.exclusions} == {
        ("excluded", "configured"),
        ("host_only", "execution-space:host"),
        ("internal_detail", "skip-prefix"),
    }


def test_collects_type_alias_enum_and_opaque_struct():
    record = SimpleNamespace(
        name="handle",
        sizeof_=16,
        alignof_=8,
        fields=[SimpleNamespace(name="value", type_=FakeType("long"))],
    )
    typedef = SimpleNamespace(
        name="team_t",
        underlying_name="this spelling must not be parsed",
        underlying_type=StructuredType("builtin", type_name="int"),
    )
    enum = SimpleNamespace(
        name="status",
        underlying_type=FakeType("unsigned int"),
        enumerators=["SUCCESS", "FAILURE"],
        enumerator_values=["0", "2"],
    )
    plan = CudaOxideBindingPlan.from_declarations(
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

    assert isinstance(plan, CudaOxideBindingPlan)
    assert isinstance(plan.functions[0], CudaOxideFunction)
    assert isinstance(plan.functions[0].parameters[0], CudaOxideParameter)
    assert isinstance(plan.enums[0], CudaOxideEnum)
    assert isinstance(plan.structs[0], CudaOxideStruct)
    assert isinstance(plan.type_aliases[0], CudaOxideTypeAlias)
    assert [(item.name, item.storage_type) for item in plan.structs] == [
        ("handle", "[u64; 2]")
    ]
    assert [(item.name, item.rust_underlying_type) for item in plan.enums] == [
        ("status", "u32")
    ]
    assert plan.enums[0].enumerators == (
        ("SUCCESS", "0"),
        ("FAILURE", "2"),
    )
    assert [item.name for item in plan.type_aliases] == ["team_t"]
    assert plan.render_rust_type(plan.type_aliases[0].underlying) == "i32"


def test_identity_struct_typedef_does_not_emit_redundant_type_alias():
    record = SimpleNamespace(
        name="record_t",
        sizeof_=4,
        alignof_=4,
        fields=[SimpleNamespace(name="value", type_=FakeType("int"))],
    )
    typedef = SimpleNamespace(
        name="record_t",
        underlying_name="this spelling must not be parsed",
        underlying_type=StructuredType("record", type_name="record_t"),
    )
    plan = CudaOxideBindingPlan.from_declarations(
        declarations(
            functions=[
                function("library_record", params=(("record", "record_t *"),))
            ],
            structs=[record],
            typedefs=[typedef],
        ),
        config(),
    )

    assert [item.name for item in plan.structs] == ["record_t"]
    assert plan.type_aliases == []


def test_public_alias_cannot_shadow_another_native_symbol():
    with pytest.raises(
        CudaOxideBindingError, match="conflicts with native symbol"
    ):
        CudaOxideBindingPlan.from_declarations(
            declarations(
                functions=[
                    function("library_foo", "int"),
                    function("library_library_foo", "int"),
                ]
            ),
            config(),
        )


def test_rust_keyword_function_names_are_preserved():
    plan = CudaOxideBindingPlan.from_declarations(
        declarations(
            functions=[
                function("match", "int"),
                function("union", "int"),
                function("library_type", "int"),
            ]
        ),
        config(),
    )

    assert [item.native_name for item in plan.functions] == [
        "library_type",
        "match",
        "union",
    ]
    assert plan.functions[0].public_name == "type"

    with pytest.raises(CudaOxideBindingError, match="exact CUDA-Oxide"):
        CudaOxideBindingPlan.from_declarations(
            declarations(functions=[function("self", "int")]), config()
        )


def test_cuda_storage_aliases_are_architecture_specific():
    parsed = declarations(
        functions=[
            function("library_half", params=(("value", "__half *"),)),
            function(
                "library_bfloat",
                params=(("value", "__nv_bfloat16 *"),),
            ),
            function("library_vector", params=(("values", "const double2 *"),)),
        ]
    )
    legacy = CudaOxideBindingPlan.from_declarations(parsed, config())

    assert legacy.cuda_aliases == {
        "__half": ("u16", 2, 2),
        "__nv_bfloat16": ("u16", 2, 2),
        "double2": ("[u128; 1]", 16, 16),
    }
    modern = CudaOxideBindingPlan.from_declarations(
        parsed, config(gpu_arch=["sm_100"])
    )
    assert modern.cuda_aliases["__half"] == ("f16", 2, 2)


@pytest.mark.parametrize(
    "type_name",
    [
        "bool",
        "char",
        "short",
        "uint8_t",
        "uint16_t",
        "__half",
        "__nv_bfloat16",
    ],
)
def test_pre_blackwell_rejects_sub_32_bit_values_at_extern_boundary(type_name):
    parsed = declarations(
        functions=[function("small_value", params=(("value", type_name),))]
    )

    with pytest.raises(CudaOxideBindingError, match=r"sm_100\+"):
        CudaOxideBindingPlan.from_declarations(parsed, config())

    modern = CudaOxideBindingPlan.from_declarations(
        parsed, config(gpu_arch=["sm_100f"])
    )
    assert modern.functions[0].parameters[0].type_.base_name == type_name


def test_pre_blackwell_allows_sub_32_bit_values_behind_pointers():
    plan = CudaOxideBindingPlan.from_declarations(
        declarations(
            functions=[
                function(
                    "small_pointers",
                    params=(("character", "char *"), ("half", "__half *")),
                )
            ]
        ),
        config(),
    )

    assert len(plan.functions[0].parameters) == 2


def test_pre_blackwell_rejects_enum_with_sub_32_bit_underlying_type():
    enum = SimpleNamespace(
        name="small_enum",
        underlying_type=FakeType("unsigned short"),
        enumerators=["VALUE"],
        enumerator_values=["0"],
    )

    with pytest.raises(CudaOxideBindingError, match=r"sm_100\+"):
        CudaOxideBindingPlan.from_declarations(
            declarations(
                functions=[
                    function(
                        "small_enum_value", params=(("value", "small_enum"),)
                    )
                ],
                enums=[enum],
            ),
            config(),
        )


def test_invalid_struct_storage_is_reported():
    record = SimpleNamespace(
        name="bad_record", sizeof_=12, alignof_=8, fields=[]
    )
    with pytest.raises(CudaOxideBindingError, match="cannot represent"):
        CudaOxideBindingPlan.from_declarations(
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
