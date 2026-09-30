# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from numbast.cuda_oxide_binding_model import (
    CudaOxideBindingPlan,
    CudaOxideEnum,
    CudaOxideFunction,
    CudaOxideParameter,
    CudaOxideStruct,
    CudaOxideTypeAlias,
)
from numbast.cuda_oxide_renderer import (
    CudaOxideConstant,
    render_cuda_oxide_bindings,
)
from numbast.errors import CudaOxideBindingError
from numbast.rust_types import (
    CudaOxidePointer,
    CudaOxideType,
)


def type_(
    base_name: str, *layers: CudaOxidePointer, spelling: str | None = None
) -> CudaOxideType:
    return CudaOxideType(spelling or base_name, base_name, layers)


def test_renders_complete_cuda_oxide_plan():
    plan = CudaOxideBindingPlan(
        functions=[
            CudaOxideFunction(
                native_name="library_copy",
                public_name="copy",
                execution_space="device",
                return_type=type_("status"),
                parameters=(
                    CudaOxideParameter(
                        "in",
                        "r#in",
                        type_("float", CudaOxidePointer("const")),
                    ),
                    CudaOxideParameter(
                        "handle",
                        "handle",
                        type_("handle_t", CudaOxidePointer("mut")),
                    ),
                ),
            )
        ],
        enums=[
            CudaOxideEnum(
                "status",
                "u32",
                (("STATUS_SUCCESS", "0"), ("match", "1")),
            ),
            CudaOxideEnum("", "i32", (("ANONYMOUS_VALUE", "4"),)),
        ],
        structs=[CudaOxideStruct("handle", 16, 8, "[u64; 2]", ())],
        type_aliases=[CudaOxideTypeAlias("handle_t", type_("handle"))],
        cuda_aliases={"__half": ("f16", 2, 2)},
    )

    rendered = render_cuda_oxide_bindings(plan)

    assert "pub type __half = f16;" in rendered
    assert "core::mem::align_of::<__half>()" in rendered
    assert "pub type handle = [u64; 2];" in rendered
    assert "pub type status = u32;" in rendered
    assert "pub type handle_t = handle;" in rendered
    assert "pub const STATUS_SUCCESS: status = 0;" in rendered
    assert "pub const r#match: status = 1;" in rendered
    assert "pub const ANONYMOUS_VALUE: i32 = 4;" in rendered
    assert (
        "pub fn library_copy(r#in: *const f32, handle: *mut handle_t) -> status;"
        in rendered
    )
    assert "pub use self::library_copy as copy;" in rendered


def test_void_return_and_long_signatures_are_rendered():
    parameters = tuple(
        CudaOxideParameter(f"argument{index}", f"argument{index}", type_("int"))
        for index in range(8)
    )
    plan = CudaOxideBindingPlan(
        functions=[
            CudaOxideFunction(
                "many_arguments",
                "many_arguments",
                "device",
                type_("void"),
                parameters,
            )
        ]
    )

    rendered = render_cuda_oxide_bindings(plan)

    assert "    pub fn many_arguments(\n" in rendered
    assert "        argument0: i32,\n" in rendered
    assert "    );" in rendered
    assert " -> " not in rendered


@pytest.mark.parametrize(
    "enums",
    [
        [
            CudaOxideEnum("", "i32", (("self", "1"),)),
            CudaOxideEnum("", "i32", (("self_", "2"),)),
        ],
        [CudaOxideEnum("", "i32", (("not-an-identifier", "1"),))],
    ],
)
def test_rejects_invalid_or_colliding_constant_names(enums):
    with pytest.raises(CudaOxideBindingError):
        render_cuda_oxide_bindings(CudaOxideBindingPlan(enums=enums))


def test_rejects_constant_and_function_name_collision():
    plan = CudaOxideBindingPlan(
        functions=[
            CudaOxideFunction("VALUE", "VALUE", "device", type_("void"), ())
        ],
        enums=[CudaOxideEnum("", "i32", (("VALUE", "1"),))],
    )

    with pytest.raises(CudaOxideBindingError, match="conflicts with function"):
        render_cuda_oxide_bindings(plan)


def test_renders_explicit_constants_and_checks_parsed_collisions():
    plan = CudaOxideBindingPlan(
        enums=[CudaOxideEnum("", "i32", (("PARSED_VALUE", "1"),))]
    )

    rendered = render_cuda_oxide_bindings(
        plan, [CudaOxideConstant("CONFIGURED_VALUE", "u32", "2")]
    )
    assert "pub const CONFIGURED_VALUE: u32 = 2;" in rendered

    with pytest.raises(CudaOxideBindingError, match="conflicts with parsed"):
        render_cuda_oxide_bindings(
            plan, [CudaOxideConstant("PARSED_VALUE", "i32", "1")]
        )
