# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from numba_cuda_mlir import cuda
from numba_cuda_mlir.types import Type
from numba_cuda_mlir.models import StructModel


def _check_shim_name_contains_mangled_name(src: str, mangled_name: str):
    lower_name = f"_lower_{mangled_name}"
    idx = src.find(lower_name)
    assert idx != -1, (
        f"Failed to find the lower function name {lower_name} in the source"
    )

    lower_body = src[idx:]
    idx = lower_body.find('extern "C" __device__')
    assert idx != -1, (
        'Failed to find the extern "C" __device__ in the lower function body'
    )

    idx_end = lower_body.find("}")
    assert idx_end != -1, "Failed to find the end of the lower function body"

    shim_raw_str = lower_body[idx:idx_end]

    assert mangled_name in shim_raw_str, (
        f"Failed to find the mangled name {mangled_name} in the shim raw string"
    )


def test_conflict_func_names(make_binding):
    res1 = make_binding(
        "conflict_func_name_1.cuh", {"Foo": Type}, {"Foo": StructModel}
    )
    res2 = make_binding(
        "conflict_func_name_2.cuh", {"Bar": Type}, {"Bar": StructModel}
    )

    binding1 = res1["bindings"]
    binding2 = res2["bindings"]

    with open("/tmp/conflict_func_name_1.py", "w") as f:
        f.write(res1["src"])

    with open("/tmp/conflict_func_name_2.py", "w") as f:
        f.write(res2["src"])

    Foo = binding1["Foo"]
    Bar = binding2["Bar"]
    foo_type_name = Foo._nbtype.name
    bar_type_name = Bar._nbtype.name

    assert foo_type_name.startswith("Foo::")
    assert bar_type_name.startswith("Bar::")
    assert "_type_class_Foo" in foo_type_name
    assert "_type_class_Bar" in bar_type_name
    assert foo_type_name != bar_type_name

    _check_shim_name_contains_mangled_name(res1["src"], "_ZN3FooC1Ev")
    _check_shim_name_contains_mangled_name(res2["src"], "_ZN3BarC1Ev")

    @cuda.jit
    def kernel():
        foo = Foo()
        bar = Bar()
        _ = foo + foo
        _ = bar + bar

    kernel[1, 1]()
