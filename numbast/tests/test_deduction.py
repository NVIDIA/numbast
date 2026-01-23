# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import textwrap

import pytest

from ast_canopy import parse_declarations_from_source
from numba.cuda import types as nbtypes

from numbast.deduction import deduce_templated_overloads


_CXX_SOURCE = textwrap.dedent(
    """\
    #pragma once

    template <typename T>
    __device__ T add(T a, T b) { return a + b; }

    template <typename T>
    __device__ T add(T a, T b, T c) { return a + b + c; }

    template <typename T>
    __device__ T add_int(int a, T b) { return a + b; }

    template <typename T>
    __device__ void store_ptr(T *out, T value) { *out = value; }

    template <typename T>
    __device__ void store_ref(T &out, T value) { out = value; }

    template <typename T>
    __device__ T return_only();

    template <typename T>
    __device__ void bad_out(T value) { (void)value; }

    struct Box {
      template <typename T>
      __device__ T mul(T a, T b) const { return a * b; }

      template <typename T>
      __device__ void write(T &out, T value) const { out = value; }
    };
    """
)


@pytest.fixture(scope="module")
def deduction_decls(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("deduction")
    header_path = tmp_dir / "deduction_sample.cuh"
    header_path.write_text(_CXX_SOURCE, encoding="utf-8")
    decls = parse_declarations_from_source(
        str(header_path),
        [str(header_path)],
        "sm_80",
        verbose=False,
    )
    return decls


def _get_function_templates(decls, name: str):
    templs = [
        templ
        for templ in decls.function_templates
        if templ.function.name == name
    ]
    if not templs:
        raise AssertionError(f"No function templates found for {name!r}")
    return templs


def _get_struct_method_templates(decls, struct_name: str, method_name: str):
    for struct in decls.structs:
        if struct.name == struct_name:
            templs = [
                templ
                for templ in struct.templated_methods
                if templ.function.name == method_name
            ]
            if not templs:
                raise AssertionError(
                    f"No templated methods found for {struct_name}::{method_name}"
                )
            return templs
    raise AssertionError(f"Struct {struct_name!r} not found")


def test_overload_arity_selection(deduction_decls):
    """Select the overload that matches the visible argument arity."""
    overloads = _get_function_templates(deduction_decls, "add")
    specialized, intent_errors = deduce_templated_overloads(
        qualname="add",
        overloads=overloads,
        args=(nbtypes.int32, nbtypes.int32),
        overrides=None,
    )

    assert intent_errors == []
    assert len(specialized) == 1
    func = specialized[0].function
    assert len(func.params) == 2
    assert [p.type_.unqualified_non_ref_type_name for p in func.params] == [
        "int",
        "int",
    ]
    assert func.return_type.unqualified_non_ref_type_name == "int"


def test_conflicting_deduction_skips_overload(deduction_decls):
    """Skip overloads when template placeholders deduce conflicting types."""
    overloads = _get_function_templates(deduction_decls, "add")
    specialized, intent_errors = deduce_templated_overloads(
        qualname="add",
        overloads=overloads,
        args=(nbtypes.int32, nbtypes.float32),
        overrides=None,
    )

    assert intent_errors == []
    assert specialized == []


def test_non_templated_param_requires_match(deduction_decls):
    """Require exact matches for non-templated parameters."""
    overloads = _get_function_templates(deduction_decls, "add_int")
    specialized, intent_errors = deduce_templated_overloads(
        qualname="add_int",
        overloads=overloads,
        args=(nbtypes.int32, nbtypes.float32),
        overrides=None,
    )

    assert intent_errors == []
    assert len(specialized) == 1
    func = specialized[0].function
    assert [p.type_.unqualified_non_ref_type_name for p in func.params] == [
        "int",
        "float",
    ]
    assert func.return_type.unqualified_non_ref_type_name == "float"

    specialized, intent_errors = deduce_templated_overloads(
        qualname="add_int",
        overloads=overloads,
        args=(nbtypes.float32, nbtypes.float32),
        overrides=None,
    )

    assert intent_errors == []
    assert specialized == []


def test_return_only_placeholder_skipped(deduction_decls):
    """Skip overloads with unresolved placeholders only in return type."""
    overloads = _get_function_templates(deduction_decls, "return_only")
    specialized, intent_errors = deduce_templated_overloads(
        qualname="return_only",
        overloads=overloads,
        args=(),
        overrides=None,
    )

    assert intent_errors == []
    assert specialized == []


def test_pass_ptr_override_deduces_from_pointer(deduction_decls):
    """Allow out_ptr overrides to pass pointers for reference params."""
    overloads = _get_function_templates(deduction_decls, "store_ref")
    specialized, intent_errors = deduce_templated_overloads(
        qualname="store_ref",
        overloads=overloads,
        args=(nbtypes.CPointer(nbtypes.int32), nbtypes.int32),
        overrides={"out": "out_ptr"},
    )

    assert intent_errors == []
    assert len(specialized) == 1
    func = specialized[0].function
    assert [p.type_.unqualified_non_ref_type_name for p in func.params] == [
        "int",
        "int",
    ]
    assert func.params[0].type_.is_left_reference()
    assert not func.params[1].type_.is_left_reference()


def test_invalid_override_surfaces_error(deduction_decls):
    """Return intent errors when overrides are incompatible with param types."""
    overloads = _get_function_templates(deduction_decls, "bad_out")
    specialized, intent_errors = deduce_templated_overloads(
        qualname="bad_out",
        overloads=overloads,
        args=(nbtypes.int32,),
        overrides={"value": "out_ptr"},
    )

    assert specialized == []
    assert intent_errors
    err = intent_errors[0]
    assert isinstance(err, ValueError)
    assert "reference parameters" in str(err)


def test_struct_method_specialization(deduction_decls):
    """Specialize templated struct methods into concrete types."""
    overloads = _get_struct_method_templates(deduction_decls, "Box", "mul")
    specialized, intent_errors = deduce_templated_overloads(
        qualname="Box.mul",
        overloads=overloads,
        args=(nbtypes.float32, nbtypes.float32),
        overrides=None,
    )

    assert intent_errors == []
    assert len(specialized) == 1
    func = specialized[0].function
    assert func.name == "mul"
    assert [p.type_.unqualified_non_ref_type_name for p in func.params] == [
        "float",
        "float",
    ]
    assert func.return_type.unqualified_non_ref_type_name == "float"


def test_unmappable_numba_arg_skips_overload(deduction_decls):
    """Skip overloads when Numba args cannot map to C++ types."""
    overloads = _get_function_templates(deduction_decls, "add")
    unmappable = nbtypes.float32[:]
    specialized, intent_errors = deduce_templated_overloads(
        qualname="add",
        overloads=overloads,
        args=(unmappable, unmappable),
        overrides=None,
    )

    assert intent_errors == []
    assert specialized == []
