# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest

from numba import cuda
import numpy as np
import cffi

from ast_canopy import parse_declarations_from_source
from numbast import bind_cxx_function_templates, MemoryShimWriter


@pytest.fixture
def sample_function_templates():
    DATA_FOLDER = os.path.join(os.path.dirname(__file__), "data")
    p = os.path.join(DATA_FOLDER, "sample_function_template.cuh")
    decls = parse_declarations_from_source(p, [p], "sm_80", verbose=True)
    shim_writer = MemoryShimWriter(f'#include "{p}"')

    func_bindings = bind_cxx_function_templates(
        function_templates=decls.function_templates,
        shim_writer=shim_writer,
    )

    return func_bindings, shim_writer


def find_binding(bindings, name):
    for f in bindings:
        if getattr(f, "__name__", None) == name:
            return f
    raise AssertionError(
        f"Binding '{name}' not found in {[getattr(f, '__name__', None) for f in bindings]}"
    )


def test_templated_function_overload_selection(sample_function_templates):
    func_bindings, shim_writer = sample_function_templates
    add = find_binding(func_bindings, "add")

    @cuda.jit(link=shim_writer.links())
    def kernel(x, y, z, out):
        out[0] = add(x[0], y[0])
        out[1] = add(x[0], y[0], z[0])

    x = np.array([1.0], dtype=np.float32)
    y = np.array([2.0], dtype=np.float32)
    z = np.array([3.0], dtype=np.float32)
    out = np.zeros((2,), dtype=np.float32)
    kernel[1, 1](x, y, z, out)

    np.testing.assert_allclose(out, np.array([3.0, 6.0], dtype=np.float32))


def test_templated_function_explicit_specialization(sample_function_templates):
    func_bindings, shim_writer = sample_function_templates
    add = find_binding(func_bindings, "add")

    @cuda.jit(link=shim_writer.links())
    def kernel(int_a, int_b, float_a, float_b, out_int, out_float):
        out_int[0] = add(int_a[0], int_b[0])
        out_float[0] = add(float_a[0], float_b[0])

    int_a = np.array([1], dtype=np.int32)
    int_b = np.array([2], dtype=np.int32)
    float_a = np.array([1.5], dtype=np.float32)
    float_b = np.array([2.5], dtype=np.float32)
    out_int = np.array([0], dtype=np.int32)
    out_float = np.array([0], dtype=np.float32)
    kernel[1, 1](int_a, int_b, float_a, float_b, out_int, out_float)

    assert out_int[0] == 103
    np.testing.assert_allclose(out_float, np.array([4.0], dtype=np.float32))


def test_templated_function_default_non_type(sample_function_templates):
    func_bindings, shim_writer = sample_function_templates
    add_default = find_binding(func_bindings, "add_default")

    @cuda.jit(link=shim_writer.links())
    def kernel(inp, out):
        out[0] = add_default(inp[0])

    inp = np.array([10], dtype=np.int32)
    out = np.array([0], dtype=np.int32)
    kernel[1, 1](inp, out)

    assert out[0] == 17


def test_templated_function_default_type(sample_function_templates):
    func_bindings, shim_writer = sample_function_templates
    add_default_type = find_binding(func_bindings, "add_default_type")

    @cuda.jit(link=shim_writer.links())
    def kernel(inp, out):
        out[0] = add_default_type(inp[0])

    inp = np.array([9], dtype=np.int32)
    out = np.array([0], dtype=np.int32)
    kernel[1, 1](inp, out)

    assert out[0] == 13


def test_templated_function_multiple_template_args(
    sample_function_templates,
):
    func_bindings, shim_writer = sample_function_templates
    add_cast = find_binding(func_bindings, "add_cast")

    @cuda.jit(link=shim_writer.links())
    def kernel(int_a, float_b, out):
        out[0] = add_cast(int_a[0], float_b[0])

    int_a = np.array([4], dtype=np.int32)
    float_b = np.array([2.5], dtype=np.float32)
    out = np.array([0], dtype=np.int32)
    kernel[1, 1](int_a, float_b, out)

    assert out[0] == 6


def test_templated_function_type_and_non_type(
    sample_function_templates,
):
    func_bindings, shim_writer = sample_function_templates
    add_with_non_type = find_binding(func_bindings, "add_with_non_type")

    @cuda.jit(link=shim_writer.links())
    def kernel(a, b, out):
        out[0] = add_with_non_type(a[0], b[0])

    a = np.array([5], dtype=np.int32)
    b = np.array([6], dtype=np.int32)
    out = np.array([0], dtype=np.int32)
    kernel[1, 1](a, b, out)

    assert out[0] == 16


def test_templated_function_out_return():
    DATA_FOLDER = os.path.join(os.path.dirname(__file__), "data")
    p = os.path.join(DATA_FOLDER, "sample_function_template.cuh")
    decls = parse_declarations_from_source(p, [p], "sm_80", verbose=True)
    shim_writer = MemoryShimWriter(f'#include "{p}"')

    func_bindings = bind_cxx_function_templates(
        function_templates=decls.function_templates,
        shim_writer=shim_writer,
        arg_intent={
            "add_out": {"out": "out_return"},
            "add_out_ret": {"out": "out_return"},
        },
    )

    add_out = find_binding(func_bindings, "add_out")
    add_out_ret = find_binding(func_bindings, "add_out_ret")

    @cuda.jit(link=shim_writer.links())
    def kernel(inp, out_single, out_pair):
        val = inp[0]
        out_single[0] = add_out(val)
        ret, out = add_out_ret(val)
        out_pair[0] = ret
        out_pair[1] = out

    inp = np.array([10], dtype=np.int32)
    out_single = np.array([0], dtype=np.int32)
    out_pair = np.array([0, 0], dtype=np.int32)
    kernel[1, 1](inp, out_single, out_pair)

    assert out_single[0] == 11
    assert out_pair[0] == 13
    assert out_pair[1] == 12


def test_templated_function_out_ptr():
    DATA_FOLDER = os.path.join(os.path.dirname(__file__), "data")
    p = os.path.join(DATA_FOLDER, "sample_function_template.cuh")
    decls = parse_declarations_from_source(p, [p], "sm_80", verbose=True)
    shim_writer = MemoryShimWriter(f'#include "{p}"')

    func_bindings = bind_cxx_function_templates(
        function_templates=decls.function_templates,
        shim_writer=shim_writer,
        arg_intent={"add_out": {"out": "out_ptr"}},
    )

    add_out = find_binding(func_bindings, "add_out")
    ffi = cffi.FFI()

    @cuda.jit(link=shim_writer.links())
    def kernel(inp, out):
        out_ptr = ffi.from_buffer(out)
        add_out(out_ptr, inp[0])

    inp = np.array([8], dtype=np.int32)
    out = np.array([0], dtype=np.int32)
    kernel[1, 1](inp, out)

    assert out[0] == 9
