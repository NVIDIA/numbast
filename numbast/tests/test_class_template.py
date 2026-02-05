# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import textwrap

import numpy as np

from numba import cuda
from numba.cuda import types as nbtypes

import cffi


from ast_canopy import parse_declarations_from_source
from numbast import bind_cxx_class_templates, MemoryShimWriter
from numbast.class_template import _make_templated_method_shim_arg_strings


import pytest

ffi = cffi.FFI()


@pytest.fixture
def _sample_class_templates():
    DATA_FOLDER = os.path.join(os.path.dirname(__file__), "data")
    p = os.path.join(DATA_FOLDER, "sample_class_template.cuh")
    decls = parse_declarations_from_source(p, [p], "sm_80", verbose=True)
    shim_writer = MemoryShimWriter(f'#include "{p}"')

    apis = bind_cxx_class_templates(
        decls.class_templates, header_path=p, shim_writer=shim_writer
    )

    return apis, shim_writer


@pytest.fixture
def _sample_class_template_templated_methods():
    """Bindings fixture for class-template + templated-method unit tests."""
    DATA_FOLDER = os.path.join(os.path.dirname(__file__), "data")
    p = os.path.join(DATA_FOLDER, "sample_class_template_templated_method.cuh")
    decls = parse_declarations_from_source(p, [p], "sm_80", verbose=True)
    shim_writer = MemoryShimWriter(f'#include "{p}"')

    apis = bind_cxx_class_templates(
        decls.class_templates, header_path=p, shim_writer=shim_writer
    )

    return apis, shim_writer


@pytest.fixture
def decl(_sample_class_templates):
    return _sample_class_templates[0]


@pytest.fixture
def shim_writer(_sample_class_templates):
    return _sample_class_templates[1]


def test_sample_class_template_simple(decl, shim_writer):
    T = np.int32

    BlockScan = decl[0]

    # Demo
    @cuda.jit(link=shim_writer.links())
    def kern(arr_in, arr_out):
        i = cuda.grid(1)

        block_scan_t = BlockScan(T=T, BLOCK_DIM_X=128)
        block_scan = block_scan_t()

        ptr_out = ffi.from_buffer(arr_out[i:])
        block_scan.InclusiveSum(arr_in[i], ptr_out)

    arrin = np.arange(1024, dtype=T)
    out = np.zeros_like(arrin)
    kern[1, 1](arrin, out)


def test_sample_class_template_with_fields(decl, shim_writer):
    T = np.int32
    Foo = decl[1]

    @cuda.jit(link=shim_writer.links())
    def kernel(out):
        foo_t = Foo(T=T, N=128)

        foo = foo_t(256)

        out[0] = foo.t
        out[1] = foo.get_t()
        out[2] = foo.get_t2()

    out = np.zeros((3,), dtype="int32")
    kernel[1, 1](out)

    assert (out == [256, 256, 128]).all()


def test_class_template_default_param(decl, shim_writer):
    T = np.int32
    DefaultParam = next(api for api in decl if api.__name__ == "DefaultParam")

    @cuda.jit(link=shim_writer.links())
    def kernel(inp, out):
        i = cuda.grid(1)
        if i >= out.size:
            return
        default_t = DefaultParam(T=T)
        default_obj = default_t(inp[i])
        out[i] = default_obj.add_default()

    x = np.arange(1, 9, dtype=T)
    out = np.zeros_like(x)
    kernel[1, 32](x, out)
    np.testing.assert_array_equal(out, x + 5)


@pytest.mark.parametrize(
    "intent_kind",
    [
        "out_return",
        "out_ptr",
    ],
)
def test_class_template_arg_intent_regular_method(intent_kind):
    DATA_FOLDER = os.path.join(os.path.dirname(__file__), "data")
    p = os.path.join(DATA_FOLDER, "sample_class_template.cuh")
    decls = parse_declarations_from_source(p, [p], "sm_80", verbose=True)
    shim_writer = MemoryShimWriter(f'#include "{p}"')

    apis = bind_cxx_class_templates(
        decls.class_templates,
        header_path=p,
        shim_writer=shim_writer,
        arg_intent={"BlockScan": {"AddToRef": {"out": intent_kind}}},
    )

    BlockScan = apis[0]
    T = np.int32
    x = np.arange(1, 9, dtype=T)
    out = np.zeros_like(x)

    if intent_kind == "out_return":

        @cuda.jit(link=shim_writer.links())
        def kernel(inp, out):
            i = cuda.grid(1)
            if i >= out.size:
                return
            block_scan_t = BlockScan(T=T, BLOCK_DIM_X=128)
            block_scan = block_scan_t()
            out[i] = block_scan.AddToRef(inp[i])

    else:

        @cuda.jit(link=shim_writer.links())
        def kernel(inp, out):
            i = cuda.grid(1)
            if i >= out.size:
                return
            block_scan_t = BlockScan(T=T, BLOCK_DIM_X=128)
            block_scan = block_scan_t()
            out_ptr = ffi.from_buffer(out[i:])
            block_scan.AddToRef(inp[i], out_ptr)

    kernel[1, 32](x, out)
    np.testing.assert_array_equal(out, x + 1)


class TestTemplatedClassTemplatedMethod:
    """
    Covered scenarios:
    - Method template defaults: call without explicit method template args.
    - arg_intent handling for templated method outputs (return vs out ptr).

    Notes about current state:
    - The templated-method plumbing exists, but there's no user-facing API for
      specifying method template parameters yet, so explicit/partial
      specialization and missing-parameter error cases are not covered here.
    """

    @pytest.fixture
    def decl(self, _sample_class_template_templated_methods):
        return _sample_class_template_templated_methods[0]

    @pytest.fixture
    def shim_writer(self, _sample_class_template_templated_methods):
        return _sample_class_template_templated_methods[1]

    def test_method_template_defaults_no_explicit_specialization_needed(
        self, decl, shim_writer
    ):
        """
        Method template has defaults => should compile and run without any
        explicit template args.
        """
        TMix = decl[0]
        T = np.int32

        @cuda.jit(link=shim_writer.links())
        def kernel(x, out):
            i = cuda.grid(1)

            if i >= out.size:
                return

            tmix_t = TMix(T=T, N=7)
            tmix = tmix_t()

            out_ptr = ffi.from_buffer(out[i:])
            tmix.AddConstDefault(x[i], out_ptr)

        x = np.arange(1, 9, dtype=T)
        out = np.zeros_like(x)
        kernel[1, 32](x, out)

        np.testing.assert_array_equal(out, x + 7)

    @pytest.mark.parametrize(
        "intent_kind",
        [
            "out_return",
            "out_ptr",
        ],
    )
    def test_method_template_arg_intent(self, intent_kind):
        DATA_FOLDER = os.path.join(os.path.dirname(__file__), "data")
        p = os.path.join(
            DATA_FOLDER, "sample_class_template_templated_method.cuh"
        )
        decls = parse_declarations_from_source(p, [p], "sm_80", verbose=True)
        shim_writer = MemoryShimWriter(f'#include "{p}"')

        apis = bind_cxx_class_templates(
            decls.class_templates,
            header_path=p,
            shim_writer=shim_writer,
            arg_intent={"TMix": {"AddConstRef": {"out": intent_kind}}},
        )

        TMix = apis[0]
        T = np.int32
        x = np.arange(1, 9, dtype=T)
        out = np.zeros_like(x)

        if intent_kind == "out_return":

            @cuda.jit(link=shim_writer.links())
            def kernel(inp, out):
                i = cuda.grid(1)
                if i >= out.size:
                    return
                tmix_t = TMix(T=T, N=7)
                tmix = tmix_t()
                out[i] = tmix.AddConstRef(inp[i])

        else:

            @cuda.jit(link=shim_writer.links())
            def kernel(inp, out):
                i = cuda.grid(1)
                if i >= out.size:
                    return
                tmix_t = TMix(T=T, N=7)
                tmix = tmix_t()
                out_ptr = ffi.from_buffer(out[i:])
                tmix.AddConstRef(inp[i], out_ptr)

        kernel[1, 32](x, out)
        np.testing.assert_array_equal(out, x + 7)


def test_templated_method_array_ref_non_numeric_size(tmp_path):
    source = textwrap.dedent(
        """\
        #pragma once
        template <int N>
        __device__ void take_array(int (&arr)[N]) { (void)arr; }
        """
    )
    header_path = tmp_path / "array_ref_non_numeric.cuh"
    header_path.write_text(source, encoding="utf-8")

    decls = parse_declarations_from_source(
        str(header_path),
        [str(header_path)],
        "sm_80",
        verbose=False,
    )
    templs = [
        templ
        for templ in decls.function_templates
        if templ.function.name == "take_array"
    ]
    assert templs
    params = templs[0].function.params
    assert params[0].type_.unqualified_non_ref_type_name.endswith("[N]")

    formal_args_str, actual_args_str = _make_templated_method_shim_arg_strings(
        param_types_inner=(nbtypes.CPointer(nbtypes.int32),),
        cxx_params=params,
    )

    assert formal_args_str == ",int (&arg0)[N]"
    assert actual_args_str == "arg0"
