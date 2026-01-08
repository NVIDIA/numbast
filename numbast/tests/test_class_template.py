# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np

from numba import cuda

import cffi


from ast_canopy import parse_declarations_from_source
from numbast import bind_cxx_class_templates, MemoryShimWriter


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


class TestTemplatedClassTemplatedMethodUnitStyle:
    """
    Test plan (unit-test-ish, not a full CCCL/CUB e2e):

    Ultimately we want an end-to-end *binding-generation* story for a
    templated-class + templated-method system. These tests focus on the
    specialization/deduction/default-arg surface area.

    Covered scenarios:
    - Fully specialized:
      class template args provided; method template args fully provided.
    - Partially specialized but deducible:
      some method template args explicitly provided (e.g. a non-type), while
      the rest (e.g. a type param) is deduced from call argument types.
    - Not fully specialized:
      method template args are neither provided nor deducible, and no defaults
      exist; should error.
    - Not fully specialized but defaults exist:
      method template args omitted but method template has defaults, so the
      call is still well-formed.

    Notes about current state:
    - The templated-method plumbing exists, but there's no user-facing API for
      specifying method template parameters yet. Those tests are xfail to
      document the gap without implementing it.
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
