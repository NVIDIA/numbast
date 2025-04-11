# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from tempfile import NamedTemporaryFile

import pytest

from numba import cuda

from ast_canopy import parse_declarations_from_source
from numbast import bind_cxx_functions, MemoryShimWriter

function_template = """
__device__ int {name}(int a, int b) {{
    return a + b;
}}
"""


@pytest.fixture(params=[1, 10, 100, 1000])
def simulate_header(request):
    """Create a header file that contains N headers to add two integers together.
    Used to benchmark the impact of number of headers to kernel launch.
    """
    N = request.param

    tmp = NamedTemporaryFile(mode="w", suffix=".cuh", delete=False)
    functions = [function_template.format(name=f"add{i}") for i in range(N)]
    tmp.write("\n".join(functions))
    tmp.flush()

    major, minor = cuda.get_current_device().compute_capability
    decls = parse_declarations_from_source(
        tmp.name, [tmp.name], f"sm_{major}{minor}"
    )
    shim_writer = MemoryShimWriter(f'#include "{tmp.name}"')
    adds = bind_cxx_functions(shim_writer, decls.functions)

    yield adds, shim_writer


def test_rtc(benchmark, simulate_header):
    def bench():
        adds, shim_writer = simulate_header
        add = adds[0]

        @cuda.jit(link=shim_writer.links())
        def kernel():
            _ = add(1, 2)

        kernel[1, 1]()

    benchmark(bench)
