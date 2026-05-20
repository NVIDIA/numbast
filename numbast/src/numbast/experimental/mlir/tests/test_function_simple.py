from numbast.experimental.mlir import bind_cxx_functions, MemoryShimWriter

import numpy as np

from numba_cuda_mlir import cuda as cs


def test_function_simple(decl_of):
    decls, path = decl_of("sample_function_simple.cuh")
    funcs = decls.functions
    shim_writer = MemoryShimWriter(f'#include "{path}"')
    func_bindings = bind_cxx_functions(
        shim_writer,
        funcs,
    )

    assert len(func_bindings) == 1
    add = func_bindings[0]

    @cs.jit(link=list(shim_writer.links()))
    def kernel(a, b, out):
        out[0] = add(a[0], b[0])

    a = cs.to_device(np.array([1], dtype=np.int32))
    b = cs.to_device(np.array([2], dtype=np.int32))
    out = cs.to_device(np.array([0], dtype=np.int32))
    kernel[1, 1](a, b, out)
    assert out.copy_to_host()[0] == 3
