import os
from ast_canopy import parse_declarations_from_source
from numbast import bind_cxx_struct, bind_cxx_function, MemoryShimWriter

from numba import types, cuda
from numba.core.datamodel.models import PrimitiveModel

import numpy as np

# Use `AST_Canopy` to parse demo.cuh as AST, read all declarations from it.
source = os.path.join(os.path.dirname(__file__), "demo.cuh")
# Assume your machine has a GPU that supports "sm_80" compute capability,
# parse the header with sm_80 compute capability.
decls = parse_declarations_from_source(source, [source], "sm_80")
structs, functions = decls.structs, decls.functions

shim_writer = MemoryShimWriter(f'#include "{source}"')

# Make Numba bindings from the declarations.
# New type "myfloat16" is a Number type, data model is PrimitiveModel.
myfloat16 = bind_cxx_struct(
    shim_writer, structs[0], types.Number, PrimitiveModel
)
bind_cxx_function(shim_writer, functions[0])
hsqrt = bind_cxx_function(shim_writer, functions[1])


@cuda.jit(link=shim_writer.links())
def kernel(arr):
    one = myfloat16(1.0)
    two = myfloat16(2.0)
    three = one + two
    sqrt3 = hsqrt(three)
    arr[0] = types.float64(three)
    arr[1] = types.float64(sqrt3)


arr = np.array([0.0, 0.0], dtype=np.float64)
kernel[1, 1](arr)

np.testing.assert_allclose(arr, [3.0, np.sqrt(3.0)], rtol=1e-2)
