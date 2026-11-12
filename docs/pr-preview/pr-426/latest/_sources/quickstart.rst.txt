Quickstart
==========

This quickstart shows how to parse CUDA C++ declarations and generate Numba-compatible bindings with Numbast.

Prerequisites
-------------

- A working CUDA Toolkit installation
- A Python environment with ``numba``, ``numba-cuda``, and ``numbast``
- For parsing headers, a ``clang``-based toolchain (see :doc:`install`)

Example: parse and bind CUDA C++ declarations
---------------------------------------------

Given a CUDA C++ struct and functions (simplified):

.. code-block:: cuda

   struct __attribute__((aligned(2))) __myfloat16 {
   public:
       half data;
       __host__ __device__ __myfloat16();
       __host__ __device__ __myfloat16(double val);
       __host__ __device__ operator double() const;
   };

   __host__ __device__ __myfloat16 operator+(const __myfloat16 &lh, const __myfloat16 &rh);
   __device__ __myfloat16 hsqrt(const __myfloat16 a);

You can parse the declarations and generate bindings:

.. code-block:: python

   import os
   from ast_canopy import parse_declarations_from_source
   from numbast import bind_cxx_struct, bind_cxx_function, MemoryShimWriter

   from numba import types, cuda
   from numba.core.datamodel.models import PrimitiveModel
   import numpy as np

   # Use AST Canopy to parse a header into declarations
   source = os.path.join(os.path.dirname(__file__), "demo.cuh")
   decls = parse_declarations_from_source(source, [source], "sm_80")

   shim_writer = MemoryShimWriter(f'#include "{source}"')

   # Create bindings from the declarations
   myfloat16 = bind_cxx_struct(shim_writer, decls.structs[0], types.Number, PrimitiveModel)
   bind_cxx_function(shim_writer, decls.functions[0])
   hsqrt = bind_cxx_function(shim_writer, decls.functions[1])

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

Next steps
----------

- See :doc:`static` for generating distributable modules without requiring Numbast at runtime.
- See :doc:`dynamic` for runtime generation in exploratory workflows.
- See :doc:`supported_declarations` for details of supported C++ constructs.
