Dynamic binding generation
==========================

Dynamic generation produces bindings at runtime in the same environment where the CUDA C++ library is available.
Bindings are inserted into Numba's registries on the fly, so you can immediately use them.

When to use
-----------

- You are exploring a CUDA C++ library and want rapid iteration.
- You control the runtime environment (development, prototyping, CI jobs).
- You don't want a separate build/distribution step for bindings.

Requirements
------------

- CUDA Toolkit headers installed and discoverable.
- ``clangdev`` available in your development environment.
- Access to the CUDA C++ headers you want to bind.

Example
-------

Given a C++ declaration:

.. code-block:: c++

  // demo.cuh
  struct __attribute__((aligned(2))) __myfloat16
  {
  private:
    half data;

  public:
    __host__ __device__ __myfloat16();

    __host__ __device__ __myfloat16(double val);

    __host__ __device__ operator double() const;
  };

  __host__ __device__ __myfloat16 operator+(const __myfloat16 &lh, const __myfloat16 &rh);

  __device__ __myfloat16 hsqrt(const __myfloat16 a);

Numbast can generate and use bindings at runtime:

.. code-block:: python

  import os
  from ast_canopy import parse_declarations_from_source
  from numbast import bind_cxx_struct, bind_cxx_function, MemoryShimWriter

  from numba import types, cuda
  from numba.core.datamodel.models import PrimitiveModel

  from cuda.core import Device

  import numpy as np

  # Query the compute capability of current device
  dev = Device()
  cc = f"sm_{dev.compute_capability.major}{dev.compute_capability.minor}"

  # Parse the header as AST and read all declarations
  source = os.path.join(os.path.dirname(__file__), "demo.cuh")
  # Choose a compute capability that matches your GPU
  decls = parse_declarations_from_source(source, [source], cc)

  # Create a shim and generate bindings
  shim_writer = MemoryShimWriter(f'#include "{source}"')

  # Make Numba bindings from the declarations
  # New type "myfloat16" is a Number type, data model is PrimitiveModel.
  myfloat16 = bind_cxx_struct(shim_writer, decls.structs[0], types.Number, PrimitiveModel)
  bind_cxx_function(shim_writer, decls.functions[0])
  hsqrt = bind_cxx_function(shim_writer, decls.functions[1])

  # Use within Numba
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

Struct name overrides
---------------------

``bind_cxx_struct`` accepts a ``name=`` keyword for cases where the parsed record
name is not the C++ type spelling that should appear in generated shim code and
Numba's type registry. This is useful when binding a concrete class-template
specialization through the lower-level struct API.

.. code-block:: python

  matrix = next(
      s
      for s in decls.class_template_specializations
      if s.qual_name == "Eigen::Matrix<float, 3, 1>"
  )

  Matrix3f = bind_cxx_struct(
      shim_writer,
      matrix,
      name="Eigen::Matrix<float, 3, 1>",
  )

External C++ type mappings
--------------------------

Use ``numbast.types.register_cxx_type`` when a function or method signature
mentions a C++ type that ast_canopy did not parse, but you already know the
Numba type that should represent it.

The ``cxx_name`` argument should match the type string Numbast receives from
ast_canopy for a signature, usually ``Type.unqualified_non_ref_type_name`` from
a parameter or return type. For namespaced types this normally includes the
C++ namespace qualification, such as ``third_party::Handle``. It is not the
Python API name, and it is not necessarily a declaration's ``qual_name``.

When in doubt, inspect the parsed declaration before registering the mapping:

.. code-block:: python

  for param in decls.functions[0].params:
      print(param.type_.unqualified_non_ref_type_name)

Top-level ``const`` and reference qualifiers are stripped from this lookup key;
pointer types are resolved by looking up their pointee type.

.. code-block:: python

  from numba import types as nbtypes
  from numbast.types import register_cxx_type

  register_cxx_type("third_party::Handle", nbtypes.uint64)

Class template calls
--------------------

For class template constructor usage and examples, see :doc:`template`.

Guidance and best practices
---------------------------

- Ensure the CUDA toolkit found at generation time is the same as the runtime toolkit.
- Match the compute capability (e.g., ``sm_80``) to your target GPU. We recommend using ``cuda.core`` to discover the
  compute capability of the current device.
- Keep header search paths consistent; custom include directories can be supplied to the parser if needed.

Differences vs static generation
--------------------------------

- No standalone Python module is produced; bindings live in memory.
- No distribution step; ideal for internal development and experiments.
- The environment must contain ``numbast``, ``ast_canopy``, and the CUDA headers.
