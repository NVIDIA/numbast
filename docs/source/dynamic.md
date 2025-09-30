# Dynamic binding generation

Dynamic generation produces bindings at runtime in the same environment where the CUDA C++ library is available. Bindings are inserted into Numba’s registries on the fly, so you can immediately use them.

## When to use
- You are exploring a CUDA C++ library and want rapid iteration.
- You control the runtime environment (development, prototyping, CI jobs).
- You don’t want a separate build/distribution step for bindings.

## Requirements
- CUDA Toolkit headers installed and discoverable.
- `clangdev` available (see `conda/environment.yaml`).
- Access to the CUDA C++ headers you want to bind.

## Example

Given a C++ declaration:

```c++
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
```

Numbast can generate and use bindings at runtime:

```python
import os
from ast_canopy import parse_declarations_from_source
from numbast import bind_cxx_struct, bind_cxx_function, MemoryShimWriter

from numba import types, cuda
from numba.core.datamodel.models import PrimitiveModel

import numpy as np

# Parse the header as AST and read all declarations
source = os.path.join(os.path.dirname(__file__), "demo.cuh")
# Choose a compute capability that matches your GPU
decls = parse_declarations_from_source(source, [source], "sm_80")

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
```

## Guidance and best practices
- Ensure the CUDA toolkit found at generation time is the same as the runtime toolkit.
- Match the compute capability (e.g., `sm_80`) to your target GPU.
- Keep header search paths consistent; custom include directories can be supplied to the parser if needed.

## Differences vs static generation
- No standalone Python module is produced; bindings live in memory.
- No distribution step; ideal for internal development and experiments.
- The environment must contain `numbast`, `ast_canopy`, and the CUDA headers.
