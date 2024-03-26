# Numbast

## Overview
Numbast = Numba + AST (Abstract Syntax Tree)

Numbast's mission is to establish an automated pipeline that converts CUDA APIs into Numba bindings. On a high level, top-level declarations are read from CUDA C++ header files, serailized as string and passed to python APIs. Numba binding generators then iterate through these bindings and make Numba extensions for each of the APIs.

There are several subcomponents: AST_Canopy, Numbast and a set of Numba Extensions

- [ast_canopy](ast_canopy/README.md): CUDA Header Parser that depends on clangTooling
- [numbast](numbast/README.md): Numba Binding Generator
- [numba_extensions](numba_extensions/README.md): a set of Numba bindings for several CUDA libraries using Numbast

## Get Started

First, install conda environment and activate:

```bash
conda env create -f conda/environment.yaml && \
  conda activate numbast
```

Next, install all subcomponents:

```bash
pip install -e ast_canopy \
  numbast \
  numba_extensions/bf16 \
  numba_extensions/fp16 \
  numba_extensions/curand_device \
  numba_extensions/curand_host
```

Validate the installation by running the tests:

```bash
pytest canopy/ numba_extensions/
```

### Example

Given a C++ struct (or function) declaration:
```c++
// demo.cuh
struct Foo
{
    __device__ Foo(int x) : x(x) {}
    int __device__ get() { return x; }

    int x;
};
```

Numbast can convert it into Numba bindings:

```python
# bindings.py
import os
from ast_canopy import parse_declarations_from_source
from numbast import bind_cxx_struct, ShimWriter

from numba import types
from numba.core.datamodel.models import StructModel

# Use `Canopy` to parse demo.cuh as AST, read all declarations from it.
source = os.path.join(os.path.dirname(__file__), "demo.cuh")
# Assume your machine has a GPU that supports "sm_80" compute capability,
# parse the header with sm_80 compute capability.
structs, * = parse_declarations_from_source(ast, [source], "sm_80")

shim_writer = ShimWriter("shim.cu", f'#include "{source}"')

# Make Numba bindings from the declarations.
# New type "Foo" is a generic Numba type, data model is StructModel.
Foo = bind_cxx_struct(shim_writer, structs[0], types.Type, StructModel)
```

`Foo` struct can now be used within Numba:

```python
# test.py
from bindings import Foo
from numba import cuda


@cuda.jit(link=["shim.cu"])
def kernel(arr):
    a = Foo(42)
    b = Foo(43)
    arr[0] = a.get() + b.get() # Or simply a.x + b.x


arr = cuda.device_array(1, dtype=int)
kernel[1, 1](arr)
assert arr[0] == 85
```

## Contribution Guidelines
See [CONTRIBUTING.md](./CONTRIBUTING.md)

## Community
Discussions are welcome! If you spotted bugs / have idea for new features, please submit at the issue board.

## References
The project depends on [Clang](https://github.com/llvm/llvm-project) and [Numba](https://numba.readthedocs.io/en/stable/)

## License
This project is licensed under the Apache 2.0 License - see the LICENSE.md file for details

## Key Visual

The numbat (Myrmecobius fasciatus) is a small, endangered marsupial native to Western Australia.

![Australian Numbat](./static/numbat.png)
