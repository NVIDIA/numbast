# Numbast

Numbast generates Python bindings for CUDA C++ declarations so they can be used
from Python in Numba CUDA kernels. It is intended for CUDA Python projects that
need to expose existing C++ device APIs without hand-writing Numba typing,
lowering, and shim code for every function or type.

Numbast works with
[ast_canopy](https://pypi.org/project/ast-canopy/), which parses CUDA C++
headers with Clang and serializes declaration metadata. The `numbast` package
consumes those declarations and registers or emits Numba CUDA-compatible
bindings.

## What It Can Bind

Numbast supports a focused subset of CUDA C++ declarations:

- Free functions, overload sets, and supported operator overloads.
- Structs and classes, including constructors, public fields, methods, and
  conversion operators.
- Enums exposed as Python `IntEnum` values with Numba-compatible typing.
- Function templates and class templates where concrete specializations can be
  materialized for Numba CUDA.
- External C++ type mappings when a signature mentions a type that should map to
  an already-known Numba type.

See the
[supported declarations](https://nvidia.github.io/numbast/latest/supported_declarations.html)
page for the full support matrix and current limitations.

## Binding Workflows

Numbast supports two complementary workflows.

### Dynamic Binding Generation

Dynamic generation parses headers and registers bindings at runtime. Use it for
experimentation, development, and environments where Numbast, ast_canopy, Clang
tooling, and the CUDA headers are all available.

```python
from ast_canopy import parse_declarations_from_source
from numba import cuda, types
from numba.core.datamodel.models import PrimitiveModel
from numbast import MemoryShimWriter, bind_cxx_function, bind_cxx_struct

decls = parse_declarations_from_source("demo.cuh", ["demo.cuh"], "sm_80")
shim_writer = MemoryShimWriter('#include "demo.cuh"')

MyType = bind_cxx_struct(shim_writer, decls.structs[0], types.Number, PrimitiveModel)
my_function = bind_cxx_function(shim_writer, decls.functions[0])


@cuda.jit(link=shim_writer.links())
def kernel(out):
    value = MyType(1.0)
    out[0] = my_function(value)
```

### Static Binding Generation

Static generation writes a standalone Python module that can be packaged with a
project. The generated module depends on Numba CUDA, but end users do not need
Numbast or Clang tooling at runtime.

```bash
python -m numbast --cfg-path config.yml --output-dir ./generated
```

## Installation

```bash
pip install numbast
```

Generating bindings requires a CUDA Toolkit and Clang tooling environment that
can parse the target headers. Running generated bindings requires a compatible
Numba CUDA runtime environment.

## Documentation

- [Overview](https://nvidia.github.io/numbast/latest/overview.html)
- [Installation](https://nvidia.github.io/numbast/latest/install.html)
- [Quickstart](https://nvidia.github.io/numbast/latest/quickstart.html)
- [Dynamic binding generation](https://nvidia.github.io/numbast/latest/dynamic.html)
- [Static binding generation](https://nvidia.github.io/numbast/latest/static.html)
- [Supported declarations](https://nvidia.github.io/numbast/latest/supported_declarations.html)
- [API reference](https://nvidia.github.io/numbast/latest/api_reference.html)

## Project Links

- Source: <https://github.com/NVIDIA/numbast>
- Documentation: <https://nvidia.github.io/numbast/latest/>
- Issues: <https://github.com/NVIDIA/numbast/issues>
