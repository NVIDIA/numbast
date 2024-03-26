# AST_Canopy package

Canopy [ka·nuh·pee]

(Forest Ecology) The upper layer or habitat zone, formed by mature tree crowns and including other biological organisms.

![canopy](static/canopy.png)

## Overview

AST_Canopy is a tool that seiralizes the top level declarations from the abstract syntax tree using `clangTooling`.

Additionally, this pacakge can maintain translation rules from C++ semantics to python semantics (such as mapping `operator +(const foo &lhs, const foo &rhs)` to `operator.add`, as well as providing functions to separate a arithmetic opertor from a conversion operator). *It is a non-goal for this package to build a python binding for clangTooling.*

## Main APIs:

The main APIs are:
```python
import ast_canopy
ast_canopy.parse_declarations_from_source(
    source_file_path,
    files_to_retain,
    compute_capability,
    cccl_root,
    cudatoolkit_include_dir,
    cxx_standard)
```

`parse_declarations_from_source` parses the given CUDA header file and returns python objects containing the information of the declaration. `source_file_path` is the path to the source file, `files_to_retain` are list of source files to keep. Since the original AST file could contain thousands of files from the entire CUDA codebase, this function allows developer to focus on the headers that we want to look at. `compute_capability` is the compute capability to parse the CUDA header with (such as "sm_80"). `cudatoolkit_include_dir` specifies the path to the `include/` directory of the CUDAToolkit on the machine. `cxx_standard` is the c++ standard to parse the header with, defaults to `c++17`.

## Project Structure

- cpp: the c++ portion of the project, depends on libclang and clang C-API.
- ast_canopy: the python package that includes APIs and the bindings to the cpp code.

To build the package:

```python
pip install .
```

To autogenerate python interface file (pyi) for `pylibastcanopy`:

```bash
stubgen -m pylibastcanopy -o ast_canopy
```

### Build System

The package is built with [scikit-build-core](https://scikit-build-core.readthedocs.io/en/latest/index.html). In total, 3 libraries (packages) are built:

- libcanopy: the library that traverses the clang AST.
- pylibastcanopy: the python bindings to the `canopy` library, built with `pybind11`.
- ast_canopy: the python API using the `pylibastcanopy` bindings.

## Run the tests

The bindings are tested via `pytest`. To run the tests, execute
```
pytest .
```
