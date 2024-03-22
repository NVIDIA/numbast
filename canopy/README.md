# Canopy package

Canopy [ka·nuh·pee]

(Forest Ecology) The upper layer or habitat zone, formed by mature tree crowns and including other biological organisms.

![canopy](static/canopy.png)

## Overview

Canopy is a tool that seiralizes the top level declarations from the abstract syntax tree using `clangTooling`.

Additionally, this pacakge can maintain translation rules from C++ semantics to python semantics (such as mapping `operator +(const foo &lhs, const foo &rhs)` to `operator.add`, as well as providing functions to separate a arithmetic opertor from a conversion operator). *It is a non-goal for this package to build a python binding for clangTooling.*

## Main APIs:

The main APIs are:
```python
import canopy
canopy.make_ast_from_source(source_file_path, compute_capability, cudatoolkit_include_dir, cxx_standard)
canopy.parse_declarations_from_ast(ast_file_path, files_to_retain)
```

`make_ast_from_source` creates a clang AST file from the given CUDA header file and returns a path to the file. `source_file_path` is the path to the source file, `compute_capability` is the compute capability to parse the CUDA header with (such as "sm_80"). `cudatoolkit_include_dir` specifies the path to the `include/` directory of the CUDAToolkit on the machine. `cxx_standard` is the c++ standard to parse the header with, defaults to `c++17`.

`parse_declaration_from_ast` parses the generated AST file using libclang C-API. `ast_file_path` is the path to the AST file generated in step 1, whereas `files_to_retain` are list of *source files* (not the ast file) to filter out from the AST. Since the original AST file could contain thousands of files from the entire CUDA codebase, this function allows us to focus on the headers that we want to look at.

## Project Structure

- cpp: the c++ portion of the project, depends on libclang and clang C-API.
- canopy: the python package that includes APIs and the bindings to the cpp code.

To build the package:

```python
pip install .
```

To autogenerate python interface file (pyi) for `pylibcanopy`:

```bash
stubgen -m pylibcanopy -o canopy
```

### Build System

The package is built with [scikit-build-core](https://scikit-build-core.readthedocs.io/en/latest/index.html). In total, 3 libraries (packages) are built:

- libcanopy: the library that traverses the clang AST.
- pylibcanopy: the python bindings to the `canopy` library, built with `pybind11`.
- canopy: the python API using the `pylibcanopy` bindings.

## Run the tests

The bindings are tested via `pytest`. To run the tests, execute
```
pytest .
```
