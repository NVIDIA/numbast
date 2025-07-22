# AST_Canopy package

Canopy [ka·nuh·pee]

(Forest Ecology) The upper layer or habitat zone, formed by mature tree crowns and including other biological organisms.

![canopy](static/canopy.png)

## Overview

AST_Canopy is a tool that seiralizes the top level declarations from the abstract syntax tree using `clangTooling`.

Additionally, this package can maintain translation rules from C++ semantics to python semantics (such as mapping `operator +(const foo &lhs, const foo &rhs)` to `operator.add`, as well as providing functions to separate a arithmetic operator from a conversion operator). *It is a non-goal for this package to build a python binding for clangTooling.*

## Feature:

The main API of ast_canopy is:
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

## Building from source

To build the package, first create a conda environment according to steps outlined in [Get Started](../README.md#get-started).
Then run:

```bash
build.sh
```

To autogenerate python interface file (pyi) for `pylibastcanopy` (for better typing support in python):

```bash
stubgen -m pylibastcanopy -o ast_canopy
```

### Custom Installation Path

By default, `libastcanopy` is installed to the system's default CMake install prefix. You can override this behavior using the `ASTCANOPY_INSTALL_PATH` environment variable. This is useful when building in environments where you don't have write access to system directories, or when you want to install to a specific location.

#### Using environment variable:
```bash
ASTCANOPY_INSTALL_PATH=/path/to/custom/install ./build.sh
```

**Note:** When using a custom install path, make sure that `LD_LIBRARY_PATH` is set so that pylibastcanopy can locate the library.

### Build System

The build system depends on [cmake](https://cmake.org/) and [scikit-build-core](https://scikit-build-core.readthedocs.io/en/latest/index.html) and optionally [ninja](https://ninja-build.org/). In total, 3 libraries (packages) are built:

- libastcanopy: the library that traverses the clang AST.
- pylibastcanopy: the python bindings to the `libastcanopy` library, built with `pybind11`.
- ast_canopy: the python API using the `pylibastcanopy` bindings.

`libastcanopy` is built independent from `pylibastcanopy` and `ast_canopy`. CMake is used to manage its build and install process. We first compile `libastcanopy` into a shared library and installed to proper location. Depending on whether you use a conda environment or not, developers may need to set `CMAKE_INSTALL_PREFIX` accordingly. A sample cmake build and install command can be found at [build.sh](./build.sh).

`pylibastcanopy` and `ast_canopy` are built with `scikit-build-core`. It depends on `libastcanopy` via dynamic linking. `scikit-build-core`'s cmake subsystem discovers libastcanopy via `find_package`. This depends on libastcanopy is built, installed and exported on target system (see [libastcanopy build config](cpp/CMakeLists.txt)). `scikit-build-core` then takes care of the build of `pylibastcanopy` and `ast_canopy` package.

## Run the tests

The bindings are tested via `pytest`. To run the tests, execute
```
pytest .
```

## Folder Structure

- cpp: the c++ portion of the project, depends on libclang and clang C-API. We dubbed this as `libastcanopy`.
- ast_canopy: the python package that includes APIs and the bindings to the cpp code. It contains two targets to build `pylibastcanopy` and `ast_canopy`. See [*Build System*](#build-system) section.
- tests: containing tests relating to the package
