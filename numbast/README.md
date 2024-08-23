# Numbast Project

# Numbast - auto Numba binding generator

## Overview

Numbast is an auto binding generator for CUDA C++ headers. It consumes parsed headers from `AST_Canopy` and dynamically create new types, data models and lowering for C++ types. 

## Supported CUDA C++ declarations

Currently, the following C++ features are recognized and generated.

- Concrete Struct / Classes (e.g. `struct Foo`)
    - Constructors e.g. `Foo()`
    - Conversion Operators e.g. `operator float()`
    - Attribute Read Access e.g. `Foo().x`
- Concrete Functions `e.g. bar() {}`

### Requirement

- `ast_canopy>=0.1.0`
- `numba`>=0.59
- `pynvjitlink`>=0.22

## Folder Structure

Numbast organizes its files in [src layout](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/).

- benchmarks: contain the benchmark for Numbast. Including compilation pipeline overhead measurements.
- src: contains Numbast module. Each file is organized according to the C++ feature that it's addressing. For example
`struct.py` handles the C++ structs declarations.
- tests: contains the tests to each module in Numbast
