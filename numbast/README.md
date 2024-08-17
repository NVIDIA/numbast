# Numbast Project

## Overview

Numbast is the auto binding generator for CUDA C++ headers. It consumes parsed headers from `AST_Canopy` and dynamically create new types, data models and lowering for C++ types. One could see Numbast as a C++-Python syntax translator.

## Supported CUDA C++ declarations

- Concrete Struct / Classes (e.g. `struct Foo`)
    - Constructors e.g. `Foo()`
    - Conversion Operators e.g. `operator float()`
    - Attribute Read Access e.g. `Foo().x`
- Concrete Functions `e.g. bar() {}`

### Requirement

- `ast_canopy`
- `numba`>=0.59
- `pynvjitlink`>=0.22

## Folder Structure

Numbast organizes its files in [src layout](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/).

- benchmarks: contain the benchmark for Numbast. Including compilation pipeline overhead measurements.
- src: contains Numbast module. Each file is organized according to the C++ feature that it's addressing. For example
`struct.py` handles the declaration of C++ structs.
- tests: contains the tests to each module in Numbast
