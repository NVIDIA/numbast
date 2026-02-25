# Numbast Extensions

## Overview

This folder contains several Numba bindings for CUDA libraries created via `Numbast`.

Beware these bindings are built only to demonstrate the usage of Numbast and Canopy.
They are not official bindings and should not be used under production environments.

### Requirements

Numbast is required to run these extensions.

Minimum required CUDAToolkit Version: 12.5

## Contents

The following struct / function / operations are tested:

`nv_bfloat16`:
- Constructors from `float64` `float32`, `int16`, `int32`, `int64`, `uint16`, `uint32`, `uint64`, `float16`
- Arithmetic and Logical Operators: `+`, `-` (unary and binary), `*`, `/`, `+=`, `-=`, `*=`, `/=`, `==`, `!=`, `>=`, `<=`, `>`, `<`
- Native casts to `float32`, `int16`, `int32`, `int64`, `uint16`, `uint32`, `uint64`
- Math functions: `htrunc`, `hceil`, `hfloor`, `hrint`, `hsqrt`, `hrsqrt`, `hrcp`, `hlog`, `hlog2`, `hlog10`, `hcos`, `hsin`, `hexp`, `hexp2`, `hexp10`

`nv_bfloat162`:
- Constructors
- Factory function `make_bfloat162`
- Arithmetic and Logical Operators: `+`, `-` (unary and binary), `*`, `/`, `+=`, `-=`, `*=`, `/=`, `==`, `!=`, `>=`, `<=`, `>`, `<`
- Math functions: `htrunc`, `hceil`, `hfloor`, `hrint`, `hsqrt`, `hrsqrt`, `hrcp`, `hlog`, `hlog2`, `hlog10`, `hcos`, `hsin`, `hexp`, `hexp2`, `hexp10`
- Member access: `.x`, `.y`

`fp16` and `cuRAND` extension bindings were removed from this repository.
Use `numba-cuda` for fp16 support and `nvmath` for cuRAND APIs.

## Packaging

`numbast_extensions` package is created to demonstrate the remote deployment of Numbast and Canopy.
It includes:
- bfloat16 feature

## Numba Extension Entrance

The extensions for `nv_bfloat16` type is created via `bf16_bindings.py`. By default it looks for `/usr/local/cuda` for CUDA headers and generate ASTs with `cuda_bf16.h`.

## Running the tests

The tests are organized under `tests` folder. And are run by `pytest` framework. Run the tests via `pytest .`
