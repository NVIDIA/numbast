# Numba Extensions

## Overview

This folder contains several Numba bindings for CUDA libraries created via `Numbast`.

Beware these bindings are built only to demonstrate the usage of Numbast and Canopy.
They are not official bindings and should not be used under production environments.

### Requirements

Numbast is required to run these extensions.

Tested under CUDA 12.3, driver 545.23.08.

## Contents

The following struct / function / operations are tested:

`nv_bfloat16` and `half`:
- Constructors from `float64` `float32`, `int16`, `int32`, `int64`, `uint16`, `uint32`, `uint64`, `float16`
- Arithmetic and Logical Operators: `+`, `-` (unary and binary), `*`, `/`, `+=`, `-=`, `*=`, `/=`, `==`, `!=`, `>=`, `<=`, `>`, `<`
- Native casts to `float32`, `int16`, `int32`, `int64`, `uint16`, `uint32`, `uint64`
- Math functions: `htrunc`, `hceil`, `hfloor`, `hrint`, `hsqrt`, `hrsqrt`, `hrcp`, `hlog`, `hlog2`, `hlog10`, `hcos`, `hsin`, `hexp`, `hexp2`, `hexp10`

`nv_bfloat162` and `half2`:
- Constructors
- Factory function `make_bfloat162`, `make_half2`
- Arithmetic and Logical Operators: `+`, `-` (unary and binary), `*`, `/`, `+=`, `-=`, `*=`, `/=`, `==`, `!=`, `>=`, `<=`, `>`, `<`
- Math functions: `htrunc`, `hceil`, `hfloor`, `hrint`, `hsqrt`, `hrsqrt`, `hrcp`, `hlog`, `hlog2`, `hlog10`, `hcos`, `hsin`, `hexp`, `hexp2`, `hexp10`
- Member access: `.x`, `.y`

Known Limitations:
- Native cast from `nv_bfloat16` to `half` is not yet supported. In C++ this is supported through a `half` constructor with `nv_bfloat16` argument. The definition is in `cuda_bf16.hpp` as a standalone definition, which is currently *ignored* by `canopy`.

`cuRAND` device APIs
`curand_device` package contains the device APIs from cuRAND, and `curand_host` contains a small set of host APIs necessary to make the (tested) device APIs work, including:
- Random integer generation, supported generators
    - XORWOW
    - MRG32k3a
    - Philox4_32_10
- Continuous distribution of uniform and normal, supported generators
    - XORWOW
    - MRG32k3a
    - Philox4_32_10
    - Sobol
    - SobolScramble
- Poisson sampling, supported generators
    - XORWOW

## Packaging

`numba_extensions` package is created to demonstrate the remote deployment of Numbast and Canopy.
It includes:
- bfloat16 feature

## Numba Extension Entrance

The extensions for `nv_bfloat16` type is created via `bf16_bindings.py`. By default it looks for `/usr/local/cuda` for CUDA headers and generate ASTs with `cuda_bf16.h`.

## Running the tests

The tests are organized under `tests` folder. And are run by `pytest` framework. Run the tests via `pytest .`
