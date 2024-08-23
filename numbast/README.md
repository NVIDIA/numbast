# Numbast - AST based Numba Binding Generator

## Overview

Numbast is an auto binding generator for CUDA C++ headers. It consumes parsed headers from `AST_Canopy` and dynamically create new types, data models and lowering for C++ types.

## Supported CUDA C++ declarations

Currently, the following C++ features are recognized and generated. We use a `__myfloat16` struct declaration in C++ below to demonstrate Numbast features.

```C++
// demo.cuh
struct __attribute__((aligned(2))) __myfloat16
{
public:
 half data;

 __host__ __device__ __myfloat16();

 __host__ __device__ __myfloat16(double val);

 __host__ __device__ operator float() const;
};
__host__ __device__ __myfloat16 operator+(const __myfloat16 &lh, const __myfloat16 &rh);

__device__ __myfloat16 hsqrt(const __myfloat16 a);
```

### Concrete Struct / Classes (e.g. `struct __myfloat16`)
|Kind|C++ Declaration|C++ Usage|Numba Usage|Explanation|
|---	|---	|---	|---	|---	|
|Concrete Struct Constructor|`__myfloat16(double val)`|`auto f = __myfloat16(3.14)`|`f = __myfloat16(3.14)`|A new Numba type for `Foo` is created; Struct data model for type `Foo` is created; Typings and lowerings for `Foo` constructors are also generated.|
|Concrete Struct Conversion Operator|`operator float() {}`|`float x = float(f)`|`x = float(f)`|Conversion operators defined for `Foo` struct are generated.|
|Concrete Struct Attribute|`half data;`|`auto data = f.data`|`data = f.data`|Public attributes for structs are accessible. Note: only read access is supported.|

### Concrete Functions (e.g. `__myfloat16 hsqrt(__myfloat16 a)` or `__myfloat16 operator+(const __myfloat16&, const __myfloat16&)`)
|Kind|C++ Declaration|C++ Usage|Numba|Explanation|
|---	|---	|---	|---	|---	|
|Concrete functions|`__myfloat16 hsqrt(__myfloat16 a)`|`auto res = hsqrt(f)`|`res = hsqrt(f)`|Function (for all overloads) typing and lowering are generated.|
|Operator overloads|``__myfloat16 operator+(const __myfloat16&, const __myfloat16&)``|`auto twof = f + f`|`twof = f + f`|Operators (for all overloads) typing and lowering are generated. Operators from C++ are mapped to its corresponding operation in Python, e.g. (`operator +` mapped to `operator.add` or `operator.pos` based on number of arguments.)|


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
