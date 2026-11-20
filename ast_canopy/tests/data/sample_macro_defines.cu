#include "sample_macro_defines_include.cuh"

#define foo 123
#define FLAG
#define MAKE_VALUE(x) ((x) + 1)

__device__ int use_macro_define() { return foo; }
