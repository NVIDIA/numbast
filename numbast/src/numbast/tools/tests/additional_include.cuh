#include "bar.cuh"

void __device__ set42(int * arr) { *arr = bar(42); }