#pragma once

template <typename T, int BLOCK_DIM_X> struct BlockScan {
  __device__ BlockScan() { printf("BlockScan constructor called\n"); }

  void __device__ InclusiveSum(T input, T *output) {
    printf("BlockScan InclusiveSum called\n");
  }
};
