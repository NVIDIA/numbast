#include <cstdlib>
#include <iostream>
#include <string>

#include <cuda_bf16.h>

__global__ void simple_kernel(float *arith) {
  // Binary Arithmetic Operators
  nv_bfloat16 a = nv_bfloat16(1.0f);
  nv_bfloat16 b = nv_bfloat16(2.0f);
  arith[0] = float(a + b);
  arith[1] = float(a - b);
  arith[2] = float(a * b);
  arith[3] = float(a / b);
}

int main(void) {
  char *repetition_char = std::getenv("NUMBAST_BENCH_KERN_REPETITION");
  if (repetition_char == nullptr)
    std::cout << "Unable to retrieve NUMBAST_BENCH_KERN_REPETITION environment "
                 "variable in `gold`. Assume repetition 1000."
              << std::endl;
  int repetition =
      repetition_char ? std::stoi(std::string(repetition_char)) : 1000;

  int N = 4;
  float *arith, *arith_d;
  arith = (float *)malloc(N * sizeof(float));

  cudaMalloc(&arith_d, N * sizeof(float));

  for (int i = 0; i < N; i++) {
    arith[i] = 0.0f;
  }
  cudaMemcpy(arith_d, arith, N * sizeof(float), cudaMemcpyHostToDevice);

  for (int i = 0; i < repetition; i++)
    simple_kernel<<<1, 1>>>(arith_d);

  cudaDeviceSynchronize();

  cudaMemcpy(arith, arith_d, N * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(arith_d);
  free(arith);

  return 0;
}
