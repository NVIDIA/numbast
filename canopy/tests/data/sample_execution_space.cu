void __device__ dfoo() {}
void __host__ hfoo() {}
void __global__ gfoo() {}
void __device__ __host__ dhfoo() {}
void foo() {}

struct Bar {
  void __device__ dfoo() {}
  void __host__ hfoo() {}
  void __device__ __host__ dhfoo() {}
  void foo() {}
};
