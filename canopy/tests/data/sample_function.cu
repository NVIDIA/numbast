int __device__ add(int a, int b) { return a + b; }

int __device__ mul(const int &a, int const *b) { return a * (*b); }

int __device__ add2(int &&a, int &&b) { return a + b; }

/* Set up function decorations */
#if defined(__CUDACC__)
#define __CUDA_HOSTDEVICE__ __host__ __device__
#else
#define __CUDA_HOSTDEVICE__
#endif /* defined(__CUDACC_) */

int __CUDA_HOSTDEVICE__ add_hostdevice(int a, int b) { return a + b; }

// Template functions are not parsed in the "function" list,
// rather it is parsed in the "function template" list
template <typename T> __CUDA_HOSTDEVICE__ T add_template(T a, T b) {
  return a + b;
}
