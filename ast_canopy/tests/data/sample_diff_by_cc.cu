float __device__ constexpr fpi() { return 3.14f; }

struct Functor {

  float k;

  float __device__ operator()(float a, float b) { return a * k + b; }

#if (defined(__CUDACC__) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800)))

  double dk;

  double __device__ operator()(double a, double b) { return a * dk + b; }

#endif
};

#if (defined(__CUDACC__) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800)))

double __device__ constexpr dpi() { return 3.14; }

struct AdvancedFunctor {

  float k;

  float __device__ operator()(float a, float b) { return a * k + b * k; }
};

#endif
