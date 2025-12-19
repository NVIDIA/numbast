// The below is patching CUDA 13.0 headers for clang CUDA mode.
#if defined(__clang__) && defined(__CUDA__) && defined(__CUDA_ARCH__)

// NVCC 13 internally defines these to silidently deprecate certain functions.
#define __NV_SILENCE_DEPRECATION_BEGIN
#define __NV_SILENCE_DEPRECATION_END

// CUDA 13.0 internally defines new aligned data types for double4.
struct alignas(16) double4_16a {
  double x, y, z, w;
};
struct alignas(32) double4_32a {
  double x, y, z, w;
};

#endif
