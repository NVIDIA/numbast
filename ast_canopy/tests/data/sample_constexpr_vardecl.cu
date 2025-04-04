#include <cuda/std/cstdint>

constexpr int __device__ i4 = 42;
constexpr unsigned int __device__ u4 = 42;
constexpr long __device__ i8 = 42;
constexpr unsigned long __device__ u8 = 42;
constexpr long long __device__ i8b = 42;
constexpr unsigned long long __device__ u8b = 42;

constexpr int16_t __device__ i2 = 42;
constexpr int32_t __device__ i4b = 42;
constexpr int64_t __device__ i8c = 42;
constexpr uint16_t __device__ u2 = 42;
constexpr uint32_t __device__ u4b = 42;
constexpr uint64_t __device__ u8c = 42;

constexpr float __device__ f4 = 3.14;
constexpr double __device__ f8 = 3.14;

// non constexpr, do not show up
int __device__ bar = 43;
