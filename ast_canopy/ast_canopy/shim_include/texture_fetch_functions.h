#pragma once

// This shim ensures Clang's __clang_cuda_runtime_wrapper.h only includes
// NVIDIA's texture_fetch_functions.h for CUDA < 13. For CUDA >= 13, the
// wrapper should not include it.
//
// We use include_next to forward to the vendor header that follows this
// directory on the include search path.

#if defined(__CUDA_VERSION__)
#if __CUDA_VERSION__ < 13000
#include_next <texture_fetch_functions.h>
#endif
#elif defined(CUDA_VERSION)
#if CUDA_VERSION < 13000
#include_next <texture_fetch_functions.h>
#endif
#endif
