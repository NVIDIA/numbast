#pragma once

// Temporary CUDA 13.2 shim for clang-20 parsing of
// crt/math_functions.hpp (rsqrt/rsqrtf declarations).
//
// Only apply on CUDA 13.2, and only when the specifier macro is still
// undefined in this include context.
#if defined(CUDA_VERSION) && (CUDA_VERSION >= 13020) && (CUDA_VERSION < 13030)
#if !defined(_NV_RSQRT_SPECIFIER)

// Mirror the vendor definition from crt/math_functions.h.
#if defined(__GNUC__) && !defined(__ANDROID__) && !defined(__QNX__) &&         \
    !defined(__APPLE__) && !defined(__HORIZON__)
#include <features.h>
#if __GLIBC_PREREQ(2, 42)
#define _NV_RSQRT_SPECIFIER noexcept(true)
#else
#define _NV_RSQRT_SPECIFIER
#endif
#else
#define _NV_RSQRT_SPECIFIER
#endif

#endif
#endif

#include_next <crt/math_functions.hpp>
