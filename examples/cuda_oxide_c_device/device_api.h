// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#include <type_traits>

extern "C" {
#endif

typedef enum round1_mode {
  ROUND1_MODE_ADD = 0,
  ROUND1_MODE_SUBTRACT = 1,
} round1_mode;

struct round1_options {
  uint32_t scale;
  int32_t bias;
  char reserved[8];
};

#ifdef __cplusplus
static_assert(sizeof(round1_mode) == 4);
static_assert(
    std::is_same_v<std::underlying_type_t<round1_mode>, unsigned int>);
static_assert(sizeof(round1_options) == 16);
static_assert(alignof(round1_options) == 4);
#endif

__device__ int32_t round1_add(int32_t left, int32_t right);
__device__ int32_t round1_accumulate(const int32_t *values, size_t count);
__device__ int32_t round1_apply(int32_t value, enum round1_mode mode,
                                const struct round1_options *options);

#ifdef __cplusplus
}
#endif
