// SPDX-License-Identifier: Apache-2.0

#include "device_api.h"

extern "C" __device__ int32_t round1_add(int32_t left, int32_t right) {
  return left + right;
}

extern "C" __device__ int32_t round1_accumulate(const int32_t *values,
                                                size_t count) {
  int32_t total = 0;
  for (size_t index = 0; index < count; ++index) {
    total += values[index];
  }
  return total;
}

extern "C" __device__ int32_t
round1_apply(int32_t value, enum round1_mode mode,
             const struct round1_options *options) {
  const int32_t scaled = value * (int32_t)options->scale;
  return mode == ROUND1_MODE_ADD ? scaled + options->bias
                                 : scaled - options->bias;
}
