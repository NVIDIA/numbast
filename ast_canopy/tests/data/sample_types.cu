// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#include <stdint.h>

struct Foo {

  int a;
  const int b;

  int *c;
  int **d;

  int &e;

  const int *f;
  int *const g;
  const int *const h;

  const int *const *const i;
};

void __device__ bar(uint64_t a, int_fast32_t b, int32_t &c, uint8_t *d,
                    const int64_t *e, uint32_t *const f) {}
