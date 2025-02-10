// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

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
