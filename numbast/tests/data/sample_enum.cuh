// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

enum Fruit { Apple = 1, Banana = 3, Orange = 5 };

void __device__ eat(Fruit fruit, int *out) {
  switch (fruit) {
  case Fruit::Apple:
    out[0] = 1;
    break;
  case Fruit::Banana:
    out[0] = 2;
    break;
  case Fruit::Orange:
    out[0] = 3;
    break;
  }
}
