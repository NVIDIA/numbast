// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

enum Fruit { Apple = 1, Banana = 3, Orange = 5 };

enum class Animal { Cat, Dog, Horse };

void __device__ feed(Animal animal, int *out) {
  switch (animal) {
  case Animal::Cat:
    out[0] = 1;
    break;
  case Animal::Dog:
    out[0] = 2;
    break;
  case Animal::Horse:
    out[0] = 3;
    break;
  default:
    break;
  }
}
