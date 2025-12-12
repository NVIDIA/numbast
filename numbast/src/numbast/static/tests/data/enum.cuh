// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#include <cuda/std/cstdint>

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

// Test enum with different underlying integer types

enum Car : uint8_t { Sedan, SUV, Pickup, Hatchback };

enum Color : int16_t { Red, Green, Blue, Black = -1 };

size_t __device__ get_color_name(Color color, char *out) {
  switch (color) {
  case Color::Red:
    memcpy(out, "Red", 3);
    return 3;
  case Color::Green:
    memcpy(out, "Green", 5);
    return 5;
  case Color::Blue:
    memcpy(out, "Blue", 4);
    return 4;
  case Color::Black:
    memcpy(out, "Black", 5);
    return 5;
  default:
    return 0;
  }
}

size_t __device__ get_car_name(Car car, char *out) {
  switch (car) {
  case Car::Sedan:
    memcpy(out, "Sedan", 5);
    return 5;
  case Car::SUV:
    memcpy(out, "SUV", 3);
    return 3;
  case Car::Pickup:
    memcpy(out, "Pickup", 6);
    return 6;
  case Car::Hatchback:
    memcpy(out, "Hatchback", 9);
    return 9;
  default:
    return 0;
  }
}

void __device__ car_with_color(Car car, Color color, char *out) {
  char *color_buf = new char[32];
  char *car_buf = new char[32];

  size_t color_length = get_color_name(color, color_buf);
  size_t car_length = get_car_name(car, car_buf);

  memcpy(out, color_buf, color_length);
  out[color_length] = ' ';
  memcpy(out + color_length + 1, car_buf, car_length);
  out[color_length + car_length + 1] = '\0';

  delete[] color_buf;
  delete[] car_buf;
}
