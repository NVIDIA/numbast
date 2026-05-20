// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#include <cuda/std/cstdint>

enum Fruit { Apple = 1, Banana = 3, Orange = 5 };

enum class Animal { Cat, Dog, Horse };

/**
 * @brief Map an animal enum to an integer code.
 *
 * @param animal The animal to encode.
 * @param[out] out Output pointer where the encoded value is written to
 * `out[0]`.
 */
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

/**
 * @brief Convert a `Color` enum value to its ASCII name.
 *
 * @param color The color to convert.
 * @param[out] out Output buffer to receive the name (not null-terminated).
 * @return The number of bytes written to `out`. Returns 0 for unknown values.
 *
 * @note `out` must have space for at least 5 bytes (e.g., "Green"/"Black").
 */
size_t __device__ __inline__ get_color_name(Color color, char *out) {
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

/**
 * @brief Convert a `Car` enum value to its ASCII name.
 *
 * @param car The car to convert.
 * @param[out] out Output buffer to receive the name (not null-terminated).
 * @return The number of bytes written to `out`. Returns 0 for unknown values.
 *
 * @note `out` must have space for at least 9 bytes (e.g., "Hatchback").
 */
size_t __device__ __inline__ get_car_name(Car car, char *out) {
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

/**
 * @brief Format "`<Color> <Car>`" into an output buffer.
 *
 * Writes a null-terminated string consisting of the color name, a single space,
 * and the car name.
 *
 * @param car The car value to format.
 * @param color The color value to format.
 * @param[out] out Output buffer for the formatted string.
 *
 * @note `out` must have enough space for the longest combination, e.g.
 * "Black Hatchback\0" (16 bytes).
 */
void __device__ car_with_color(Car car, Color color, char *out) {
  char color_buf[32];
  char car_buf[32];

  size_t color_length = get_color_name(color, color_buf);
  size_t car_length = get_car_name(car, car_buf);

  memcpy(out, color_buf, color_length);
  out[color_length] = ' ';
  memcpy(out + color_length + 1, car_buf, car_length);
  out[color_length + car_length + 1] = '\0';
}
