// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <curand.h>

#include <iostream>

namespace py = pybind11;

PYBIND11_MODULE(curand_host, m) {
  m.doc() = "Python bindings for curand.h host APIs";

  py::enum_<curandDirectionVectorSet>(m, "curandDirectionVectorSet")
      .value("CURAND_DIRECTION_VECTORS_32_JOEKUO6",
             CURAND_DIRECTION_VECTORS_32_JOEKUO6)
      .value("CURAND_DIRECTION_VECTORS_64_JOEKUO6",
             CURAND_DIRECTION_VECTORS_64_JOEKUO6)
      .value("CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6",
             CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6)
      .value("CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6",
             CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6);

  m.def(
      "curandGetDirectionVectors64",
      [](py::size_t size, curandDirectionVectorSet_t set) {
        curandDirectionVectors64_t *vectors;
        curandGetDirectionVectors64(&vectors, set);

        unsigned long long *v = reinterpret_cast<unsigned long long *>(vectors);
        py::capsule free_when_done(v, [](void *v) {});

        return py::array_t<unsigned long long>({size * 64}, {8}, v,
                                               free_when_done);
      },
      "Get direction vectors for 64-bit quasirandom number generation");

  m.def(
      "curandGetScrambleConstants64",
      [](py::size_t size) {
        unsigned long long *constants;
        curandGetScrambleConstants64(&constants);

        py::capsule free_when_done(constants, [](void *c) {});

        return py::array_t<unsigned long long>({size}, {8}, constants,
                                               free_when_done);
      },
      "Get scramble constants for 64-bit scrambled Sobol' .");
}
