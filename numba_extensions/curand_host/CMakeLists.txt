# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Require CMake 3.15+ (matching scikit-build-core) Use new versions of all
# policies up to CMake 3.27
cmake_minimum_required(VERSION 3.15...3.27)

# Scikit-build-core sets these values for you, or you can just hard-code the
# name and version.
project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES CXX)

# Find the module development requirements (requires FindPython from 3.17 or
# scikit-build-core's built-in backport)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# As of CTK 12.4, this is where cuda_runtime.h locates
set(CUDAToolkit_INCLUDE_DIRECTORIES $ENV{CONDA_PREFIX}/targets/x86_64-linux/include)
find_package(CUDAToolkit REQUIRED)

# Add a library using FindPython's tooling (pybind11 also provides a helper like
# this)
python_add_library(curand_host MODULE curand_host_api.cpp WITH_SOABI)


target_link_libraries(curand_host PRIVATE pybind11::headers CUDA::curand)

# This is passing in the version as a define just as an example
target_compile_definitions(curand_host PRIVATE VERSION_INFO=${PROJECT_VERSION})

# The install directory is the output (wheel) directory
install(TARGETS curand_host DESTINATION .)
