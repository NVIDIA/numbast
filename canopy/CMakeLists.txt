# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.15...3.27)
set(CMAKE_CXX_FLAGS_DEBUG  "-g")

project(${SKBUILD_PROJECT_NAME}
        VERSION ${SKBUILD_PROJECT_VERSION}
        LANGUAGES CXX)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

add_subdirectory(cpp)

python_add_library(pylibcanopy MODULE canopy/pylibcanopy.cpp WITH_SOABI)
target_link_libraries(pylibcanopy PRIVATE pybind11::headers)

target_link_libraries(pylibcanopy PUBLIC libcanopy)

install(TARGETS pylibcanopy LIBRARY DESTINATION .)
