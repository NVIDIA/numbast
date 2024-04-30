# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.28)
set(CMAKE_CXX_FLAGS_DEBUG  "-g")

project(${SKBUILD_PROJECT_NAME}
        VERSION ${SKBUILD_PROJECT_VERSION}
	LANGUAGES C CXX)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

add_subdirectory(cpp)

python_add_library(pylibastcanopy MODULE ast_canopy/pylibastcanopy.cpp WITH_SOABI)
target_link_libraries(pylibastcanopy PRIVATE pybind11::headers)

target_link_libraries(pylibastcanopy PUBLIC libastcanopy)

install(TARGETS pylibastcanopy LIBRARY DESTINATION .)
