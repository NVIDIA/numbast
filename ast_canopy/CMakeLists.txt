# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.29)

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" _version_raw)
string(STRIP ${_version_raw} _version)

# Oddly, C language is needed here to build in PyTorch image on aarch64
# machine

project(${SKBUILD_PROJECT_NAME}
        VERSION ${SKBUILD_PROJECT_VERSION}
	LANGUAGES C CXX)

# Prevent C++ modules from being scanned (not used in this project).
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(astcanopy ${_version} REQUIRED)

python_add_library(pylibastcanopy MODULE ast_canopy/pylibastcanopy.cpp WITH_SOABI)

set_property(TARGET pylibastcanopy PROPERTY CXX_STANDARD 17)

# Ensure RTTI is enabled for the Python bindings as well
target_compile_options(pylibastcanopy PRIVATE -frtti)

target_link_libraries(pylibastcanopy
  PRIVATE pybind11::headers
  PUBLIC astcanopy::astcanopy)

install(TARGETS pylibastcanopy LIBRARY DESTINATION ast_canopy)
