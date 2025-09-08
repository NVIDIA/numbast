# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.29)

# Oddly, C language is needed here to build in PyTorch image on aarch64
# machine

project(${SKBUILD_PROJECT_NAME}
        VERSION ${SKBUILD_PROJECT_VERSION}
	LANGUAGES C CXX)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(astcanopy ${SKBUILD_PROJECT_VERSION} REQUIRED)

# Ensure Clang/LLVM headers are available to the Python extension build
## Consumers get LLVM/Clang include paths via astcanopy target's INTERFACE

python_add_library(pylibastcanopy MODULE ast_canopy/pylibastcanopy.cpp WITH_SOABI)

set_property(TARGET pylibastcanopy PROPERTY CXX_STANDARD 17)

# Ensure RTTI is enabled for the Python bindings as well
target_compile_options(pylibastcanopy PRIVATE -frtti)

# Add Clang/LLVM include paths for compiling the bindings that include astcanopy headers
# Prefer variables exported by astcanopy's package config if available
if(DEFINED ASTCANOPY_LLVM_INCLUDE_DIRS OR DEFINED ASTCANOPY_CLANG_INCLUDE_DIRS)
  target_include_directories(pylibastcanopy PRIVATE ${ASTCANOPY_LLVM_INCLUDE_DIRS} ${ASTCANOPY_CLANG_INCLUDE_DIRS})
else()
  target_include_directories(pylibastcanopy PRIVATE ${LLVM_INCLUDE_DIRS} ${CLANG_INCLUDE_DIRS})
endif()

target_link_libraries(pylibastcanopy PRIVATE pybind11::headers)
target_link_libraries(pylibastcanopy PUBLIC astcanopy::astcanopy)

install(TARGETS pylibastcanopy LIBRARY DESTINATION .)
