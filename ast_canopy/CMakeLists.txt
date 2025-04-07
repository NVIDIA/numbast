# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.28)

# Oddly, C language is needed here to build in PyTorch image on aarch64
# machine

project(${SKBUILD_PROJECT_NAME}
        VERSION ${SKBUILD_PROJECT_VERSION}
	LANGUAGES C CXX)


find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(astcanopy ${SKBUILD_PROJECT_VERSION} REQUIRED)

python_add_library(pylibastcanopy MODULE ast_canopy/pylibastcanopy.cpp WITH_SOABI)

set_property(TARGET pylibastcanopy PROPERTY CXX_STANDARD 17)

target_link_libraries(pylibastcanopy PRIVATE pybind11::headers)
target_link_libraries(pylibastcanopy PUBLIC astcanopy::astcanopy)

install(TARGETS pylibastcanopy LIBRARY DESTINATION .)
