# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.28)
set(CMAKE_CXX_FLAGS_DEBUG  "-g")

if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/<PROJ>_RAPIDS.cmake)
  file(DOWNLOAD https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-24.06/RAPIDS.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/<PROJ>_RAPIDS.cmake)
endif()
include(${CMAKE_CURRENT_BINARY_DIR}/<PROJ>_RAPIDS.cmake)
include(rapids-cmake)
rapids_cmake_support_conda_env( conda_env MODIFY_PREFIX_PATH )

# Oddly, C language is needed here to build in PyTorch image on aarch64
# machine

project(${SKBUILD_PROJECT_NAME}
        VERSION ${SKBUILD_PROJECT_VERSION}
	LANGUAGES C CXX)

message("cmake_prefix: " "${CMAKE_PREFIX_PATH}")
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(astcanopy 0.1.0 REQUIRED)


python_add_library(pylibastcanopy MODULE ast_canopy/pylibastcanopy.cpp WITH_SOABI)
target_link_libraries(pylibastcanopy PRIVATE pybind11::headers)
target_link_libraries(pylibastcanopy PRIVATE conda_env)

target_link_libraries(pylibastcanopy PUBLIC astcanopy)

install(TARGETS pylibastcanopy LIBRARY DESTINATION .)
