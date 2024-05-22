# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.28)

project(astcanopy VERSION 0.1.0 LANGUAGES C CXX)

add_library(astcanopy SHARED
            src/ast_canopy.cpp
            src/enum.cpp
            src/class_template.cpp
            src/field.cpp
            src/function.cpp
            src/function_template.cpp
            src/method.cpp
            src/param_var.cpp
            src/record.cpp
            src/template.cpp
            src/template_param.cpp
            src/type.cpp
            src/typedef.cpp
            src/detail/matchers/function_matcher.cpp
            src/detail/matchers/record_matcher.cpp
            src/detail/matchers/typedef_matcher.cpp
            src/detail/matchers/function_template_matcher.cpp
            src/detail/matchers/class_template_matcher.cpp
            src/detail/matchers/enum_matcher.cpp)

# astcanopy shared lib target
find_package(Clang REQUIRED)

# Add path to LLVM modules
set(CMAKE_MODULE_PATH
  ${CMAKE_MODULE_PATH}
  "${LLVM_CMAKE_DIR}"
)

# For distro-independent install dirs
include(GNUInstallDirs)

# import LLVM CMake functions
include(AddLLVM)

set(CLANG_LIBRARY_DIR ${CLANG_INSTALL_PREFIX}/lib)

target_include_directories(astcanopy PUBLIC ${CLANG_INCLUDE_DIRS}
                                      PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                                      PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/detail/)

add_definitions(${CLANG_DEFINITIONS})

target_link_directories(astcanopy PRIVATE ${CLANG_LIBRARY_DIR})

target_link_libraries(astcanopy PRIVATE clangTooling)

set_property(TARGET astcanopy PROPERTY POSITION_INDEPENDENT_CODE ON)

# Install built library into host environment
install(TARGETS astcanopy
  EXPORT astcanopy-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(EXPORT astcanopy-targets
  FILE astcanopy-targets.cmake
  NAMESPACE astcanopy::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/astcanopy)
