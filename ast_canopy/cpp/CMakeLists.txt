# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.28)

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/../VERSION" _version_raw)
string(STRIP ${_version_raw} _version)

project(astcanopy VERSION ${_version} LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Define the sources to compile for astcanopy here.
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
            src/constexpr_vardecl.cpp
            src/detail/matchers/function_matcher.cpp
            src/detail/matchers/record_matcher.cpp
            src/detail/matchers/typedef_matcher.cpp
            src/detail/matchers/function_template_matcher.cpp
            src/detail/matchers/class_template_matcher.cpp
            src/detail/matchers/enum_matcher.cpp
            src/detail/matchers/constexpr_vardecl_matcher.cpp
            src/detail/matchers/class_template_partial_specialization_matcher.cpp)

# Define the public headers for astcanopy here.
target_sources(astcanopy
  PUBLIC
  FILE_SET public_headers
  TYPE HEADERS
  BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
  FILES include/ast_canopy/ast_canopy.hpp
  include/ast_canopy/error.hpp)

# astcanopy shared lib target
find_package(Clang REQUIRED)

# Add path to LLVM modules
set(CMAKE_MODULE_PATH
  ${CMAKE_MODULE_PATH}
  "${LLVM_CMAKE_DIR}"
)

# import LLVM CMake functions
include(AddLLVM)

set(CLANG_LIBRARY_DIR ${CLANG_INSTALL_PREFIX}/lib)

target_include_directories(astcanopy PUBLIC ${CLANG_INCLUDE_DIRS}
                                      PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                                      PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/detail/>
                                      INTERFACE $<INSTALL_INTERFACE:include>)

add_definitions(${CLANG_DEFINITIONS})

target_link_directories(astcanopy PRIVATE ${CLANG_LIBRARY_DIR})

# Ensure RTTI is enabled for this target
target_compile_options(astcanopy PRIVATE -frtti)

# Link required clang libraries - clangTooling alone is not sufficient
# We need additional libraries that contain AST matcher symbols with RTTI
target_link_libraries(astcanopy PRIVATE
    clangTooling
    clangASTMatchers
    clangAST
    clangBasic
    clangFrontend
    clangSerialization
    clangDriver
    clangParse
    clangSema
    clangAnalysis
    clangEdit
    clangRewrite
    clangLex
)

set_property(TARGET astcanopy PROPERTY POSITION_INDEPENDENT_CODE ON)

# For distro-independent install dirs
include(GNUInstallDirs)


# Install built library into host environment
install(TARGETS astcanopy
  EXPORT astcanopy-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILE_SET public_headers
)

install(EXPORT astcanopy-targets
  FILE astcanopy-config.cmake
  NAMESPACE astcanopy::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/astcanopy)

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/astcanopy-config-version.cmake
  VERSION ${_version}
  COMPATIBILITY SameMajorVersion)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/astcanopy-config-version.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/astcanopy)
