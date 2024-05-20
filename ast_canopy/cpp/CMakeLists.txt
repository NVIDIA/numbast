# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(libastcanopy SHARED
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

# libastcanopy shared lib targget
find_package(Clang REQUIRED)

# Add path to LLVM modules
set(CMAKE_MODULE_PATH
  ${CMAKE_MODULE_PATH}
  "${LLVM_CMAKE_DIR}"
)

# import LLVM CMake functions
include(AddLLVM)

set(CLANG_LIBRARY_DIR ${CLANG_INSTALL_PREFIX}/lib)

target_include_directories(libastcanopy PUBLIC ${CLANG_INCLUDE_DIRS}
                                      PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
                                      PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/detail/)

add_definitions(${CLANG_DEFINITIONS})

target_link_directories(libastcanopy PRIVATE ${CLANG_LIBRARY_DIR})

target_link_libraries(libastcanopy PRIVATE clangTooling)

set_property(TARGET libastcanopy PROPERTY POSITION_INDEPENDENT_CODE ON)

install(TARGETS libastcanopy LIBRARY DESTINATION ${SKBUILD_PLATLIB_DIR})
