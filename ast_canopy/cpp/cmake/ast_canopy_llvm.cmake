# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

function(ast_canopy_check_llvm_linkage library_target)
  if(NOT DEFINED LLVM_LINKAGE)
    message(STATUS "LLVM_LINKAGE is not defined, using SHARED")
    set(LLVM_LINKAGE "SHARED")
  endif()
endfunction()

function(ast_canopy_target_link_clang library_target)
  # Check that we are getting the linkage requested for Clang and LLVM
  if(TARGET Clang::clangTooling)
    message(STATUS "Modern Clang::clangTooling target found")
    set(_target_prefix "Clang::")
  else()
    message(STATUS "Old clangTooling target fallback")
    set(_clang_targets ${_clang_targets})
  endif()

  set(_clang_targets
    ${_target_prefix}clangTooling
    ${_target_prefix}clangASTMatchers
    ${_target_prefix}clangAST
    ${_target_prefix}clangBasic
    ${_target_prefix}clangFrontend
    ${_target_prefix}clangSerialization
    ${_target_prefix}clangDriver
    ${_target_prefix}clangParse
    ${_target_prefix}clangSema
    ${_target_prefix}clangAnalysis
    ${_target_prefix}clangEdit
    ${_target_prefix}clangRewrite
    ${_target_prefix}clangLex
  )

  target_link_libraries(${library_target} PRIVATE ${_clang_targets})

  if(NOT DEFINED LLVM_LINKAGE)
    message(STATUS "LLVM_LINKAGE is not defined, using SHARED")
    set(LLVM_LINKAGE "SHARED")
  endif()

  foreach(_clang_target ${_clang_targets})
    message(STATUS "Linking ${_clang_target}")
    get_target_property(_clang_target_type ${_clang_target} TYPE)
    message(STATUS "Type: ${_clang_target_type}")
    if (NOT "${_clang_target_type}" STREQUAL "${LLVM_LINKAGE}_LIBRARY")
      message(FATAL_ERROR "LLVM_LINKAGE is set to ${LLVM_LINKAGE}, but "
                          "${_clang_target} is of type ${_clang_target_type}. "
                          "Please install a correctly compiled Clang/LLVM.")
    endif()
  endforeach()
endfunction()
