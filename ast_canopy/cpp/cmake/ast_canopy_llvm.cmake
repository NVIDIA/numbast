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
  set(_clang_targets
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

  target_link_libraries(${library_target} PRIVATE ${_clang_targets})

  if(NOT DEFINED LLVM_LINKAGE)
    message(STATUS "LLVM_LINKAGE is not defined, using SHARED")
    set(LLVM_LINKAGE "SHARED")
  endif()

  foreach(_clang_target ${_clang_targets})
    get_target_property(_clang_target_type ${_clang_target} TYPE)
    if(NOT "${_clang_target_type}" STREQUAL "${LLVM_LINKAGE}_LIBRARY")
      get_target_property(_clang_target_location ${_clang_target} LOCATION)
      message(STATUS "Library file path: ${_clang_target_location}")
      if("${LLVM_LINKAGE}" STREQUAL "STATIC")
        message(FATAL_ERROR "LLVM_LINKAGE is set to ${LLVM_LINKAGE}, but "
                            "${_clang_target} is of type ${_clang_target_type}. "
                            "Please install a correctly compiled Clang/LLVM.")
      else()
        message(STATUS "Warning:LLVM_LINKAGE is set to ${LLVM_LINKAGE}, but "
                       "${_clang_target} is of type ${_clang_target_type}.")
      endif()
    endif()
  endforeach()
endfunction()
