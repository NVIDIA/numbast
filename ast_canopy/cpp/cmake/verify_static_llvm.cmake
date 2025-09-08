# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Usage (from add_test):
#   ${CMAKE_COMMAND} -Dlib=<path-to-libastcanopy> -P verify_static_llvm.cmake

if(NOT lib)
  message(FATAL_ERROR "Variable 'lib' (path to target library) not provided")
endif()

if(NOT EXISTS "${lib}")
  message(FATAL_ERROR "Library not found: ${lib}")
endif()

if(NOT UNIX OR APPLE)
  message(STATUS "Static linkage check is only implemented for ELF (Linux). Passing by default.")
  return()
endif()

find_program(READELF_EXECUTABLE NAMES readelf)
if(NOT READELF_EXECUTABLE)
  message(FATAL_ERROR "'readelf' not found in PATH; cannot verify static linkage")
endif()

execute_process(
  COMMAND "${READELF_EXECUTABLE}" -d "${lib}"
  OUTPUT_VARIABLE readelf_out
  ERROR_VARIABLE readelf_err
  RESULT_VARIABLE readelf_rc
)

if(NOT readelf_rc EQUAL 0)
  message(FATAL_ERROR "readelf failed (rc=${readelf_rc}): ${readelf_err}")
endif()

# Look for dynamic NEEDED entries that would indicate dynamic LLVM/Clang deps
string(REGEX MATCH "NEEDED[^
]*libLLVM[^
]*" match_llvm "${readelf_out}")
string(REGEX MATCH "NEEDED[^
]*libclang[^
]*" match_clang "${readelf_out}")

if(match_llvm OR match_clang)
  message(FATAL_ERROR "Static LLVM linkage check failed. Found dynamic dependency: '${match_llvm}${match_clang}' in ${lib}")
endif()

message(STATUS "Static LLVM linkage check passed: no dynamic libLLVM/libclang dependencies in ${lib}")


