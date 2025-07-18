# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# LLVM Configuration Module for astcanopy
# This module handles LLVM/Clang discovery and configures static/dynamic linking

function(configure_llvm_for_target target_name)
    # Find Clang package
    find_package(Clang REQUIRED)

    # Add path to LLVM modules
    set(CMAKE_MODULE_PATH
        ${CMAKE_MODULE_PATH}
        "${LLVM_CMAKE_DIR}"
        PARENT_SCOPE
    )

    # Import LLVM CMake functions
    include(AddLLVM)

        set(CLANG_LIBRARY_DIR ${CLANG_INSTALL_PREFIX}/lib)

    # Determine LLVM linkage based on user preference
    if(NOT DEFINED LLVM_LINKAGE)
        set(LLVM_LINKAGE "SHARED" CACHE STRING "LLVM linkage type (STATIC or SHARED)")
    endif()

    # Validate LLVM_LINKAGE value
    if(NOT LLVM_LINKAGE MATCHES "^(STATIC|SHARED)$")
        message(FATAL_ERROR "LLVM_LINKAGE must be either 'STATIC' or 'SHARED', got: ${LLVM_LINKAGE}")
    endif()

    # Check if requested linkage is available
    set(LLVM_USE_STATIC_LIBS OFF)
    if(LLVM_LINKAGE STREQUAL "STATIC")
        if(EXISTS "${CLANG_LIBRARY_DIR}/libclangTooling.a")
            message(STATUS "Using static LLVM linking as requested")
            set(LLVM_USE_STATIC_LIBS ON)

            # When building statically against clang, explicitly require ZLIB
            # since some clang components may depend on it
            find_package(ZLIB REQUIRED)

            # For static linking, we need to link against all required LLVM libraries
            set(CLANG_STATIC_LIBS
                clangTooling
                clangFrontendTool
                clangFrontend
                clangDriver
                clangSerialization
                clangCodeGen
                clangParse
                clangSema
                clangStaticAnalyzerFrontend
                clangStaticAnalyzerCheckers
                clangStaticAnalyzerCore
                clangAnalysis
                clangAST
                clangASTMatchers
                clangRewrite
                clangRewriteFrontend
                clangEdit
                clangLex
                clangBasic
            )

            # Find LLVM static libraries
            find_package(LLVM REQUIRED CONFIG)
            llvm_map_components_to_libnames(LLVM_STATIC_LIBS
                core
                irreader
                codegen
                target
                linker
                analysis
                scalaropts
                instcombine
                transformutils
                bitwriter
                x86codegen
                x86asmparser
                x86desc
                x86info
                mc
                mcparser
                mcdisassembler
                object
                option
                profiledata
            )
        else()
            message(FATAL_ERROR "Static LLVM libraries not found at ${CLANG_LIBRARY_DIR}/libclangTooling.a but LLVM_LINKAGE=STATIC was requested")
        endif()
    else()
        message(STATUS "Using dynamic LLVM linking as requested")
    endif()

    # Configure target include directories
    target_include_directories(${target_name} PUBLIC ${CLANG_INCLUDE_DIRS})

    # Add Clang definitions
    target_compile_definitions(${target_name} PRIVATE ${CLANG_DEFINITIONS})

    # Configure target link directories
    target_link_directories(${target_name} PRIVATE ${CLANG_LIBRARY_DIR})

    # Link libraries based on static/dynamic availability
    if(LLVM_USE_STATIC_LIBS)
        target_link_libraries(${target_name} PRIVATE ${CLANG_STATIC_LIBS} ${LLVM_STATIC_LIBS})
        # Static linking requires additional system libraries
        if(UNIX AND NOT APPLE)
            target_link_libraries(${target_name} PRIVATE dl pthread rt z)
        elseif(APPLE)
            target_link_libraries(${target_name} PRIVATE dl pthread z)
        endif()
    else()
        target_link_libraries(${target_name} PRIVATE clangTooling)
    endif()
endfunction()
