# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# LLVM Configuration Module for astcanopy
# This module handles LLVM/Clang discovery and configures static/dynamic linking

function(configure_llvm_for_target target_name)
	# Determine LLVM linkage based on user preference
	if(NOT DEFINED LLVM_LINKAGE)
		set(LLVM_LINKAGE "SHARED" CACHE STRING "LLVM linkage type (STATIC or SHARED)")
	endif()

	if(NOT LLVM_LINKAGE MATCHES "^(STATIC|SHARED)$")
		message(FATAL_ERROR "LLVM_LINKAGE must be either 'STATIC' or 'SHARED', got: ${LLVM_LINKAGE}")
	endif()

	# Discover LLVM and Clang after flags are set
	find_package(LLVM REQUIRED CONFIG)
	find_package(Clang REQUIRED CONFIG)

	message(STATUS "Found LLVM: ${LLVM_DIR}")
	message(STATUS "Found Clang: ${Clang_DIR}")

	include_directories(SYSTEM ${LLVM_INCLUDE_DIRS} ${CLANG_INCLUDE_DIRS})
	add_definitions(${LLVM_DEFINITIONS} ${CLANG_DEFINITIONS})

	# Derive library directories if not explicitly provided by the packages
	if(NOT DEFINED CLANG_LIBRARY_DIR)
		if(DEFINED CLANG_INSTALL_PREFIX)
			set(CLANG_LIBRARY_DIR ${CLANG_INSTALL_PREFIX}/lib)
		else()
			# Clang_DIR typically points to <prefix>/lib/cmake/clang
			get_filename_component(_clang_pkg_dir ${Clang_DIR} ABSOLUTE)
			# Go three levels up to reach <prefix>
			get_filename_component(_clang_prefix ${_clang_pkg_dir}/../../.. ABSOLUTE)
			set(CLANG_LIBRARY_DIR ${_clang_prefix}/lib)
		endif()
	endif()
	if(NOT DEFINED LLVM_LIBRARY_DIR)
		if(DEFINED LLVM_LIBRARY_DIRS)
			list(GET LLVM_LIBRARY_DIRS 0 LLVM_LIBRARY_DIR)
		else()
			# LLVM_DIR typically points to <prefix>/lib/cmake/llvm
			get_filename_component(_llvm_pkg_dir ${LLVM_DIR} ABSOLUTE)
			# Go three levels up to reach <prefix>
			get_filename_component(_llvm_prefix ${_llvm_pkg_dir}/../../.. ABSOLUTE)
			set(LLVM_LIBRARY_DIR ${_llvm_prefix}/lib)
		endif()
	endif()

	# Use imported targets for correct static/shared resolution
	if(LLVM_LINKAGE STREQUAL "STATIC")
		# When building statically against clang, explicitly require ZLIB
		find_package(ZLIB REQUIRED)

		# Validate presence of static archives; fail early with guidance
		if(NOT EXISTS ${CLANG_LIBRARY_DIR}/libclangTooling.a)
			message(FATAL_ERROR "Requested static LLVM/Clang linkage, but 'libclangTooling.a' was not found in ${CLANG_LIBRARY_DIR}. Install a static Clang toolchain (build LLVM/Clang from source with BUILD_SHARED_LIBS=OFF and LLVM_BUILD_LLVM_DYLIB=OFF) and set CMAKE_PREFIX_PATH/LLVM_DIR/Clang_DIR accordingly.")
		endif()
		# Require the monolithic libLLVM.a or a representative core archive to ensure static LLVM is available
		find_file(_LLVM_MONO_A NAMES libLLVM.a PATHS ${LLVM_LIBRARY_DIR} NO_DEFAULT_PATH)
		if(NOT _LLVM_MONO_A)
			# Fallback: accept component archives, but require at least core/support
			find_file(_LLVM_SUPPORT_A NAMES libLLVMSupport.a PATHS ${LLVM_LIBRARY_DIR} NO_DEFAULT_PATH)
			find_file(_LLVM_CORE_A NAMES libLLVMCore.a PATHS ${LLVM_LIBRARY_DIR} NO_DEFAULT_PATH)
			if(NOT (_LLVM_SUPPORT_A AND _LLVM_CORE_A))
				message(FATAL_ERROR "Requested static LLVM/Clang linkage, but neither 'libLLVM.a' nor the component archives 'libLLVMSupport.a' and 'libLLVMCore.a' were found in ${LLVM_LIBRARY_DIR}. Install static LLVM libraries and set CMAKE_PREFIX_PATH/LLVM_DIR accordingly.")
			endif()
		endif()

		# Clang components (prefer imported targets; fallback to bare library names)
		if(TARGET Clang::clangTooling)
			set(CLANG_STATIC_TARGETS
				Clang::clangTooling
				Clang::clangFrontendTool
				Clang::clangFrontend
				Clang::clangDriver
				Clang::clangSerialization
				Clang::clangCodeGen
				Clang::clangParse
				Clang::clangSema
				Clang::clangStaticAnalyzerFrontend
				Clang::clangStaticAnalyzerCheckers
				Clang::clangStaticAnalyzerCore
				Clang::clangAnalysis
				Clang::clangAST
				Clang::clangASTMatchers
				Clang::clangRewrite
				Clang::clangRewriteFrontend
				Clang::clangEdit
				Clang::clangLex
				Clang::clangBasic
			)
		else()
			set(CLANG_STATIC_TARGETS
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
		endif()

		# Map LLVM components to libraries (obeys LLVM_USE_STATIC_LIBS and LLVM_LINK_LLVM_DYLIB)
		llvm_map_components_to_libnames(LLVM_STATIC_LIBS
			# Core IR and analysis pipeline
			core
			analysis
			scalaropts
			instcombine
			transformutils
			linker
			object
			option
			profiledata

			# IR and bitcode IO
			bitreader
			bitstreamreader
			bitwriter
			irreader

			# Support and utilities
			support
			demangle
			remarks
			binaryformat
			textapi
			windowsdriver

			# Target and MC layers
			target
			mc
			mcparser
			mcdisassembler
			x86codegen
			x86asmparser
			x86desc
			x86info
			targetparser

			# Debug info and OpenMP used transitively by Clang components
			debuginfodwarf
			frontendopenmp
		)
        # Link with static Clang targets and specific LLVM component libs
        target_link_libraries(${target_name} PRIVATE ${CLANG_STATIC_TARGETS} ${LLVM_STATIC_LIBS})

		# Add system libs for static linking
		if(UNIX AND NOT APPLE)
			target_link_libraries(${target_name} PRIVATE dl pthread rt z)
		elseif(APPLE)
			target_link_libraries(${target_name} PRIVATE dl pthread z)
		endif()
	else()
		# Dynamic: link to high-level Clang target and let LLVM config pick the right deps
		if(TARGET Clang::clangTooling)
			target_link_libraries(${target_name} PRIVATE Clang::clangTooling)
		else()
			target_link_libraries(${target_name} PRIVATE clangTooling)
		endif()
	endif()
endfunction()
