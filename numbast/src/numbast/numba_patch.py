# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import warnings
import logging

from numba import cuda, config

from ast_canopy.api import get_default_cuda_compiler_includes

logger = logging.getLogger(__name__)

old_cuda_include_path = config.CUDA_INCLUDE_PATH
new_cuda_include_path = get_default_cuda_compiler_includes(config.CUDA_INCLUDE_PATH)
if old_cuda_include_path != new_cuda_include_path:
    logger.info("Updating CUDA include path to %s", new_cuda_include_path)
os.environ["NUMBA_CUDA_INCLUDE_PATH"] = new_cuda_include_path
config.reload_config()

# TODO: upstream the changes here to Numba.

# Add extra options (include path, optimization flags, etc) for NVRTC here
extra_options = [
    "-std=c++17",
]

extra_include_paths = [
    "-I/home/wangm/cccl/cub/",
    "-I/home/wangm/cccl/libcudacxx/include/",
    "-I/home/wangm/cccl/thrust/",
]


# Copied from numba.cuda.cudadrv.nvrtc, modified to allow extra options to be
# added.


def nvrtc_compile(src, name, cc):
    """
    Compile a CUDA C/C++ source to PTX for a given compute capability.

    :param src: The source code to compile
    :type src: str
    :param name: The filename of the source (for information only)
    :type name: str
    :param cc: A tuple ``(major, minor)`` of the compute capability
    :type cc: tuple
    :return: The compiled PTX and compilation log
    :rtype: tuple
    """
    nvrtc = cuda.cudadrv.nvrtc.NVRTC()
    program = nvrtc.create_program(src, name)

    # Compilation options:
    # - Compile for the current device's compute capability.
    # - The CUDA include path is added.
    # - Relocatable Device Code (rdc) is needed to prevent device functions
    #   being optimized away.
    major, minor = cc
    arch = f"--gpu-architecture=compute_{major}{minor}"
    include = f"-I{config.CUDA_INCLUDE_PATH}"

    cudadrv = cuda.cudadrv.__file__
    cudadrv_path = os.path.dirname(os.path.abspath(cudadrv))
    numba_cuda_path = os.path.dirname(cudadrv_path)
    numba_include = f"-I{numba_cuda_path}"
    options = [arch, *extra_include_paths, include, numba_include, "-rdc", "true"]
    options += extra_options

    # Compile the program
    compile_error = nvrtc.compile_program(program, options)

    # Get log from compilation
    log = nvrtc.get_compile_log(program)

    # If the compile failed, provide the log in an exception
    if compile_error:
        msg = f"NVRTC Compilation failure whilst compiling {name}:\n\n{log}"
        raise cuda.cudadrv.nvrtc.NvrtcError(msg)

    # Otherwise, if there's any content in the log, present it as a warning
    if log:
        msg = f"NVRTC log messages whilst compiling {name}:\n\n{log}"
        warnings.warn(msg)

    ptx = nvrtc.get_ptx(program)
    return ptx, log


# Monkey-patch the existing implementation
cuda.cudadrv.nvrtc.compile = nvrtc_compile
