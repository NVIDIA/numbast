# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import os
import tempfile
import logging
from typing import Optional

from numba.cuda.cuda_paths import get_nvidia_nvvm_ctk, get_cuda_home

import pylibastcanopy as bindings

from ast_canopy.decl import Function, Struct, ClassTemplate

logger = logging.getLogger(f"AST_Canopy.{__name__}")


def get_default_cuda_path() -> Optional[str]:
    """Return the path to the default CUDA home directory."""

    # Allow overriding from os.environ
    if home := get_cuda_home():
        return home

    if nvvm_path := get_nvidia_nvvm_ctk():
        # In the form of $ROOT/nvvm/lib64/libnvvm.so, go up 2 levels for cuda home.
        cuda_home = os.path.dirname(os.path.dirname(nvvm_path))

        if os.path.exists(cuda_home):
            logger.info(f"Found CUDA home: {cuda_home}")
            return cuda_home


def get_default_nvcc_path() -> Optional[str]:
    """Return the path to the default NVCC compiler binary."""
    nvvm_path = get_nvidia_nvvm_ctk()

    if not nvvm_path:
        return

    root = os.path.dirname(os.path.dirname(nvvm_path))
    nvcc_path = os.path.join(root, "bin", "nvcc")

    if os.path.exists(nvcc_path):
        logger.info(f"Found NVCC path: {nvcc_path}")
        return nvcc_path


def get_default_compiler_search_paths() -> list[str]:
    """Compile an empty file with clang++ and print logs verbosely.
    Extract the default system header search paths from the logs and return them.
    """

    # Alternatively, use `clang++ --print-search-dirs`
    clang_compile_empty = (
        subprocess.check_output(
            ["clang++", "-E", "-v", "-xc++", "/dev/null"], stderr=subprocess.STDOUT
        )
        .decode()
        .strip()
        .split("\n")
    )
    start = clang_compile_empty.index("#include <...> search starts here:")
    end = clang_compile_empty.index("End of search list.")
    clang_system_header_search_paths = clang_compile_empty[start + 1 : end]
    search_paths = [x.strip() for x in clang_system_header_search_paths]
    logger.info(f"Default compiler search paths: {search_paths=}")
    return search_paths


def get_default_cuda_compiler_include(default="/usr/local/cuda/include") -> str:
    """Compile an empty file with NVCC and extract its default include path."""

    nvcc_bin = get_default_nvcc_path()
    if not nvcc_bin:
        logger.warning("Could not find NVCC binary. Using default nvcc bin from env.")
        nvcc_bin = "nvcc"

    with tempfile.NamedTemporaryFile(suffix=".cu") as tmp_file:
        nvcc_compile_empty = (
            subprocess.run([nvcc_bin, "-E", "-v", tmp_file.name], capture_output=True)
            .stderr.decode()
            .strip()
            .split("\n")
        )

    if s := [i for i in nvcc_compile_empty if "INCLUDES=" in i]:
        include_path = s[0].lstrip("#$ INCLUDES=").strip().strip('"').lstrip("-I")
        logger.info(f"Found NVCC default include path, {include_path=}")
        return include_path
    else:
        logger.warning(
            f"Could not find NVCC default include path. Using default: {default=}"
        )
        return default


def parse_declarations_from_source(
    source_file_path: str,
    files_to_retain: list[str],
    compute_capability: str,
    cccl_root: str = "",
    cudatoolkit_include_dir: str = get_default_cuda_compiler_include(),
    cxx_standard="c++17",
) -> tuple[
    list[Struct],
    list[Function],
    list[bindings.FunctionTemplate],
    list[ClassTemplate],
    list[bindings.Typedef],
    list[bindings.Enum],
]:
    if cccl_root:
        cccl_libs = [
            os.path.join(cccl_root, "libcudacxx", "include"),
            os.path.join(cccl_root, "cub"),
            os.path.join(cccl_root, "thrust"),
        ]
        cccl_libs = [f"-I{lib}" for lib in cccl_libs]
    else:
        cccl_libs = []

    clang_resource_file = (
        subprocess.check_output(["clang++", "-print-resource-dir"]).decode().strip()
    )

    clang_search_paths = get_default_compiler_search_paths()

    def custom_cuda_home() -> list[str]:
        cuda_path = get_default_cuda_path()
        if cuda_path:
            return [f"--cuda-path={cuda_path}"]
        else:
            return []

    command_line_options = [
        "clang++",
        "--cuda-device-only",
        "-xcuda",
        f"--cuda-gpu-arch={compute_capability}",
        *custom_cuda_home(),
        f"-std={cxx_standard}",
        f"-isystem{clang_resource_file}/include/",
        *[f"-I{path}" for path in clang_search_paths],
        *cccl_libs,
        f"-I{cudatoolkit_include_dir}",
        source_file_path,
    ]

    logger.info(f"{command_line_options=}")

    decls = bindings.parse_declarations_from_command_line(
        command_line_options, files_to_retain
    )
    structs = [Struct.from_c_obj(c_obj) for c_obj in decls.records]
    functions = [Function.from_c_obj(c_obj) for c_obj in decls.functions]
    class_templates = [
        ClassTemplate.from_c_obj(c_obj) for c_obj in decls.class_templates
    ]

    return (
        structs,
        functions,
        decls.function_templates,
        class_templates,
        decls.typedefs,
        decls.enums,
    )
