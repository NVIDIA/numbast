# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import shutil
import os
import tempfile
import logging
from typing import Optional
from dataclasses import dataclass

from numba.cuda.cuda_paths import get_nvidia_nvvm_ctk, get_cuda_home

import pylibastcanopy as bindings

from ast_canopy.decl import Function, Struct, ClassTemplate
from ast_canopy.fdcap_min import capture_fd, STREAMFD

logger = logging.getLogger(f"AST_Canopy.{__name__}")


@dataclass
class Declarations:
    structs: list[Struct]
    functions: list[Function]
    function_templates: list[bindings.FunctionTemplate]
    class_templates: list[ClassTemplate]
    typedefs: list[bindings.Typedef]
    enums: list[bindings.Enum]


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
        return shutil.which("nvcc")

    root = os.path.dirname(os.path.dirname(nvvm_path))
    nvcc_path = os.path.join(root, "bin", "nvcc")

    if os.path.exists(nvcc_path):
        logger.info(f"Found NVCC path: {nvcc_path}")
        return nvcc_path


def get_default_compiler_search_paths() -> list[str]:
    """Compile an empty file with clang++ and print logs verbosely.
    Extract the default system header search paths from the logs and return them.
    """

    # clang++ needs to be put in cuda mode so that it can include the proper headers
    clang_compile_empty = (
        subprocess.check_output(
            [
                "clang++",
                "-fsyntax-only",
                "-v",
                "--cuda-device-only",
                "--no-cuda-version-check",
                "-xcuda",
                "/dev/null",
            ],
            stderr=subprocess.STDOUT,
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
    """Compile an empty file with NVCC and extract its default include path.

    ast_canopy depends on a healthy cuda environment to function. If nvcc fails
    to compile an empty CUDA file, this function will raise an error.
    """

    nvcc_bin = get_default_nvcc_path()
    if not nvcc_bin:
        logger.warning(
            "Could not find NVCC binary. AST_Canopy will attempt to "
            "invoke `nvcc` directly in the subsequent commands."
        )
        nvcc_bin = "nvcc"

    with tempfile.NamedTemporaryFile(suffix=".cu") as tmp_file:
        try:
            nvcc_compile_empty = (
                subprocess.run(
                    [nvcc_bin, "-E", "-v", tmp_file.name],
                    capture_output=True,
                    check=True,
                )
                .stderr.decode()
                .strip()
                .split("\n")
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"NVCC failed to compile an empty cuda file. \n {e.stdout.decode('utf-8')} \n {e.stderr.decode('utf-8')}"
            ) from e

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
    cxx_standard: str = "c++17",
    additional_includes: list[str] = [],
    anon_filename_decl_prefix_allowlist: list[str] = [],
    verbose: bool = False,
) -> tuple[
    list[Struct],
    list[Function],
    list[bindings.FunctionTemplate],
    list[ClassTemplate],
    list[bindings.Typedef],
    list[bindings.Enum],
]:
    """Given a source file, parse all *top-level* declarations from it.
    Returns a tuple that each contains a list of declaration objects for the source file.

    `files_to_retain` is a required parameter to specify the declarations from which files
    should be ratained in the result. See `Parameters` section for more details.

    Parameters
    ----------
    source_file_path : str
        The path to the source file to parse.

    files_to_retain : list[str]
        A list of file paths, from which the parsed declarations that should be retained in
        the result. A header file usually reference other header files. A semantically intact
        AST should in theory include all referenced header files. In practice, one may not
        need to consume all the other referenced files. To only retain the declarations from
        `source_file_path`, one may pass [source_file_path] to this parameter.

    compute_capability : str
        The compute capability of the target GPU. e.g. "sm_70".

    cccl_root : str, optional
        The root directory of the CCCL project. If not provided, CCCL from default CTK headers
        are used.

    cudatoolkit_include_dir : str, optional
        The path to the CUDA Toolkit include directory. If not provided, the default CUDA include
        directory is used.

    cxx_standard : str, optional
        The C++ standard to use. Default is "c++17".

    additional_includes : list[str], optional
        A list of additional include directories to search for headers.

    anon_filename_decl_prefix_allowlist : list[str], optional
        A list of prefixes to allow declarations with anonymous filename from. This is a temporary
        workaround to allow expaneded macros to be included in the AST.

    verbose : bool, optional
        If True, print the stderr from clang++ invocation.

    Returns
    -------
    tuple:
        A tuple containing lists of declaration objects from the source file.
        See decl.py and pylibastcanopy.pyi for the declaration object types.
    """

    if not os.path.exists(source_file_path):
        raise FileNotFoundError(f"File not found: {source_file_path}")

    for p in additional_includes:
        if not isinstance(p, str):
            raise TypeError(f"Additional include path must be a string: {p}")
        if p.startswith("-I"):
            raise ValueError(f"Additional include path must not start with -I: {p}")
        if not os.path.exists(p):
            raise FileNotFoundError(f"Additional include path not found: {p}")

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
        *[f"-I{path}" for path in additional_includes],
        source_file_path,
    ]

    logger.debug(f"{command_line_options=}")
    if verbose:
        print(f"{command_line_options=}")

    with capture_fd(STREAMFD.STDERR) as cap:
        decls = bindings.parse_declarations_from_command_line(
            command_line_options, files_to_retain, anon_filename_decl_prefix_allowlist
        )

    werr = cap.snap()
    if werr:
        liblogger = logging.getLogger("libastcanopy")
        liblogger.debug(werr)
        if verbose:
            print(werr)
        if (
            "CUDA version" in werr
            and "is newer than the latest supported version" in werr
        ):
            liblogger.info(
                "Installed cudaToolkit version is newer than the latest supported version of the clangTooling "
                "backend. clangTooling will treat the cudaToolkit as if it is its latest supported version."
            )

    structs = [Struct.from_c_obj(c_obj) for c_obj in decls.records]
    functions = [Function.from_c_obj(c_obj) for c_obj in decls.functions]
    class_templates = [
        ClassTemplate.from_c_obj(c_obj) for c_obj in decls.class_templates
    ]

    return Declarations(
        structs,
        functions,
        decls.function_templates,
        class_templates,
        decls.typedefs,
        decls.enums,
    )
