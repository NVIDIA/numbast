# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import os
import tempfile
import logging
from typing import Any
from dataclasses import dataclass
import warnings
import functools

from cuda.pathfinder import find_nvidia_header_directory

from ast_canopy import pylibastcanopy as bindings

from ast_canopy.decl import (
    Function,
    Struct,
    FunctionTemplate,
    ClassTemplate,
    ConstExprVar,
)
from ast_canopy.fdcap_min import capture_fd, STREAMFD

logger = logging.getLogger(f"AST_Canopy.{__name__}")


@dataclass
class Declarations:
    structs: list[Struct]
    functions: list[Function]
    function_templates: list[FunctionTemplate]
    class_templates: list[ClassTemplate]
    typedefs: list[bindings.Typedef]
    enums: list[bindings.Enum]


def paths_to_include_flags(paths: list[str]) -> list[str]:
    return [f"-I{path}" for path in paths]


def get_default_compiler_search_paths() -> list[str]:
    """Compile an empty CUDA file with clang++ in verbose mode and parse the
    output to extract the default system header search paths."""

    # clang++ needs to be put in cuda mode so that it can include the proper headers.
    # The bare minimum of the cuda mode is with `-nocudainc` and `-no-cuda-version-check`
    # since we are only interested in the cuda patches that clang will include for
    # std headers.
    clang_compile_empty = (
        subprocess.check_output(
            [
                "clang++",
                "-fsyntax-only",
                "-v",
                "--cuda-device-only",
                "-nocudainc",
                "-nocudalib",
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


def get_cuda_path_for_clang() -> str | None:
    """Get the cuda root directory for clang's `--cuda-path` flag.

    `cuda-path` requires unix-like folder structure. For CUDA 12 wheel user,
    we will instead fallback to the CUDA_HOME environment variable.
    """

    def _validate_root(root: str) -> bool:
        if not os.path.exists(root):
            return False

        subdirs = ["include", "bin"]

        for subdir in subdirs:
            if not os.path.exists(os.path.join(root, subdir)):
                return False

        return True

    cudaruntime_dir = get_cuda_include_dir_for_clang()["cudart"]
    if cudaruntime_dir:
        root = os.path.dirname(cudaruntime_dir)
        if _validate_root(root):
            return root

    cuda_home = os.environ.get("CUDA_HOME", None)
    if not cuda_home:
        raise RuntimeError(
            "Unable to find CUDA root directory for clang's "
            "`--cuda-path` flag. Please set CUDA_HOME environment variable. "
            "For CUDA 12 wheel user, please install the CUDA Toolkit via "
            "system package manager and set CUDA_HOME environment variable."
        )
    return cuda_home


@functools.lru_cache(maxsize=1)
def get_cuda_include_dir_for_clang() -> dict[str, str]:
    """
    Get all include directories for the CUDA API.

    Note
    ----
    This function provides the include directory for clang -xcuda
    so that it compiles a CUDA file with the proper CUDA headers.

    This function raises a warning if a specific required CUDA API is missing.

    For CUDA 12, required pip packages are:
        cuda-toolkit[cudart,nvcc,curand,cccl]

        The include directory is:
        - cuda_runtime/include/
        - cuda_nvcc/include/
        - nvidia/curand/include/
        - cuda_cccl/include/

    For CUDA 13, required pip packages are:
        cuda-toolkit[cudart,crt,curand,cccl]

        The include directory is:
        - nvidia/cu13/include/

    Returns
    -------
    include_dirs: dict[str, str]
        The include directories for the CUDA API. The key is the name of the CUDA API,
        and the value is the include directory.
    """

    paths = {}

    if path := find_nvidia_header_directory("cudart"):
        paths["cudart"] = path
    else:
        warnings.warn(
            "Missing CUDA Runtime installation. Please install the CUDA Runtime."
        )

    if path := find_nvidia_header_directory("nvcc"):
        # Required for crt/host_defines.h in
        paths["nvcc"] = path
    else:
        warnings.warn("Missing NVCC installation. Please install the NVCC.")

    if path := find_nvidia_header_directory("curand"):
        # Required for curand_mtgp32_kernel.h in CUDA 12
        paths["curand"] = path
    else:
        warnings.warn("Missing CURAND installation. Please install the CURAND.")

    if path := find_nvidia_header_directory("cccl"):
        paths["cccl"] = path
    else:
        warnings.warn("Missing CCCL installation. Please install the CCCL.")

    return paths


def get_clang_resource_dir():
    clang_resource_file = (
        subprocess.check_output(["clang++", "-print-resource-dir"])
        .decode()
        .strip()
    )

    return clang_resource_file


def parse_declarations_from_source(
    source_file_path: str,
    files_to_retain: list[str],
    compute_capability: str,
    cudatoolkit_include_dirs: list[str] = [],
    cxx_standard: str = "gnu++17",
    additional_includes: list[str] = [],
    defines: list[str] = [],
    verbose: bool = True,
    bypass_parse_error: bool = False,
) -> Declarations:
    """Given a source file, parse all top-level declarations from it and return
    a ``Declarations`` object containing lists of declaration objects found in
    the source.

    Parameters
    ----------
    source_file_path : str
        The path to the source file to parse.

    files_to_retain : list[str]
        A list of file paths whose parsed declarations should be retained in the
        result. A header file usually references other headers. A semantically
        intact AST should, in theory, include all referenced headers. In
        practice, you may not need all of them. To retain only declarations
        from ``source_file_path``, pass ``[source_file_path]``.

    compute_capability : str
        The compute capability of the target GPU. e.g. "sm_70".

    cudatoolkit_include_dirs : list[str], optional
        The paths to the CUDA Toolkit include directories to override the default CUDA include
        directories. If not provided, ast_canopy will use cuda.pathfinder to find the CUDA include
        directories. If provided, the default CUDA include directories will be ignored.

    cxx_standard : str, optional
        The C++ standard to use. Default is "gnu++17".

    additional_includes : list[str], optional
        A list of additional include directories to search for headers.

    defines : list[str], optional
        A list of implicit defines that are passed to clangTooling via the
        "-D" flag.

    verbose : bool, optional
        If True, print stderr from the clang++ invocation.

    bypass_parse_error : bool, optional
        If True, bypass parse error and continue generating bindings.

    Returns
    -------
    Declarations
        See ``Declarations`` struct definition for details.
    """

    clang_verbose_flag = ["--verbose"] if verbose else []

    if not cudatoolkit_include_dirs:
        cudatoolkit_include_dirs = list(
            set(get_cuda_include_dir_for_clang().values())
        )
        cudatoolkit_include_dirs = [
            x for x in cudatoolkit_include_dirs if x is not None
        ]

    cuda_path = get_cuda_path_for_clang()

    if not os.path.exists(source_file_path):
        raise FileNotFoundError(f"File not found: {source_file_path}")

    for p in additional_includes:
        if not isinstance(p, str):
            raise TypeError(f"Additional include path must be a string: {p}")
        if p.startswith("-I"):
            raise ValueError(
                f"Additional include path must not start with -I: {p}"
            )
        if not os.path.exists(p):
            raise FileNotFoundError(f"Additional include path not found: {p}")

    _validate_compute_capability(compute_capability)

    clang_resource_file = get_clang_resource_dir()

    clang_search_paths = get_default_compiler_search_paths()

    define_flags = [f"-D{define}" for define in defines]

    # The include paths are ordered a below:
    # 1. clang resource file include directory (via -isystem flag)
    # 2. default compiler search paths (clang cuda wrapper headers)
    # 3. CUDA Toolkit include directories
    # 4. Additional include directories
    command_line_options = [
        "clang++",
        *clang_verbose_flag,
        "--cuda-device-only",
        "-xcuda",
        f"--cuda-path={cuda_path}",
        f"--cuda-gpu-arch={compute_capability}",
        f"-std={cxx_standard}",
        f"-isystem{clang_resource_file}/include/",
        *[f"-I{path}" for path in clang_search_paths],
        *paths_to_include_flags(cudatoolkit_include_dirs),
        *[f"-I{path}" for path in additional_includes],
        *define_flags,
        source_file_path,
    ]

    logger.debug(f"{command_line_options=}")
    if verbose:
        print(f"{command_line_options=}")

    decls = bindings.parse_declarations_from_command_line(
        command_line_options,
        files_to_retain,
        bypass_parse_error,
    )

    structs = [
        Struct.from_c_obj(c_obj, source_file_path) for c_obj in decls.records
    ]
    functions = [
        Function.from_c_obj(c_obj, source_file_path)
        for c_obj in decls.functions
    ]
    function_templates = [
        FunctionTemplate.from_c_obj(c_obj, source_file_path)
        for c_obj in decls.function_templates
    ]
    class_templates = [
        ClassTemplate.from_c_obj(c_obj, source_file_path)
        for c_obj in decls.class_templates
    ]

    return Declarations(
        structs,
        functions,
        function_templates,
        class_templates,
        decls.typedefs,
        decls.enums,
    )


def value_from_constexpr_vardecl(
    source: str,
    vardecl_name: str,
    compute_capability: str,
    cxx_standard: str = "gnu++17",
    verbose: bool = False,
) -> bindings.ConstExprVar | None:
    """Extract the value from a constexpr ``VarDecl`` with the given name.

    Parameters
    ----------
    source : str
        The source code to parse.

    vardecl_name : str
        The name of the constexpr variable declaration to extract the value from.

    compute_capability : str
        The compute capability of the target GPU. e.g. "sm_70".

    cxx_standard : str, optional
        The C++ standard to use. Default is "gnu++17".

    verbose : bool, optional
        If True, print the stderr from clang++ invocation.

    Returns
    -------
    ConstExprVar | None
        See ``ConstExprVar`` struct definition for details.
    """

    with tempfile.NamedTemporaryFile(mode="w") as f:
        f.write(source)
        f.flush()

        clang_resource_file = get_clang_resource_dir()

        clang_search_paths = get_default_compiler_search_paths()

        cudatoolkit_include_dirs = list(
            set(get_cuda_include_dir_for_clang().values())
        )
        cudatoolkit_include_dirs = [
            x for x in cudatoolkit_include_dirs if x is not None
        ]

        cuda_path = get_cuda_path_for_clang()

        command_line_options = [
            "clang++",
            "--cuda-device-only",
            "-xcuda",
            f"--cuda-path={cuda_path}",
            f"--cuda-gpu-arch={compute_capability}",
            f"-std={cxx_standard}",
            f"-isystem{clang_resource_file}/include/",
            *[f"-I{path}" for path in clang_search_paths],
            *paths_to_include_flags(cudatoolkit_include_dirs),
            f.name,
        ]

        with capture_fd(STREAMFD.STDERR) as cap:
            c_result = bindings.value_from_constexpr_vardecl(
                command_line_options, vardecl_name
            )

        werr = cap.snap()
        if werr and verbose:
            print(werr)

        result = (
            ConstExprVar.from_c_obj(c_result) if c_result is not None else None
        )
        return result


def _validate_compute_capability(compute_capability: Any):
    """Validate the compute capability string.

    Parameters
    ----------
    compute_capability : Any
        The compute capability string to validate.
    """
    if not isinstance(compute_capability, str):
        raise TypeError(
            f"Compute capability must be a string: {compute_capability}"
        )

    if not compute_capability.startswith("sm_"):
        raise ValueError(
            f"Compute capability must start with 'sm_': {compute_capability}"
        )
    if not compute_capability[3:].isdigit():
        raise ValueError(
            f"Compute capability must be in the form of 'sm_<compute_capability>': {compute_capability}"
        )
