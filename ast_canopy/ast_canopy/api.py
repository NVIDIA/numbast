# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import shutil
import os
import tempfile
import logging
from typing import Optional, Any
from dataclasses import dataclass

from numba.cuda.cuda_paths import get_cuda_paths, get_cuda_home

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


def get_default_cuda_path() -> Optional[str]:
    """Return the path to the default CUDA home directory."""

    # Allow overriding from os.environ
    if home := get_cuda_home():
        return home

    by, nvvm_path = get_cuda_paths()["nvvm"]
    if nvvm_path is not None:
        # In the form of $CUDA_HOME/nvvm/lib64/libnvvm.so, go up 3 levels for cuda home.
        cuda_home = os.path.dirname(os.path.dirname(os.path.dirname(nvvm_path)))

        if os.path.exists(cuda_home):
            logger.info(f"Found CUDA home: {cuda_home}")
            return cuda_home

    return None


def get_default_nvcc_path() -> Optional[str]:
    """Return the path to the default NVCC compiler binary."""
    by, nvvm_path = get_cuda_paths()["nvvm"]

    if nvvm_path is None:
        return shutil.which("nvcc")

    root = os.path.dirname(os.path.dirname(nvvm_path))
    nvcc_path = os.path.join(root, "bin", "nvcc")

    if os.path.exists(nvcc_path):
        logger.info(f"Found NVCC path: {nvcc_path}")
        return nvcc_path

    return None


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
    """Compile an empty CUDA file with NVCC and extract its default include path.

    ast_canopy depends on a healthy CUDA environment to function. If NVCC fails
    to preprocess an empty CUDA file, this function raises a RuntimeError.
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
        include_path = (
            s[0].lstrip("#$ INCLUDES=").strip().strip('"').lstrip("-I")
        )
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
    cudatoolkit_include_dir: str | None = None,
    cxx_standard: str = "c++17",
    additional_includes: list[str] = [],
    defines: list[str] = [],
    verbose: bool = False,
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

    cccl_root : str, optional
        The root directory of the CCCL project. If not provided, CCCL from the
        default CTK headers is used.

    cudatoolkit_include_dir : str, optional
        The path to the CUDA Toolkit include directory. If not provided, the default CUDA include
        directory is used.

    cxx_standard : str, optional
        The C++ standard to use. Default is "c++17".

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

    if cudatoolkit_include_dir is None:
        cudatoolkit_include_dir = get_default_cuda_compiler_include()

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
        subprocess.check_output(["clang++", "-print-resource-dir"])
        .decode()
        .strip()
    )

    clang_search_paths = get_default_compiler_search_paths()

    def custom_cuda_home() -> list[str]:
        cuda_path = get_default_cuda_path()
        if cuda_path:
            return [f"--cuda-path={cuda_path}"]
        else:
            return []

    define_flags = [f"-D{define}" for define in defines]

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
    cxx_standard: str = "c++17",
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
        The C++ standard to use. Default is "c++17".

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

        clang_resource_file = (
            subprocess.check_output(["clang++", "-print-resource-dir"])
            .decode()
            .strip()
        )

        clang_search_paths = get_default_compiler_search_paths()

        def custom_cuda_home() -> list[str]:
            cuda_path = get_default_cuda_path()
            if cuda_path:
                return [f"--cuda-path={cuda_path}"]
            else:
                return []

        cudatoolkit_include_dir: str = get_default_cuda_compiler_include()
        command_line_options = [
            "clang++",
            "--cuda-device-only",
            "-xcuda",
            f"--cuda-gpu-arch={compute_capability}",
            *custom_cuda_home(),
            f"-std={cxx_standard}",
            f"-isystem{clang_resource_file}/include/",
            *[f"-I{path}" for path in clang_search_paths],
            f"-I{cudatoolkit_include_dir}",
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
