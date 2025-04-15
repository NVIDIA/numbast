# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings

import numba
from numba.cuda.cudadrv.runtime import get_version
from numba.cuda.vector_types import vector_types

from numbast import __version__ as numbast_ver

from ast_canopy import __version__ as ast_canopy_ver
from ast_canopy.api import get_default_cuda_path


class BaseRenderer:
    Pynvjitlink_guard = """
import importlib

if importlib.util.find_spec("pynvjitlink"):
    numba.config.CUDA_ENABLE_PYNVJITLINK = True
else:
    raise RuntimeError("Numbast generated bindings require pynvjitlink.")
"""

    Shim = """
{shim_include}
shim_stream = io.StringIO()
shim_stream.write(shim_include)
shim_obj = CUSource(shim_stream)
"""

    Imports: set[str] = set()
    """One element stands for one line of python import."""

    Imported_VectorTypes: list[str] = []
    """Numba.cuda vector_types imported. Handled in _get_rendered_imports."""

    _imported_numba_types = set()
    """Set of imported numba type in strings."""

    MemoryShimWriterTemplate = """
c_ext_shim_source = CUSource(\"""{shim_funcs}\""")
"""

    ShimFunctions: list[str] = []

    _imported_numba_types = set()
    """Set of imported numba type in strings."""

    includes_template = "#include <{header_path}>"
    """Template for including a header file."""

    Includes: set[str] = set()
    """includes to add in c extension shims."""

    def __init__(self, decl):
        self.Imports.add("import numba")
        self.Imports.add("import io")
        self._decl = decl

    @classmethod
    def _try_import_numba_type(cls, typ: str):
        if typ in cls._imported_numba_types:
            return

        if typ in vector_types:
            # CUDA target specific types
            cls.Imports.add("from numba.cuda.vector_types import vector_types")
            cls.Imported_VectorTypes.append(typ)
            cls._imported_numba_types.add(typ)

        elif typ in numba.types.__dict__:
            cls.Imports.add(f"from numba.types import {typ}")
            cls._imported_numba_types.add(typ)

        else:
            warnings.warn(f"{typ} is not added to imports.")

    def render_as_str(
        self,
        *,
        with_prefix: bool,
        with_imports: bool,
        with_shim_functions: bool,
    ) -> str:
        raise NotImplementedError()


def clear_base_renderer_cache():
    BaseRenderer.Imports = set()
    BaseRenderer.Imported_VectorTypes = []
    BaseRenderer.Includes = set()
    BaseRenderer.ShimFunctions = []
    BaseRenderer._imported_numba_types = set()


def get_reproducible_info(
    config_rel_path: str, cmd: str, sbg_params: dict[str, str]
) -> str:
    info = [
        f"Ast_canopy version: {ast_canopy_ver}",
        f"Numbast version: {numbast_ver}",
        f"Generation command: {cmd}",
        f"Static binding generator parameters: {sbg_params}",
        f"Config file path (relative to the path of the generated binding): {config_rel_path}",
        f"Cudatoolkit version: {get_version()}",
        f"Default CUDA_HOME path: {get_default_cuda_path()}",
    ]

    commented = [f"# {x}" for x in info]

    return "\n".join(commented) + "\n"


def get_pynvjitlink_guard() -> str:
    return BaseRenderer.Pynvjitlink_guard


def get_shim_stream_obj(shim_include: str) -> str:
    shim_include = f"shim_include = {shim_include}"
    return BaseRenderer.Shim.format(shim_include=shim_include)


def get_rendered_imports(additional_imports: list[str] = []) -> str:
    imports = "\n".join(BaseRenderer.Imports) + "\n"
    for imprt in additional_imports:
        imports += f"import {imprt}\n"

    imports += "\n" * 2
    for vty in BaseRenderer.Imported_VectorTypes:
        imports += f"{vty} = vector_types['{vty}']\n"

    return imports
