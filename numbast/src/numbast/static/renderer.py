# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings

import numba
from numba.cuda.vector_types import vector_types


class BaseRenderer:
    Prefix = """
import numba
numba.config.CUDA_ENABLE_PYNVJITLINK = True
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
        self, *, with_prefix: bool, with_imports: bool, with_shim_functions: bool
    ) -> str:
        raise NotImplementedError()


def clear_base_renderer_cache():
    BaseRenderer.Imports = set()
    BaseRenderer.Imported_VectorTypes = []
    BaseRenderer.Includes = set()
    BaseRenderer.ShimFunctions = []
    BaseRenderer._imported_numba_types = set()


def get_prefix() -> str:
    return BaseRenderer.Prefix


def get_rendered_imports() -> str:
    imports = "\n".join(BaseRenderer.Imports)
    imports += "\n" * 2
    for vty in BaseRenderer.Imported_VectorTypes:
        imports += f"{vty} = vector_types['{vty}']\n"

    return imports
