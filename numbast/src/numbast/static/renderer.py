# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


class BaseRenderer:
    Prefix = """
from pynvjitlink.patch import patch_numba_linker
patch_numba_linker()
"""

    Imports: set[str] = set()
    """Empty set to be filled later. One element stands for one line of import."""

    MemoryShimWriterTemplate = """
c_ext_shim_source = CUSource(\"""{shim_funcs}\""")
"""

    ShimFunctions: list[str] = []

    includes_template = "#include <{header_path}>"
    """Template for including a header file."""

    Includes: set[str] = set()
    """includes to add in c extension shims."""

    def __init__(self, decl):
        self._decl = decl

        self._imported_numba_types = set()

    def _render_typing(self):
        pass

    def _render_data_model(self):
        pass

    def _render_lowering(self):
        pass

    def _render_decl_device(self):
        pass

    def _render_shim_function(self, decl):
        pass

    def _render_python_api(self):
        pass

    def _import_numba_type(self, typ: str):
        if typ in self._imported_numba_types:
            return

        self.Imports.add(f"from numba.types import {typ}")
        self._imported_numba_types.add(typ)

    def render_as_str(self, *, with_imports: bool, with_shim_functions: bool) -> str:
        raise NotImplementedError()


def get_rendered_imports() -> str:
    return "\n".join(BaseRenderer.Imports)


def get_rendered_shims() -> str:
    includes = "\n".join(BaseRenderer.Includes)
    return BaseRenderer.MemoryShimWriterTemplate.format(
        shim_funcs=includes + "\n" + "\n".join(BaseRenderer.ShimFunctions)
    )
