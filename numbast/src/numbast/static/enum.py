# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from logging import getLogger, FileHandler
import tempfile
import os

from ast_canopy.pylibastcanopy import Enum

from numbast.static.renderer import BaseRenderer, get_rendered_imports
from numbast.static.types import register_enum_type_str
from numbast.utils import _apply_prefix_removal

file_logger = getLogger(f"{__name__}")
logger_path = os.path.join(tempfile.gettempdir(), "test.py")
file_logger.debug(f"Struct debug outputs are written to {logger_path}")
file_logger.addHandler(FileHandler(logger_path))


class StaticEnumRenderer(BaseRenderer):
    """Render a single C++ enum type.

    A C++ enum is trivially mapped into a python `IntEnum` class.
    """

    enum_template = """
class {enum_name}(IntEnum):
{enumerators}
"""
    enumerator_template = "    {enumerator} = {value}"

    def __init__(
        self, decl: Enum, enum_prefix_removal: list[str] | None = None
    ):
        self._decl = decl
        self._enum_prefix_removal = enum_prefix_removal or []

        self._enum_name = _apply_prefix_removal(
            self._decl.name, self._enum_prefix_removal
        )

        self._enum_symbols.append(self._enum_name)

    def _render(self):
        self.Imports.add("from enum import IntEnum")
        self.Imports.add("from numba.types import IntEnumMember")
        self.Imports.add("from numba.types import int64")

        register_enum_type_str(self._decl.name, self._enum_name)

        enumerators = []
        for enumerator, value in zip(
            self._decl.enumerators, self._decl.enumerator_values
        ):
            py_name = _apply_prefix_removal(
                enumerator, self._enum_prefix_removal
            )
            enumerators.append(
                self.enumerator_template.format(enumerator=py_name, value=value)
            )

        self._python_rendered = self.enum_template.format(
            enum_name=self._enum_name, enumerators="\n".join(enumerators)
        )


class StaticEnumsRenderer(BaseRenderer):
    """Create bindings for a collection of C++ Enums.

    Since enums creates a new C++ type. It should be invoked before making struct / function bindings.
    """

    def __init__(self, decls: list[Enum], enum_prefix_removal: list[str] = []):
        super().__init__(decls)
        self._decls = decls
        self._enum_prefix_removal = enum_prefix_removal

        self._python_rendered: list[str] = []

    def _render(self, with_imports):
        """Render python bindings for enums."""
        self._python_str = ""

        for decl in self._decls:
            SER = StaticEnumRenderer(decl, self._enum_prefix_removal)
            SER._render()
            self._python_rendered.append(SER._python_rendered)

        if with_imports:
            self._python_str += "\n" + get_rendered_imports()

        self._python_str += "\n" + "\n".join(self._python_rendered)

    def render_as_str(
        self,
        *,
        with_imports: bool,
        with_shim_stream: bool,
    ) -> str:
        """Return the final assembled bindings in script. This output should be final."""

        if with_shim_stream is True:
            raise ValueError("Enum renderer does not render shim functions.")

        self._render(with_imports)

        file_logger.debug(self._python_str)

        return self._python_str
