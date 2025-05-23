# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from logging import getLogger, FileHandler
import tempfile
import os

from pylibastcanopy import Enum

from numbast.static.renderer import BaseRenderer, get_rendered_imports
from numbast.static.types import register_enum_type_str

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

    def __init__(self, decl: Enum):
        self._decl = decl

    def _render(self):
        self.Imports.add("from enum import IntEnum")

        register_enum_type_str(self._decl.name)

        enumerators = []
        for enumerator, value in zip(
            self._decl.enumerators, self._decl.enumerator_values
        ):
            enumerators.append(
                self.enumerator_template.format(
                    enumerator=enumerator, value=value
                )
            )

        self._python_rendered = self.enum_template.format(
            enum_name=self._decl.name, enumerators="\n".join(enumerators)
        )


class StaticEnumsRenderer(BaseRenderer):
    """Create bindings for a collection of C++ Enums.

    Since enums creates a new C++ type. It should be invoked before making struct / function bindings.
    """

    def __init__(self, decls: list[Enum]):
        super().__init__(decls)
        self._decls = decls

        self._python_rendered = []

    def _render(self, require_pynvjitlink, with_imports):
        """Render python bindings for enums."""
        self._python_str = ""

        for decl in self._decls:
            SER = StaticEnumRenderer(decl)
            SER._render()
            self._python_rendered.append(SER._python_rendered)

        if with_imports:
            self._python_str += "\n" + get_rendered_imports()

        if require_pynvjitlink:
            self._python_str += "\n" + self.Pynvjitlink_guard

        self._python_str += "\n" + "\n".join(self._python_rendered)

    def render_as_str(
        self,
        *,
        require_pynvjitlink: bool,
        with_imports: bool,
        with_shim_stream: bool,
    ) -> str:
        """Return the final assembled bindings in script. This output should be final."""

        if with_shim_stream is True:
            raise ValueError("Enum renderer does not render shim functions.")

        self._render(require_pynvjitlink, with_imports)

        file_logger.debug(self._python_str)

        return self._python_str
