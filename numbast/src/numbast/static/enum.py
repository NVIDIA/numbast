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
        """
        Initialize the renderer for a single C++ enum and derive its Python enum name.

        Parameters:
            decl (Enum): The parsed C++ enum declaration to render.
            enum_prefix_removal (list[str] | None): Prefix strings to remove from the C++ enum and enumerator names when deriving Python identifiers; empty list if None.

        Description:
            Stores the declaration and prefix-removal configuration, computes the Python enum name from the C++ name using the provided prefixes, and appends that Python name to the renderer's symbol list.
        """
        self._decl = decl
        self._enum_prefix_removal = enum_prefix_removal or []

        self._enum_name = _apply_prefix_removal(
            self._decl.name, self._enum_prefix_removal
        )

        self._enum_symbols.append(self._enum_name)

    def _render(self):
        """
        Render the stored C++ enum declaration into a Python IntEnum class and store the generated source.

        This method:
        - Ensures required imports for `IntEnum`, `IntEnumMember`, and `int64` are added to the renderer's import set.
        - Registers the mapping from the original C++ enum name to the computed Python enum name.
        - Applies configured prefix removal to each enumerator, formats enumerator lines, and assembles the final class text.
        - Writes the resulting Python class source into `self._python_rendered`.
        """
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

    def __init__(
        self, decls: list[Enum], enum_prefix_removal: list[str] | None = None
    ):
        """
        Initialize the renderer for a collection of C++ enum declarations.

        Parameters:
            decls (list[Enum]): The list of enum declarations to render.
            enum_prefix_removal (list[str] | None): Optional list of prefixes to remove from enum and enumerator names when generating Python bindings; defaults to an empty list.

        Notes:
            Initializes internal state and a list to accumulate per-enum rendered Python strings.
        """
        super().__init__(decls)
        self._decls = decls
        self._enum_prefix_removal = enum_prefix_removal or []

        self._python_rendered: list[str] = []

    def _render(self, with_imports):
        """
        Render all stored C++ enum declarations into Python binding source and store the result on the renderer instance.

        This populates self._python_rendered with each enum's rendered string and assembles the combined output into self._python_str. If with_imports is True, the module import block is included at the top of the assembled output.

        Parameters:
                with_imports (bool): If True, prepend the rendered import block to the assembled Python output.
        """
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
