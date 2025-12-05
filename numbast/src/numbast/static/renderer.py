# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings

import numba
from numba.cuda.cudadrv.runtime import get_version
from numba.cuda.vector_types import vector_types

from numbast import __version__ as numbast_ver

from ast_canopy import __version__ as ast_canopy_ver


class BaseRenderer:
    SeparateRegistrySetup = """
typing_registry = TypingRegistry()
register = typing_registry.register
register_attr = typing_registry.register_attr
register_global = typing_registry.register_global
target_registry = TargetRegistry()
lower = target_registry.lower
lower_attr = target_registry.lower_getattr
lower_constant = target_registry.lower_constant
lower_cast = target_registry.lower_cast
"""

    KeyedStringIO = """
class _KeyedStringIO(io.StringIO):
    def __init__(self, *arg, **kwarg):
        super().__init__(*arg, *kwarg)
        self._keys = set()

    def write_with_key(self, key: str, value: str):
        if key in self._keys:
            return
        self._keys.add(key)
        self.write(value)

    def reset(self):
        self._keys.clear()
        self.seek(0)
"""

    Shim = """
{shim_defines}
{shim_include}
shim_prefix = shim_defines + \"\\n\" + shim_include
shim_stream = _KeyedStringIO()
shim_stream.write(shim_prefix)
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

    _nbtype_symbols: list[str] = []
    """List of new created Numba types to expose."""

    _record_symbols: list[str] = []
    """List of new record handles to expose."""

    _function_symbols: list[str] = []
    """List of new function handles to expose."""

    _enum_symbols: list[str] = []
    """List of new enum handles to expose."""

    def __init__(self, decl):
        """
        Initialize the BaseRenderer with a declaration and ensure required base imports are registered.

        Parameters:
            decl: The parsed declaration object the renderer will use to produce code.

        Notes:
            This constructor records "import numba" and "import io" in the class-level Imports set as part of instance initialization.
        """
        self.Imports.add("import numba")
        self.Imports.add("import io")
        self._decl = decl

    @classmethod
    def _try_import_numba_type(cls, typ: str):
        if typ in cls._imported_numba_types:
            return

        if typ == "__nv_bfloat16":
            cls.Imports.add("from numba.cuda.types import bfloat16")
            cls._imported_numba_types.add(typ)

        elif typ == "__nv_bfloat16_raw":
            cls.Imports.add(
                "from numba.cuda._internal.cuda_bf16 import _type_unnamed1405307 as bfloat16_raw_type"
            )
            cls._imported_numba_types.add(typ)

        elif typ in vector_types:
            # CUDA target specific types
            cls.Imports.add("from numba.cuda.vector_types import vector_types")
            cls.Imported_VectorTypes.append(typ)
            cls._imported_numba_types.add(typ)

        elif typ in numba.cuda.types.__dict__:
            cls.Imports.add(f"from numba.cuda.types import {typ}")
            cls._imported_numba_types.add(typ)

        else:
            warnings.warn(f"{typ} is not added to imports.")

    def render_as_str(
        self,
        *,
        with_imports: bool,
        with_shim_stream: bool,
    ) -> str:
        raise NotImplementedError()


def clear_base_renderer_cache():
    """
    Clear all class-level caches and exposed-symbol lists on BaseRenderer.

    This resets shared renderer state by removing all entries from the following BaseRenderer attributes:
    `Imports`, `Imported_VectorTypes`, `Includes`, `ShimFunctions`, `_imported_numba_types`,
    `_nbtype_symbols`, `_record_symbols`, `_function_symbols`, and `_enum_symbols`.
    """
    BaseRenderer.Imports.clear()
    BaseRenderer.Imported_VectorTypes.clear()
    BaseRenderer.Includes.clear()
    BaseRenderer.ShimFunctions.clear()
    BaseRenderer._imported_numba_types.clear()
    BaseRenderer._nbtype_symbols.clear()
    BaseRenderer._record_symbols.clear()
    BaseRenderer._function_symbols.clear()
    BaseRenderer._enum_symbols.clear()


def get_reproducible_info(
    config_rel_path: str, cmd: str, sbg_params: dict[str, str]
) -> str:
    """
    Produce a reproducible information header composed of commented lines documenting versions, the generation command, generator parameters, and the config path.

    Parameters:
        config_rel_path (str): Path to the generator configuration file relative to the generated binding file.
        cmd (str): The command line used to invoke the generation.
        sbg_params (dict[str, str]): Static binding generator parameters to record.

    Returns:
        str: A multi-line string where each line is prefixed with "# " and the block ends with a single trailing newline.
    """
    info = [
        f"Ast_canopy version: {ast_canopy_ver}",
        f"Numbast version: {numbast_ver}",
        f"Generation command: {cmd}",
        f"Static binding generator parameters: {sbg_params}",
        f"Config file path (relative to the path of the generated binding): {config_rel_path}",
        f"Cudatoolkit version: {get_version()}",
    ]

    commented = [f"# {x}" for x in info]

    return "\n".join(commented) + "\n"


def get_shim(
    shim_include: str,
    predefined_macros: list[str] = [],
    module_callbacks: dict[str, str] = {},
) -> str:
    """Render the code block for shim functions.

    This includes:
    1. Predefined macros used to runtime-compile the shim functions
       `shim_defines`
    2. The include path that declares functions used inside the shim functions
       `shim_include`
    3. `KeyedStringIO` - a string stream that only writes once to the stream
       per key
    4. An instance of `KeyedStringIO`, a module-level object that allows lower
       function writes shim functions to.
    5. Module callbacks setup and teardown if provided
    """
    defines_expanded = [f"#define {define}" for define in predefined_macros]
    defines = "\\n".join(defines_expanded)
    defines_py = f'shim_defines = "{defines}"'

    shim_include = f"shim_include = {shim_include}"

    shim_template = BaseRenderer.Shim

    # Add callback setting if provided
    callbacks_setup = ""
    if module_callbacks:
        setup_callback = module_callbacks.get("setup", "")
        teardown_callback = module_callbacks.get("teardown", "")

        if setup_callback:
            callbacks_setup += f"shim_obj.setup_callback = {setup_callback}\n"
        if teardown_callback:
            callbacks_setup += (
                f"shim_obj.teardown_callback = {teardown_callback}\n"
            )

    return (
        BaseRenderer.KeyedStringIO
        + "\n"
        + shim_template.format(
            shim_include=shim_include, shim_defines=defines_py
        )
        + "\n"
        + callbacks_setup
    )


def get_rendered_imports(additional_imports: list[str] = []) -> str:
    imports = "\n".join(BaseRenderer.Imports) + "\n"
    for imprt in additional_imports:
        imports += f"import {imprt}\n"

    imports += "\n" * 2
    for vty in BaseRenderer.Imported_VectorTypes:
        imports += f"{vty} = vector_types['{vty}']\n"

    return imports


def _get_nbtype_symbols() -> str:
    template = """
_NBTYPE_SYMBOLS = [{nbtype_symbols}]
"""

    symbols = BaseRenderer._nbtype_symbols
    quote_wrapped = [f'"{s}"' for s in symbols]
    concat = ",".join(quote_wrapped)
    code = template.format(nbtype_symbols=concat)
    return code


def _get_record_symbols() -> str:
    template = """
_RECORD_SYMBOLS = [{record_symbols}]
"""

    symbols = BaseRenderer._record_symbols
    quote_wrapped = [f'"{s}"' for s in symbols]
    concat = ",".join(quote_wrapped)
    code = template.format(record_symbols=concat)
    return code


def _get_function_symbols() -> str:
    """
    Render a Python code block that defines the _FUNCTION_SYMBOLS list from BaseRenderer._function_symbols.

    Returns:
        code (str): A string containing a Python assignment that defines `_FUNCTION_SYMBOLS` as a list of symbol names quoted and comma-separated.
    """
    template = """
_FUNCTION_SYMBOLS = [{function_symbols}]
"""

    symbols = BaseRenderer._function_symbols
    quote_wrapped = [f'"{s}"' for s in symbols]
    concat = ",".join(quote_wrapped)
    code = template.format(function_symbols=concat)
    return code


def _get_enum_symbols() -> str:
    """
    Generate a Python source snippet that defines the `_ENUM_SYMBOLS` list from BaseRenderer._enum_symbols.

    The returned string is a ready-to-insert code block where each enum name is quoted and placed into a Python list assigned to `_ENUM_SYMBOLS`.

    Returns:
        A string containing Python code that assigns `_ENUM_SYMBOLS` to a list of quoted enum symbol names (e.g., `_ENUM_SYMBOLS = ["A","B"]`).
    """
    template = """
_ENUM_SYMBOLS = [{enum_symbols}]
"""

    symbols = BaseRenderer._enum_symbols
    quote_wrapped = [f'"{s}"' for s in symbols]
    concat = ",".join(quote_wrapped)
    code = template.format(enum_symbols=concat)
    return code


def get_all_exposed_symbols() -> str:
    """
    Produce the code block that defines and exposes all symbol name lists and the module __all__.

    Returns:
        A string containing Python code that defines `_NBTYPE_SYMBOLS`, `_RECORD_SYMBOLS`, `_FUNCTION_SYMBOLS`, `_ENUM_SYMBOLS`
        and an `__all__` list that is the concatenation of those symbol lists.
    """

    nbtype_symbols = _get_nbtype_symbols()
    record_symbols = _get_record_symbols()
    function_symbols = _get_function_symbols()
    enum_symbols = _get_enum_symbols()

    all_symbols = f"""
{nbtype_symbols}
{record_symbols}
{function_symbols}
{enum_symbols}
__all__ = _NBTYPE_SYMBOLS + _RECORD_SYMBOLS + _FUNCTION_SYMBOLS + _ENUM_SYMBOLS
"""

    return all_symbols


def registry_setup(use_separate_registry: bool) -> str:
    """Get the registry setup code.

    In Numba-CUDA, builtin registries are created a cudadecl and cudaimpl.
    By default, Numbast bindings inject the registries into the existing
    typing and target context. When use_separate_registry is True, Numbast
    bindings create a new typing and target registry. User should add the
    registries to the typing and target context manually.
    """
    if use_separate_registry:
        BaseRenderer.Imports.add(
            "from numba.cuda.typing.templates import Registry as TypingRegistry"
        )
        BaseRenderer.Imports.add(
            "from numba.cuda.core.imputils import Registry as TargetRegistry"
        )
        return BaseRenderer.SeparateRegistrySetup
    else:
        BaseRenderer.Imports.add("from numba.cuda.cudadecl import register")
        BaseRenderer.Imports.add(
            "from numba.cuda.cudadecl import register_global"
        )
        BaseRenderer.Imports.add(
            "from numba.cuda.cudadecl import register_attr"
        )
        BaseRenderer.Imports.add("from numba.cuda.cudaimpl import lower")
        BaseRenderer.Imports.add("from numba.cuda.cudaimpl import lower_attr")
        BaseRenderer.Imports.add(
            "from numba.cuda.cudaimpl import lower_constant"
        )
        BaseRenderer.Imports.add("from numba.cuda.extending import lower_cast")
        return ""
