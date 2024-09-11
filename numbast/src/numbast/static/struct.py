# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from textwrap import indent
from logging import getLogger, FileHandler
import tempfile

from numba.types import Type
from numba.core.datamodel.models import StructModel, PrimitiveModel

from pylibastcanopy import access_kind
from ast_canopy.decl import Struct, StructMethod

from numbast.static.renderer import (
    BaseRenderer,
)
from numbast.types import to_numba_type
from numbast.utils import deduplicate_overloads

file_logger = getLogger(f"{__name__}")
logger_path = os.path.join(tempfile.gettempdir(), "test.py")
file_logger.debug(f"Struct debug outputs are written to {logger_path}")
file_logger.addHandler(FileHandler(logger_path))


class StaticStructCtorRenderer(BaseRenderer):
    """Renderer for a single struct constructor.

    Parameters
    ----------
    struct_name: str
        Name of the struct.
    struct_type_class: str
        Name of the new numba type class in the binding script.
    struct_type_name: str
        Name of the instantiated numba type in the binding script.
    ctor_decl: ast_canopy.StructMethod
        Declaration of the constructor.
    """

    struct_ctor_decl_device_template = """
{struct_name}_ctor_decl = declare_device(
    '{unique_shim_name}',
    int32(
        CPointer({struct_name}_type),
        {pointer_wrapped_param_types}
    )
)
    """

    struct_ctor_device_caller_template = """
def {struct_device_caller_name}({nargs}):
    return {struct_name}_ctor_decl({nargs})
    """

    struct_ctor_c_ext_shim_template = """
extern "C" __device__ int
{unique_shim_name}(int &ignore, {struct_name} *self {arglist}) {{
    new (self) {struct_name}({args});
    return 0;
}}
    """

    struct_ctor_lowering_template = """
@lower({struct_name}, {param_types})
def ctor_impl(context, builder, sig, args):
    selfptr = builder.alloca(context.get_value_type({struct_type_name}), name="selfptr")
    argptrs = [builder.alloca(context.get_value_type(arg)) for arg in sig.args]
    for ptr, ty, arg in zip(argptrs, sig.args, args):
        builder.store(arg, ptr, align=getattr(ty, "alignof_", None))

    context.compile_internal(
        builder,
        {struct_device_caller_name},
        signature(
            int32,
            CPointer({struct_type_name}),
            {pointer_wrapped_args}
        ),
        (selfptr, *argptrs),
    )
    return builder.load(selfptr, align=getattr({struct_type_name}, "alignof_", None))
    """

    lower_overload_scope_template = """
def _{struct_name}_{param_names}_lower():
    {body}

_{struct_name}_{param_names}_lower()
"""

    struct_ctor_signature_template = "signature({struct_name}_type, {arglist})"

    _nb_param_types: list[Type]
    """A list of parameter types converted from C++ types to Numba types.
    """

    _nb_param_types_str: str
    """Concatenated string of argument types in Numba type for this constructor.
    e.g. "int32, int32, CPointer(float32)"
    """

    def __init__(
        self,
        struct_name: str,
        struct_type_class: str,
        struct_type_name: str,
        ctor_decl: StructMethod,
    ):
        self._struct_name = struct_name
        self._struct_type_class = struct_type_class
        self._struct_type_name = struct_type_name
        self._ctor_decl = ctor_decl

        # Cache the list of parameter types represented as Numba types
        self._nb_param_types = [
            to_numba_type(arg.unqualified_non_ref_type_name)
            for arg in ctor_decl.param_types
        ]
        self._nb_param_types_str = ", ".join(map(str, self._nb_param_types))

        # Cache the list of parameter types wrapped in pointer types.
        def wrap_pointer(typ):
            return f"CPointer({typ})"

        _pointer_wrapped_param_types = [
            wrap_pointer(typ) for typ in self._nb_param_types
        ]
        self._pointer_wrapped_param_types_str = ", ".join(_pointer_wrapped_param_types)

        # Cache the list of parameter types in C++ pointer types
        c_ptr_arglist = ", ".join(
            f"{arg.type_.unqualified_non_ref_type_name}* {arg.name}"
            for arg in self._ctor_decl.params
        )
        if c_ptr_arglist:
            c_ptr_arglist = ", " + c_ptr_arglist

        self._c_ext_argument_pointer_types = c_ptr_arglist

        # Cache the list of dereferenced arguments
        self._deref_args_str = ", ".join(
            "*" + arg.name for arg in self._ctor_decl.params
        )

        # Cache the unique shim name
        shim_func_name = f"__{self._struct_name}__{self._ctor_decl.mangled_name}"
        self._deduplicated_shim_name = deduplicate_overloads(shim_func_name)

        # Underscore separated names of parameters
        self._nb_param_str_concat = "_".join(map(str, self._nb_param_types))
        if not self._nb_param_str_concat:
            self._nb_param_str_concat = "void"

        # device caller name
        self._device_caller_name = f"{self._struct_name}_device_caller"

    def _render_decl_device(self):
        """Render codes that declares a foreign function for this constructor in Numba."""

        self.Imports.add("from numba.cuda import declare_device")
        self.Imports.add("from numba.core.typing import signature")
        # All arguments are passed by pointers in C-CPP shim interop
        self.Imports.add("from numba.types import CPointer")
        # Numba ABI returns int32 for exception codes
        self.Imports.add("from numba.types import int32")

        decl_device_rendered = self.struct_ctor_decl_device_template.format(
            struct_name=self._struct_name,
            unique_shim_name=self._deduplicated_shim_name,
            pointer_wrapped_param_types=self._pointer_wrapped_param_types_str,
        )

        nargs = [f"arg_{i}" for i in range(len(self._ctor_decl.params) + 1)]
        nargs_str = ", ".join(nargs)

        device_caller_rendered = self.struct_ctor_device_caller_template.format(
            struct_device_caller_name=self._device_caller_name,
            nargs=nargs_str,
            struct_name=self._struct_name,
        )

        self._decl_device_rendered = (
            decl_device_rendered + "\n" + device_caller_rendered
        )

    def _render_shim_function(self):
        """Render external C shim functions for this struct constructor."""

        self._c_ext_shim_rendered = self.struct_ctor_c_ext_shim_template.format(
            unique_shim_name=self._deduplicated_shim_name,
            struct_name=self._struct_name,
            arglist=self._c_ext_argument_pointer_types,
            args=self._deref_args_str,
        )

        self.ShimFunctions.append(self._c_ext_shim_rendered)

    def _render_lowering(self):
        """Render lowering codes for this struct constructor."""

        self.Imports.add("from numba.cuda.cudaimpl import lower")

        self._lowering_rendered = self.struct_ctor_lowering_template.format(
            struct_name=self._struct_name,
            param_types=self._nb_param_types_str,
            struct_type_name=self._struct_type_name,
            struct_device_caller_name=self._device_caller_name,
            pointer_wrapped_args=self._pointer_wrapped_param_types_str,
        )

    def _render(self):
        """Render FFI, lowering and C shim functions of the constructor.

        Note that the typing still needs to be handled on a higher layer.
        """

        self._render_decl_device()
        self._render_shim_function()
        self._render_lowering()

        lower_body = indent(
            self._decl_device_rendered + "\n" + self._lowering_rendered,
            prefix="    ",
            predicate=lambda x: True,
        )

        self._python_rendered = self.lower_overload_scope_template.format(
            struct_name=self._struct_name,
            param_names=self._nb_param_str_concat,
            body=lower_body,
        )

        self._c_rendered = self._c_ext_shim_rendered

    @property
    def numba_param_types(self):
        """Parameter types of the constructor in Numba types."""
        return self._nb_param_types

    @property
    def signature_str(self):
        """Numba.signature string of the constructor's signature."""
        return self.struct_ctor_signature_template.format(
            struct_name=self._struct_name,
            arglist=self._nb_param_types_str,
        )


class StaticStructCtorsRenderer(BaseRenderer):
    """Renderer for all constructors of a struct.

    Parameters
    ----------

    ctor_decls: list[StructMethod]
        A list of constructor declarations.
    struct_name: str
        Name of the struct.
    struct_type_class: str
        Name of the new numba type class in the binding script.
    struct_type_name: str
        Name of the instantiated numba type in the binding script.
    """

    struct_ctor_template_typing_template = """
@register
class {struct_name}_ctor_template(ConcreteTemplate):
    key = {struct_name}
    cases = [{signatures}]

register_global({struct_name}, Function({struct_name}_ctor_template))
"""

    def __init__(
        self,
        ctor_decls: list[StructMethod],
        struct_name,
        struct_type_class,
        struct_type_name,
    ):
        self._ctor_decls = ctor_decls
        self._struct_name = struct_name
        self._struct_type_class = struct_type_class
        self._struct_type_name = struct_type_name

        self._python_rendered = ""
        self._c_rendered = ""

    def _render_typing(self, signature_strs: list[str]):
        """Renders the typing of the constructors.

        Parameter
        ---------
        signature_strs: list[str]
            A list of `numba.signature` strings containing all overloads of
            the constructors.
        """

        self.Imports.add("from numba.cuda.cudadecl import register")
        self.Imports.add("from numba.cuda.cudadecl import register_global")
        self.Imports.add("from numba.core.typing.templates import ConcreteTemplate")
        self.Imports.add("from numba.types import Function")

        signatures_str = ", ".join(signature_strs)

        self._struct_ctor_typing_rendered = (
            self.struct_ctor_template_typing_template.format(
                struct_name=self._struct_name, signatures=signatures_str
            )
        )

    def _render(self):
        """Render all struct constructors."""

        signatures: list[str] = []
        for ctor_decl in self._ctor_decls:
            renderer = StaticStructCtorRenderer(
                struct_name=self._struct_name,
                struct_type_class=self._struct_type_class,
                struct_type_name=self._struct_type_name,
                ctor_decl=ctor_decl,
            )
            renderer._render()

            self._python_rendered += renderer._python_rendered
            self._c_rendered += renderer._c_rendered

            signatures.append(renderer.signature_str)

        self._render_typing(signatures)

        self._python_rendered += self._struct_ctor_typing_rendered

    @property
    def python_rendered(self) -> str:
        """The python script that contains the bindings to all constructors."""
        return self._python_rendered

    @property
    def c_rendered(self) -> str:
        """The C program that contains the shim functions to all constructors."""
        return self._c_rendered


class StaticStructRenderer(BaseRenderer):
    """Renderer that renders bindings to a single CUDA C++ struct.

    Parameters
    ----------
    decl: Struct
        Declaration of the struct.
    parent_type: type
        The parent Numba type of the new struct type created.
    data_model: type
        The data model of the new struct type.
    header_path: os.PathLike
        Path to the header that contains the declaration of the struct.
    aliases: list[str], optional
        TODO: If the struct has other aliases, specify them here. Numbast creates
        aliased objects that references the original python API object.
    """

    typing_template = """
# Typing for {struct_name}
class {struct_name}_type_class({parent_type}):
    def __init__(self):
        super().__init__(name="{struct_name}")
        self.alignof_ = {struct_alignof}
        self.bitwidth = {struct_sizeof} * 8

{struct_name}_type = {struct_name}_type_class()
"""

    python_api_template = """
# Make Python API for struct
{struct_name} = type("{struct_name}", (), {{"_nbtype": {struct_name}_type}})
"""

    primitive_data_model_template = """
@register_model({struct_name}_type_class)
class {struct_name}_model(PrimitiveModel):
    def __init__(self, dmm, fe_type):
        be_type = ir.IntType(fe_type.bitwidth)
        super({struct_name}_model, self).__init__(dmm, fe_type, be_type)
"""

    struct_data_model_template = """
@register_model({struct_name}_type_class)
class {struct_name}_model(StructModel):
    def __init__(self, dmm, fe_type):
        members = [{member_types_tuples}]
        super().__init__(dmm, fe_type, members)
"""

    resolve_methods_template = """
    def resolve_{attr_name}(self, obj):
        return {numba_type}
"""

    make_attribute_wrappers_template = """
make_attribute_wrapper({struct_name}_type_class, "{attr_name}", "{attr_name}")
"""

    struct_attribute_typing_template = """
@register_attr
class {struct_name}_attr(AttributeTemplate):
    key = {struct_name}

    {resolve_methods}

{make_attribute_wrappers}
"""

    _parent_type_str: str
    """Qualified name of parent type."""

    def __init__(
        self,
        decl: Struct,
        parent_type: type,
        data_model: type,
        header_path: os.PathLike,
        aliases: list[str] = [],
    ):
        super().__init__(decl)
        self._struct_name = decl.name
        self._aliases = aliases
        self._parent_type = parent_type
        self._data_model = data_model

        self.Imports.add(f"from numba.types import {self._parent_type.__qualname__}")
        self._parent_type_str = self._parent_type.__qualname__

        self.Imports.add(
            f"from {self._data_model.__module__} import {self._data_model.__qualname__}"
        )
        self._data_model_str = self._data_model.__qualname__

        self._struct_type_class_name = f"{self._struct_name}_type_class"
        self._struct_type_name = f"{self._struct_name}_type"

        self._header_path = header_path

    def _render_typing(self):
        """Render typing of the struct."""

        self._typing_rendered = self.typing_template.format(
            struct_type_name=self._struct_name,
            parent_type=self._parent_type_str,
            struct_name=self._struct_name,
            struct_alignof=self._decl.alignof_,
            struct_sizeof=self._decl.sizeof_,
        )

    def _render_python_api(self):
        """Render the Python API object of the struct.

        This is the python handle to use it in Numba kernels.
        """
        self._python_api_rendered = self.python_api_template.format(
            struct_name=self._struct_name
        )

    def _render_data_model(self):
        """Renders the data model of the struct."""

        self.Imports.add("from numba.core.extending import register_model")

        if self._data_model == PrimitiveModel:
            self._data_model_rendered = self.primitive_data_model_template.format(
                struct_type_name=self._struct_name,
                struct_name=self._struct_name,
            )
        elif self._data_model == StructModel:
            member_types_tuples = [
                (f.name, to_numba_type(f.type_.unqualified_non_ref_type_name))
                for f in self._decl.fields
            ]

            for _, nbty in member_types_tuples:
                self.Imports.add(f"from numba.types import {nbty}")

            member_types_tuples_strs = [
                f"('{name}', {ty})" for name, ty in member_types_tuples
            ]

            member_types_str = ", ".join(member_types_tuples_strs)

            self._data_model_rendered = self.struct_data_model_template.format(
                struct_type_name=self._struct_name,
                struct_name=self._struct_name,
                member_types_tuples=member_types_str,
            )

    def _render_struct_attr(self):
        """Renders the typings of the struct attributes."""

        if self._data_model == StructModel:
            self.Imports.add("from numba.cuda.cudadecl import register_attr")
            self.Imports.add(
                "from numba.core.typing.templates import AttributeTemplate"
            )
            self.Imports.add("from numba.core.extending import make_attribute_wrapper")

            public_fields = [
                f for f in self._decl.fields if f.access == access_kind.public_
            ]

            resolve_methods = []
            attribute_wrappers = []
            for field in public_fields:
                resolve_methods.append(
                    self.resolve_methods_template.format(
                        attr_name=field.name,
                        numba_type=to_numba_type(
                            field.type_.unqualified_non_ref_type_name
                        ),
                    )
                )
                attribute_wrappers.append(
                    self.make_attribute_wrappers_template.format(
                        struct_name=self._struct_name,
                        attr_name=field.name,
                    )
                )

            resolve_methods_str = "\n".join(resolve_methods)
            attribute_wrappers_str = "\n".join(attribute_wrappers)

            self._struct_attr_typing_rendered = (
                self.struct_attribute_typing_template.format(
                    struct_name=self._struct_name,
                    resolve_methods=resolve_methods_str,
                    make_attribute_wrappers=attribute_wrappers_str,
                )
            )

    def _render_struct_ctors(self):
        """Render constructors of the struct."""
        static_ctors_renderer = StaticStructCtorsRenderer(
            struct_name=self._struct_name,
            struct_type_class=self._struct_type_class_name,
            struct_type_name=self._struct_type_name,
            ctor_decls=self._decl.methods,
        )
        static_ctors_renderer._render()

        self._struct_ctors_python_rendered = static_ctors_renderer.python_rendered
        self._struct_ctors_c_rendered = static_ctors_renderer.c_rendered

    def render_python(self) -> tuple[set[str], str]:
        """Renders the python portion of the bindings.

        At the end of the day, all artifacts are in python scripts. However,
        there are shim functions that are in C language stored as a plain
        string in Python. This function only renders the pure-python parts
        of the bindings. This includes typing, lowering, FFI declarations
        and python imports.

        Return
        ------
        imports_and_bindings: tuple[set[str], str]
            A tuple. The first element of the tuple is a set of import strings
            required to run the bindings. The second element of the tuple
            is the concatenated bindings script.
        """
        self.Imports.add("from numba.cuda import CUSource")

        self._render_typing()
        self._render_python_api()
        self._render_data_model()
        self._render_struct_attr()
        self._render_struct_ctors()

        self._python_rendered = f"""
{self._typing_rendered}
{self._python_api_rendered}
{self._data_model_rendered}
{self._struct_attr_typing_rendered}
{self._struct_ctors_python_rendered}
"""
        return self.Imports, self._python_rendered

    def render_c(self) -> tuple[set[str], str]:
        """Renders the C shim functions of the bindings.

        At the end of the day, all artifacts are in python scripts. However,
        there are shim functions that are in C language stored as a plain
        string in Python. This function renders these shim function codes.

        Return
        ------
        includes_and_shims: tuple[set[str], str]
            A tuple. The first element of the tuple is a set of include strings
            required to compile the shim function. The second element of the tuple
            is the concatenated shim function C program.
        """
        self.Includes.add(self.includes_template.format(header_path=self._header_path))

        self._c_ext_merged_shim = "\n".join([self._struct_ctors_c_rendered])

        return self.Includes, self._c_ext_merged_shim


class StaticStructsRenderer(BaseRenderer):
    """Render a collection of CUDA struct declcarations.

    Parameters
    ----------
    decls: list[Struct]
        A list of CUDA struct declarations.
    specs: dict[str, tuple[type, type]]
        A dictionary mapping the name of the structs to a tuple of
        Numba parent type, data model and header path. If unspecified,
        use `numba.types.Type`, `StructModel` and `default_header` by default.
    header_path: str
        The path to the header that contains the cuda struct declaration.
    """

    _python_rendered: list[tuple[set[str], str]]
    """The strings containing rendered python scripts of the struct bindings. Minus the C shim functions. The
    first element of the tuple are the imports necessary to configure the numba typings and lowerings. The second
    elements are the typing and lowering of the struct.
    """

    _c_rendered: list[tuple[set[str], str]]
    """The strings containing rendered C shim functions of the struct bindings. The first element of the tuple
    are the C includes that contains the declaration of the CUDA C++ struct. The second element are the shim function
    strings.
    """

    def __init__(
        self,
        decls: list[Struct],
        specs: dict[str, tuple[type, type, os.PathLike]],
        default_header: os.PathLike | None = None,
    ):
        self._decls = decls
        self._specs = specs
        self._default_header = default_header

        self._python_rendered = []
        self._c_rendered = []

    def _render(self, with_imports: bool):
        """Render all structs in `decls`."""
        for decl in self._decls:
            name = decl.name
            nb_ty, nb_datamodel, header_path = self._specs.get(
                name, (Type, StructModel, self._default_header)
            )
            if header_path is None:
                raise ValueError(f"CUDA struct {name} does not provide a header path.")

            SSR = StaticStructRenderer(decl, nb_ty, nb_datamodel, header_path)

            self._python_rendered.append(SSR.render_python())
            self._c_rendered.append(SSR.render_c())

        imports = set()
        python_rendered = []
        for imp, py in self._python_rendered:
            imports |= imp
            python_rendered.append(py)

        if with_imports:
            self._python_str = (
                self.Prefix + "\n" + "\n".join(imports) + "\n".join(python_rendered)
            )
        else:
            self._python_str = self.Prefix + "\n" + "\n".join(python_rendered)

        includes = set()
        c_rendered = []
        for inc, c in self._c_rendered:
            includes |= inc
            c_rendered.append(c)

        self._c_str = "\n".join(includes) + "\n".join(c_rendered)

        self._shim_function_pystr = self.MemoryShimWriterTemplate.format(
            shim_funcs=self._c_str
        )

    def render_as_str(self, *, with_imports: bool, with_shim_functions: bool) -> str:
        """Return the final assembled bindings in script. This output should be final."""
        self._render(with_imports)

        if with_shim_functions:
            output = self._python_str + "\n" + self._shim_function_pystr
        else:
            output = self._python_str

        file_logger.debug(output)

        return output
