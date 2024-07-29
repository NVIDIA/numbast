# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from textwrap import indent

from numba.types import Type
from numba.core.datamodel.models import StructModel, PrimitiveModel

from pylibastcanopy import access_kind
from ast_canopy.decl import Struct, StructMethod

from numbast.static.renderer import (
    BaseRenderer,
)
from numbast.types import to_numba_type
from numbast.utils import deduplicate_overloads


class StaticStructCtorRenderer(BaseRenderer):
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
        self._c_ext_shim_rendered = self.struct_ctor_c_ext_shim_template.format(
            unique_shim_name=self._deduplicated_shim_name,
            struct_name=self._struct_name,
            arglist=self._c_ext_argument_pointer_types,
            args=self._deref_args_str,
        )

    def _render_lowering(self):
        self.Imports.add("from numba.cuda.cudaimpl import lower")

        self._lowering_rendered = self.struct_ctor_lowering_template.format(
            struct_name=self._struct_name,
            param_types=self._nb_param_types_str,
            struct_type_name=self._struct_type_name,
            struct_device_caller_name=self._device_caller_name,
            pointer_wrapped_args=self._pointer_wrapped_param_types_str,
        )

    def _render(self):
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
        return self._nb_param_types

    @property
    def signature_str(self):
        return self.struct_ctor_signature_template.format(
            struct_name=self._struct_name,
            arglist=self._nb_param_types_str,
        )


class StaticStructCtorsRenderer(BaseRenderer):
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
    def python_rendered(self):
        return self._python_rendered

    @property
    def c_rendered(self):
        return self._c_rendered


class StaticStructRenderer(BaseRenderer):
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
        struct_name: str,
        parent_type: type,
        data_model: type,
        header_path: str,
        aliases: list[str] = [],
    ):
        super().__init__(decl)
        self._struct_name = struct_name
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
        self._typing_rendered = self.typing_template.format(
            struct_type_name=self._struct_name,
            parent_type=self._parent_type_str,
            struct_name=self._struct_name,
            struct_alignof=self._decl.alignof_,
            struct_sizeof=self._decl.sizeof_,
        )

    def _render_python_api(self):
        self._python_api_rendered = self.python_api_template.format(
            struct_name=self._struct_name
        )

    def _render_data_model(self):
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
        static_ctors_renderer = StaticStructCtorsRenderer(
            struct_name=self._struct_name,
            struct_type_class=self._struct_type_class_name,
            struct_type_name=self._struct_type_name,
            ctor_decls=self._decl.methods,
        )
        static_ctors_renderer._render()

        self._struct_ctors_python_rendered = static_ctors_renderer.python_rendered
        self._struct_ctors_c_rendered = static_ctors_renderer.c_rendered

    def render_as_str(self) -> str:
        self.Imports.add("from numba.cuda import CUSource")

        self._render_typing()
        self._render_python_api()
        self._render_data_model()
        self._render_struct_attr()
        self._render_struct_ctors()

        self.Includes.append(
            self.includes_template.format(header_path=self._header_path)
        )

        c_ext_merged_shim = "\n".join([*self.Includes, self._struct_ctors_c_rendered])

        self._rendered_c_shims = self.MemoryShimWriterTemplate.format(
            shim_funcs=c_ext_merged_shim
        )

        imports_str = "\n".join(self.Imports)

        return f"""
{self.Prefix}
{imports_str}
{self._typing_rendered}
{self._python_api_rendered}
{self._data_model_rendered}
{self._struct_attr_typing_rendered}
{self._struct_ctors_python_rendered}
{self._rendered_c_shims}
"""

    def render(self, path):
        with open(path, "w") as file:
            file.write(self.render_as_str())
