# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from textwrap import indent
from logging import getLogger, FileHandler
import tempfile

from numbast.static.renderer import BaseRenderer
from numbast.types import to_numba_type
from numbast.utils import (
    deduplicate_overloads,
)

from ast_canopy.decl import Function

file_logger = getLogger(f"{__name__}")
logger_path = os.path.join(tempfile.gettempdir(), "test.py")
file_logger.debug(f"Function debug outputs are written to {logger_path}")
file_logger.addHandler(FileHandler(logger_path))


function_apis_registry = set()


class StaticFunctionRenderer(BaseRenderer):
    decl_device_template = """
{decl_name} = declare_device(
    '{unique_shim_name}',
    int32(
        {return_type},
        {pointer_wrapped_param_types}
    )
)
    """

    caller_template = """
def {caller_name}({nargs}):
    return {decl_name}({nargs})
    """

    c_ext_shim_template = """
extern "C" __device__ int
{unique_shim_name}({return_type} &retval {arglist}) {{
    retval = {method_name}({args});
    return 0;
}}
    """

    lowering_template = """
@lower({func_name}, {params})
def impl(context, builder, sig, args):
    ptrs = [builder.alloca(context.get_value_type(arg)) for arg in sig.args]
    for ptr, ty, arg in zip(ptrs, sig.args, args):
        builder.store(arg, ptr, align=getattr(ty, "alignof_", None))

    return context.compile_internal(
        builder,
        {caller_name},
        nb_signature({return_type}, {pointer_wrapped_param_types}),
        ptrs,
    )
"""

    scoped_lowering_template = """
def _{unique_function_name}_lower():
    {body}

_{unique_function_name}_lower()
"""

    function_python_api_template = """
def {func_name}():
    pass
    """

    _python_api_rendered: str
    """The Python handle used to invoke the function in Numba kernel.
    The API maybe empty because operator reuses `operator.X` handles.
    """

    _typing_rendered: str

    def __init__(self, decl: Function, header_path: str):
        super().__init__(decl)
        self._decl = decl
        self._header_path = header_path

        self._argument_numba_types = [
            to_numba_type(arg.unqualified_non_ref_type_name)
            for arg in self._decl.param_types
        ]
        self._argument_numba_types_str = ", ".join(
            ", ".join(map(str, self._argument_numba_types))
        )

        self._return_numba_type = to_numba_type(
            self._decl.return_type.unqualified_non_ref_type_name
        )

        # Cache the list of parameter types wrapped in pointer types.
        def wrap_pointer(typ):
            return f"CPointer({typ})"

        _pointer_wrapped_param_types = [
            wrap_pointer(typ) for typ in self._argument_numba_types
        ]
        self._pointer_wrapped_param_types_str = ", ".join(_pointer_wrapped_param_types)

        # Cache the unique shim name
        self._deduplicated_shim_name = deduplicate_overloads(decl.mangled_name)
        self._caller_name = f"{self._deduplicated_shim_name}_caller"

        # Cache the list of parameter types in C++ pointer types
        c_ptr_arglist = ", ".join(
            f"{arg.type_.unqualified_non_ref_type_name}* {arg.name}"
            for arg in self._decl.params
        )
        if c_ptr_arglist:
            c_ptr_arglist = ", " + c_ptr_arglist

        self._c_ext_argument_pointer_types = c_ptr_arglist

        # Cache the list of dereferenced arguments
        self._deref_args_str = ", ".join("*" + arg.name for arg in self._decl.params)

    def _render_python_api(self):
        raise NotImplementedError()

    def _render_decl_device(self):
        """Render codes that declares a foreign function for this function in Numba."""

        self.Imports.add("from numba.cuda import declare_device")
        self.Imports.add("from numba.core.typing import signature")
        # All arguments are passed by pointers in C-CPP shim interop
        self.Imports.add("from numba.types import CPointer")
        # Numba ABI returns int32 for exception codes
        self.Imports.add("from numba.types import int32")

        decl_device_rendered = self.decl_device_template.format(
            decl_name=self._deduplicated_shim_name,
            unique_shim_name=self._deduplicated_shim_name,
            pointer_wrapped_param_types=self._pointer_wrapped_param_types_str,
        )

        nargs = [f"arg_{i}" for i in range(len(self._decl.params) + 1)]
        nargs_str = ", ".join(nargs)

        caller_rendered = self.caller_template.format(
            device_caller_name=self._caller_name,
            nargs=nargs_str,
            struct_name=self._deduplicated_shim_name,
        )

        self._decl_device_rendered = decl_device_rendered + "\n" + caller_rendered

    def _render_shim_function(self):
        """Render external C shim functions for this struct constructor."""

        self._c_ext_shim_rendered = self.c_ext_shim_template.format(
            unique_shim_name=self._deduplicated_shim_name,
            arglist=self._c_ext_argument_pointer_types,
            args=self._deref_args_str,
        )

    def _render_typing(self):
        raise NotImplementedError()

    def _render_lowering(self):
        """Render lowering codes for this struct constructor."""

        self.Imports.add("from numba.cuda.cudaimpl import lower")

        self._lowering_rendered = self.lowering_template.format(
            func_name=self.func_name_python,
            params=self._argument_numba_types_str,
            caller_name=self._caller_name,
            pointer_wrapped_args=self._pointer_wrapped_param_types_str,
        )

    def _render_scoped_lower(self):
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

        self._lower_rendered = self.scoped_lowering_template.format(
            unique_function_name=self._deduplicated_shim_name,
            body=lower_body,
        )

    def render_python(self):
        """Render pure python bindings."""

        self._render_python_api()
        self._render_typing()
        self._render_scoped_lower()

        self._python_rendered = (
            self._python_api_rendered
            + "\n"
            + self._typing_rendered
            + "\n"
            + self._lower_rendered
        )

        return self.Imports, self._python_rendered

    def render_c(self):
        self.Includes.add(self.includes_template.format(header_path=self._header_path))
        self._c_rendered = self._c_ext_shim_rendered
        return self.Includes, self._c_rendered

    @property
    def func_name_python(self):
        return self._decl.name


class StaticOverloadedOperatorRenderer(StaticFunctionRenderer):
    """Render bindings of an overloaded operator, such as "operator +"

    Parameters
    ----------
    """

    signature_template = "nb_signature({return_type}, {param_types})"

    typing_template = """
@register_global({py_op_name})
class (ConcreteTemplate):
    cases = [{signature_cases}]
"""

    _py_op_name: str
    """Name of the corresponding python operator converted from C++."""

    def __init__(self, decl: Function, header_path: str):
        super().__init__(decl, header_path)

        self._py_op = decl.overloaded_operator_to_python_operator
        self._py_op_name = f"operator.{self._py_op.__name__}"

    def _signature_cases(self):
        return_type_name = str(self._return_numba_type)
        param_types_str = ", ".join(str(t) for t in self._argument_numba_types)
        return self.signature_template.format(
            return_type=return_type_name, param_types=param_types_str
        )

    def _render_python_api(self):
        """It is not necessary to create a new python API for overloaded operators.

        In Python, operators has a corresponding object (operator.X), we simply reuse
        these operators as their handles in Numba kernels.
        """
        self._python_api_rendered = ""

    def _render_typing(self):
        self._typing_rendered = self.typing_template.format(
            py_op_name=self._py_op_name,
            signature_list=self._signature_cases(),
        )

    @property
    def func_name_python(self):
        return self._py_op_name


class StaticNonOperatorFunctionRenderer(StaticFunctionRenderer):
    """Render bindings of a non-operator function.

    Parameters
    ----------
    """

    signature_template = "nb_signature({return_type}, {param_types})"

    typing_template = """
@register
class {func_typing_name}(ConcreteTemplate):
    key = {func_name}
    cases = {func_signature_list}
"""

    _py_op_name: str
    """Name of the corresponding python operator converted from C++."""

    def __init__(self, decl: Function, header_path: str):
        super().__init__(decl, header_path)

    def _render_python_api(self):
        if self._decl.name in function_apis_registry:
            return
        function_apis_registry.add(self._decl.name)
        self._python_api_rendered = self.function_python_api_template.format(
            func_name=self._decl.name
        )

    def _signature_cases(self):
        return_type_name = str(self._return_numba_type)
        param_types_str = ", ".join(str(t) for t in self._argument_numba_types)
        return self.signature_template.format(
            return_type=return_type_name, param_types=param_types_str
        )

    def _render_typing(self):
        self._typing_rendered = self.typing_template.format(
            func_typing_name=self._py_op_name,
            signature_list=self._signature_cases(),
        )

    @property
    def func_name_python(self):
        return self._py_op_name


class StaticFunctionsRenderer(BaseRenderer):
    """Render a collection of CUDA function declarations.

    Parameters
    ----------

    """

    def __init__(self, decls: list[Function], header_path: str):
        self._decls = decls
        self._header_path = header_path

        self._python_rendered = []
        self._c_rendered = []

    def _render(self):
        self.Imports.add("from numba.cuda import CUSource")

        for decl in self._decls:
            renderer = None
            if decl.is_overloaded_operator():
                renderer = StaticOverloadedOperatorRenderer(decl, self._header_path)
            elif not decl.is_operator:
                renderer = StaticNonOperatorFunctionRenderer(decl, self._header_path)

            if renderer:
                self._python_rendered.append(renderer.render_python())
                self._c_rendered.append(renderer.render_c())

        # imports = set()
        python_rendered = []
        for imp, py in self._python_rendered:
            # imports |= imp
            python_rendered.append(py)

        self._python_str = (
            self.Prefix + "\n" + "\n".join(self.Imports) + "\n".join(python_rendered)
        )

        # includes = set()
        c_rendered = []
        for inc, c in self._c_rendered:
            # includes |= inc
            c_rendered.append(c)

        self._c_str = "\n".join(self.Includes) + "\n".join(c_rendered)

        self._shim_function_pystr = self.MemoryShimWriterTemplate.format(
            shim_funcs=self._c_str
        )

    def render_as_str(self) -> str:
        """Return the final assembled bindings in script. This output should be final."""
        self._render()
        output = self._python_str + "\n" + self._shim_function_pystr

        with open("test.py", "w") as f:
            f.write(output)
        file_logger.debug(output)

        return output
