# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from textwrap import indent
from logging import getLogger, FileHandler
import tempfile
from collections import defaultdict

from numbast.static.renderer import BaseRenderer
from numbast.static.types import to_numba_type_str
from numbast.utils import deduplicate_overloads, make_function_shim

from ast_canopy.decl import Function
from pylibastcanopy import execution_space

file_logger = getLogger(f"{__name__}")
logger_path = os.path.join(tempfile.gettempdir(), "test.py")
file_logger.debug(f"Function debug outputs are written to {logger_path}")
file_logger.addHandler(FileHandler(logger_path))


function_apis_registry: set[str] = set()
"""A set of created function API names."""


class StaticFunctionRenderer(BaseRenderer):
    """Base class for function static bindings renderer.

    Many shared function bindings are implemented in the base class,
    such as lowering and C shim functions.

    Parameters
    ----------
    decl: ast_canopy.decl.Function
        A single function declaration parsed by `ast_canopy`
    header_path: str
        The path to the header file that contains the declaration
    """

    signature_template = "signature({return_type}, {param_types})"

    decl_device_template = """
{decl_name} = declare_device(
    '{unique_shim_name}',
    {return_type}(
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
    retval = {func_name}({args});
    return 0;
}}
    """

    c_ext_shim_template_void_ret = """
extern "C" __device__ int
{unique_shim_name}(int &ignored {arglist}) {{
    {func_name}({args});
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
        signature({return_type}, {pointer_wrapped_param_types}),
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

    def __init__(self, decl: Function, header_path: str):
        super().__init__(decl)
        self._decl = decl
        self._header_path = header_path

        self._argument_numba_types = [
            to_numba_type_str(arg.unqualified_non_ref_type_name)
            for arg in self._decl.param_types
        ]
        self._argument_numba_types_str = ", ".join(self._argument_numba_types)

        self._return_numba_type = to_numba_type_str(
            self._decl.return_type.unqualified_non_ref_type_name
        )
        self._return_numba_type_str = str(self._return_numba_type)

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

    @property
    def _signature_cases(self):
        """The python string that declares the signature of this function."""
        return_type_name = str(self._return_numba_type)
        param_types_str = ", ".join(str(t) for t in self._argument_numba_types)
        return self.signature_template.format(
            return_type=return_type_name, param_types=param_types_str
        )

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
            return_type=self._return_numba_type_str,
            pointer_wrapped_param_types=self._pointer_wrapped_param_types_str,
        )

        nargs = [f"arg_{i}" for i in range(len(self._decl.params))]
        nargs_str = ", ".join(nargs)

        caller_rendered = self.caller_template.format(
            caller_name=self._caller_name,
            decl_name=self._deduplicated_shim_name,
            nargs=nargs_str,
        )

        self._decl_device_rendered = decl_device_rendered + "\n" + caller_rendered

    def _render_shim_function(self):
        """Render external C shim functions for this struct constructor."""

        self._c_ext_shim_rendered = make_function_shim(
            shim_name=self._deduplicated_shim_name,
            func_name=self._decl.name,
            return_type=self._decl.return_type.unqualified_non_ref_type_name,
            params=self._decl.params,
        )

        self.ShimFunctions.append(self._c_ext_shim_rendered)

    def _render_lowering(self):
        """Render lowering codes for this struct constructor."""

        self.Imports.add("from numba.cuda.cudaimpl import lower")
        self.Imports.add("from numba.core.typing import signature")

        self._lowering_rendered = self.lowering_template.format(
            func_name=self.func_name_python,
            params=self._argument_numba_types_str,
            caller_name=self._caller_name,
            return_type=self._return_numba_type_str,
            pointer_wrapped_param_types=self._pointer_wrapped_param_types_str,
        )

    def _render_scoped_lower(self):
        """Render FFI, lowering and C shim functions of the constructor.

        Note that the typing still needs to be handled on a higher layer.
        Rendered lowering codes are placed in a function scope isolated from
        other function lowerings.
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
        """Render pure python bindings.
        A single function renderer determines the python API and the lowering code
        which are wrapped in function scope.

        Return
        ------
        python binding: str
            The string containing the rendered python function binding strings.
        """

        self._render_python_api()
        self._render_scoped_lower()

        self._python_rendered = self._python_api_rendered + "\n" + self._lower_rendered

        return self._python_rendered

    def render_c(self):
        """Render the C shim functions from Numba FFI.

        Return
        ------
        C shim functions: str
            The C shim function strings for this function.
        """
        self.Includes.add(self.includes_template.format(header_path=self._header_path))
        self._c_rendered = self._c_ext_shim_rendered
        return self._c_rendered

    @property
    def func_name_python(self):
        """The name of the function in python."""
        return self._decl.name


class StaticOverloadedOperatorRenderer(StaticFunctionRenderer):
    """Render bindings of an overloaded operator, such as "operator +"

    Parameters
    ----------
    decl: ast_canopy.decl.Function
        An operator function declaration in CUDA C++ parsed by `ast_canopy`
    header_path: str
        The path to the header file that contains the declaration
    """

    _py_op_name: str
    """Name of the corresponding python operator converted from C++."""

    def __init__(self, decl: Function, header_path: str):
        super().__init__(decl, header_path)

        self._py_op = decl.overloaded_operator_to_python_operator
        self._py_op_name = f"operator.{self._py_op.__name__}"

    def _render_python_api(self):
        """It is not necessary to create a new python API for overloaded operators.

        In Python, operators has a corresponding object (operator.X), we simply reuse
        these operators as their handles in Numba kernels.
        """
        self._python_api_rendered = ""

    @property
    def func_name_python(self):
        """The name of the operator in python, in the form of `operator.X`."""
        return self._py_op_name

    def get_signature(self):
        return self._signature_cases


class StaticNonOperatorFunctionRenderer(StaticFunctionRenderer):
    """Render bindings of a non-operator function.

    Parameters
    ----------

    decl: ast_canopy.decl.Function
        A non-operator function declaration in CUDA C++ parsed by `ast_canopy`
    header_path: str
        The path to the header file that contains the declaration
    """

    _py_op_name: str
    """Name of the corresponding python operator converted from C++."""

    def __init__(self, decl: Function, header_path: str):
        super().__init__(decl, header_path)

    def _render_python_api(self):
        if self._decl.name in function_apis_registry:
            self._python_api_rendered = ""
            return
        function_apis_registry.add(self._decl.name)
        self._python_api_rendered = self.function_python_api_template.format(
            func_name=self._decl.name
        )

    def get_signature(self):
        return self._signature_cases


class StaticFunctionsRenderer(BaseRenderer):
    """Render a collection of CUDA function declarations.

    Parameters
    ----------

    decls: list[ast_canopy.decl.Function]
        A list of function declarations in CUDA C++, parsed by `ast_canopy`
    header_path: str
        The path to the header file that contains the declarations
    excludes: list[str], Optional
        A list of function names to exclude from the generation
    skip_non_device: bool, default True
        If True, skip generating functions that are not device declared.
    """

    func_typing_template = """
@register
class {func_typing_name}(ConcreteTemplate):
    key = globals()["{func_name}"]
    cases = [{signature_list}]

register_global({func_name}, types.Function({func_typing_name}))
"""

    op_typing_template = """
@register_global({py_op_name})
class {op_typing_name}(ConcreteTemplate):
    cases = [{signature_list}]
"""

    def __init__(
        self,
        decls: list[Function],
        header_path: str,
        excludes: list[str] = [],
        skip_non_device: bool = True,
        skip_prefix: str = "__",
    ):
        self._decls = decls
        self._header_path = header_path
        self._excludes = excludes
        self._skip_non_device = skip_non_device
        self._skip_prefix = skip_prefix

        self._func_typing_signature_cache: dict[str, list[str]] = defaultdict(list)
        self._op_typing_signature_cache: dict[str, list[str]] = defaultdict(list)

        self._python_rendered = []
        self._c_rendered = []

    def _render_typings(self):
        """Render typing for all functions"""
        self.Imports.add("from numba.cuda.cudadecl import register")
        self.Imports.add("from numba.cuda.cudadecl import register_global")
        self.Imports.add("from numba import types")
        self.Imports.add("from numba.core.typing.templates import ConcreteTemplate")

        self._render_func_typings()
        self._render_op_typing()

        self._typing_rendered = (
            self._op_typing_rendered + "\n" + self._func_typing_rendered
        )

    def _render_func_typings(self):
        """Render non-operator function typings."""
        typings_rendered = []
        for func_name in self._func_typing_signature_cache:
            func_typing_name = f"_typing_{func_name}"
            signature_list = self._func_typing_signature_cache[func_name]
            signatures_str = ", ".join(signature_list)
            func_typing_str = self.func_typing_template.format(
                func_typing_name=func_typing_name,
                signature_list=signatures_str,
                func_name=func_name,
            )
            typings_rendered.append(func_typing_str)

        self._op_typing_rendered = "\n".join(typings_rendered)

    def _render_op_typing(self):
        """Render operator overload function typings."""
        typings_rendered = []
        for func_name in self._op_typing_signature_cache:
            self.Imports.add("import operator")
            func_name_id = func_name.replace(".", "_")
            func_typing_name = f"_typing_{func_name_id}"
            signature_list = self._op_typing_signature_cache[func_name]
            signatures_str = ", ".join(signature_list)
            func_typing_str = self.op_typing_template.format(
                py_op_name=func_name,
                op_typing_name=func_typing_name,
                signature_list=signatures_str,
            )
            typings_rendered.append(func_typing_str)

        self._func_typing_rendered = "\n".join(typings_rendered)

    def _render(self, with_prefix: bool, with_imports: bool):
        """Render python bindings and shim functions."""
        self.Imports.add("from numba.cuda import CUSource")

        for decl in self._decls:
            if decl.name in self._excludes:
                continue

            if decl.name.startswith(self._skip_prefix):
                continue

            if self._skip_non_device and decl.exec_space not in {
                execution_space.device,
                execution_space.host_device,
            }:
                continue

            renderer = None
            if decl.is_overloaded_operator():
                if decl.is_copy_assignment_operator():
                    # copy assignment operator, do not support in Numba / Python, skip
                    continue
                renderer = StaticOverloadedOperatorRenderer(decl, self._header_path)
                self._op_typing_signature_cache[renderer.func_name_python].append(
                    renderer.get_signature()
                )
            elif not decl.is_operator:
                renderer = StaticNonOperatorFunctionRenderer(decl, self._header_path)
                self._func_typing_signature_cache[decl.name].append(
                    renderer.get_signature()
                )

            if renderer:
                self._python_rendered.append(renderer.render_python())
                self._c_rendered.append(renderer.render_c())

        # Assemble typings
        self._render_typings()

        python_rendered = []
        for py in self._python_rendered:
            python_rendered.append(py)

        python_rendered.append(self._typing_rendered)

        self._python_str = ""
        if with_prefix:
            self._python_str += "\n" + self.Prefix

        if with_imports:
            self._python_str += "\n" + "\n".join(self.Imports)

        self._python_str += "\n" + "\n".join(python_rendered)

        c_rendered = []
        for c in self._c_rendered:
            c_rendered.append(c)

        self._c_str = "\n".join(self.Includes) + "\n".join(c_rendered)

        self._shim_function_pystr = self.MemoryShimWriterTemplate.format(
            shim_funcs=self._c_str
        )

    def render_as_str(
        self, *, with_prefix: bool, with_imports: bool, with_shim_functions: bool
    ) -> str:
        """Return the final assembled bindings in script. This output should be final."""
        self._render(with_prefix, with_imports)

        if with_shim_functions:
            output = self._python_str + "\n" + self._shim_function_pystr
        else:
            output = self._python_str

        file_logger.debug(output)

        return output


def clear_function_apis_registry():
    """Reset function APIs registry.

    This function is often used when the renderer is executed multiple times in
    the same python session. Such as pytest.
    """
    function_apis_registry.clear()
