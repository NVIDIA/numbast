# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from warnings import warn
from collections import defaultdict

from numba import types as nbtypes
from numba.cuda.typing import signature as nb_signature, Signature
from numba.cuda.typing.templates import ConcreteTemplate
from numba.cuda import declare_device
from numba.cuda.cudadecl import register_global, register
from numba.cuda.cudaimpl import lower

from ast_canopy.pylibastcanopy import execution_space
from ast_canopy.decl import Function

from numbast.types import to_numba_type
from numbast.utils import (
    deduplicate_overloads,
    make_device_caller_with_nargs,
    make_function_shim,
)
from numbast.shim_writer import MemoryShimWriter as ShimWriter

function_binding_shim_template = """
extern "C" __device__ int
{shim_name}({return_type} &retval {arglist}) {{
    retval = {method_name}({args});
    return 0;
}}
"""


def make_new_func_obj():
    def func():
        pass

    return func


overload_registry: dict[str, list[Signature]] = defaultdict(list)
func_obj_registry: dict[str, object] = defaultdict(make_new_func_obj)


def bind_cxx_operator_overload_function(
    shim_writer: ShimWriter, func_decl: Function
) -> object:
    """Create bindings for a C++ operator-overload function.

    Parameters
    ----------
    shim_writer : ShimWriter
        The shim writer to write the generated shim layer code.

    func_decl : Function
        The declaration of the function in C++.

    Returns
    -------
    shim_call : object
        The Numba-CUDA-callable Python API for the function.
    """
    if func_decl.is_copy_assignment_operator():
        # copy assignment operator, do not support in Numba / Python, skip
        return None

    return_type = to_numba_type(
        func_decl.return_type.unqualified_non_ref_type_name
    )
    param_types = [
        to_numba_type(arg.unqualified_non_ref_type_name)
        for arg in func_decl.param_types
    ]

    return_type_name = func_decl.return_type.unqualified_non_ref_type_name
    shim_func_name = deduplicate_overloads(func_decl.mangled_name)

    py_op = func_decl.overloaded_operator_to_python_operator

    assert py_op is not None

    # Crossing C / C++ boundary, pass argument by pointers.
    arglist = ", ".join(
        f"{arg.type_.unqualified_non_ref_type_name}* {arg.name}"
        for arg in func_decl.params
    )
    if arglist:
        arglist = ", " + arglist
    shim = function_binding_shim_template.format(
        shim_name=shim_func_name,
        return_type=return_type_name,
        arglist=arglist,
        method_name=func_decl.name,
        args=", ".join("*" + arg.name for arg in func_decl.params),
    )

    # Typing
    @register_global(py_op)
    class op_decl(ConcreteTemplate):
        cases = [nb_signature(return_type, *param_types)]

    # Lowering
    # FIXME: temporary solution for mismatching function prototype against definition.
    # If params are passed by value, at prototype the signature of __nv_bfloat16 is set
    # to `b32` type, but to `b64` at definition, causing a linker error. A temporary solution
    # is to pass all params by pointer and dereference them in shim. See dereferencing at the
    # shim generation below.
    func = declare_device(
        shim_func_name, return_type(*map(nbtypes.CPointer, param_types))
    )
    python_api = make_device_caller_with_nargs(
        shim_func_name + "_shim", len(param_types), func
    )

    @lower(py_op, *param_types)
    def impl(context, builder, sig, args):
        shim_writer.write_to_shim(shim, shim_func_name)
        ptrs = [builder.alloca(context.get_value_type(arg)) for arg in sig.args]
        for ptr, ty, arg in zip(ptrs, sig.args, args):
            builder.store(arg, ptr, align=getattr(ty, "alignof_", None))

        return context.compile_internal(
            builder,
            python_api,
            nb_signature(return_type, *map(nbtypes.CPointer, param_types)),
            ptrs,
        )

    return python_api


def bind_cxx_non_operator_function(
    shim_writer: ShimWriter,
    func_decl: Function,
    skip_prefix: str | None,
    exclude: set[str],
) -> object:
    """Create bindings for a C++ non-operator function.

    Parameters
    ----------
    shim_writer : ShimWriter
        The shim writer to write the generated shim layer code.

    func_decl : Function
        The declaration of the function in C++.

    skip_prefix : str | None
        Skip functions with this prefix. Has no effect if None or empty.

    exclude : set[str]
        A set of function names to exclude.


    Returns
    -------
    func : object
        The Python-callable API for the function.
    """
    global overload_registry

    if (
        skip_prefix and func_decl.name.startswith(skip_prefix)
    ) or func_decl.name in exclude:
        # Non public API
        return None

    return_type = to_numba_type(
        func_decl.return_type.unqualified_non_ref_type_name
    )
    param_types = [
        to_numba_type(arg.unqualified_non_ref_type_name)
        for arg in func_decl.param_types
    ]

    # python handle
    func = func_obj_registry[func_decl.name]
    func.__name__ = func_decl.name

    func_sig = nb_signature(return_type, *param_types)
    overload_registry[func_decl.name].append(func_sig)

    # Typing
    @register
    class func_typing(ConcreteTemplate):
        key = func
        cases = overload_registry[func_decl.name]

    register_global(func, nbtypes.Function(func_typing))

    return_type_name = func_decl.return_type.unqualified_non_ref_type_name
    shim_func_name = deduplicate_overloads(func_decl.mangled_name)

    # Declaration of the foreign function
    native_func = declare_device(
        shim_func_name, return_type(*map(nbtypes.CPointer, param_types))
    )
    shim_call = make_device_caller_with_nargs(
        shim_func_name + "_shim", len(param_types), native_func
    )

    shim = make_function_shim(
        shim_func_name, func_decl.name, return_type_name, func_decl.params
    )

    # Lowering
    @lower(func, *param_types)
    def impl(context, builder, sig, args):
        shim_writer.write_to_shim(shim, shim_func_name)
        ptrs = [builder.alloca(context.get_value_type(arg)) for arg in sig.args]
        for ptr, ty, arg in zip(ptrs, sig.args, args):
            builder.store(arg, ptr, align=getattr(ty, "alignof_", None))

        return context.compile_internal(
            builder,
            shim_call,
            nb_signature(return_type, *map(nbtypes.CPointer, param_types)),
            ptrs,
        )

    return func


def bind_cxx_function(
    shim_writer: ShimWriter,
    func_decl: Function,
    skip_prefix: str | None = None,
    skip_non_device: bool = True,
    exclude: set[str] = set(),
) -> object:
    """Create bindings for a C++ function.

    Parameters
    ----------
    shim_writer : ShimWriter
        The shim writer to write the generated shim layer code.

    func_decl : Function
        Declaration of the function in CXX

    skip_prefix : str | None
        Skip functions with this prefix. Has no effect if None or empty.

    skip_non_device : bool
        Skip non device functions. Default to True.

    exclude : set[str]
        A set of function names to exclude. Default to empty set.

    Returns
    -------
    func : object
        The Numba-CUDA-callable Python API for the function.
    """

    if skip_non_device and func_decl.exec_space not in {
        execution_space.device,
        execution_space.host_device,
    }:
        # Skip non device functions
        warn(f"Skipped non device function {func_decl.name}.")
        return None

    if func_decl.is_overloaded_operator():
        return bind_cxx_operator_overload_function(shim_writer, func_decl)
    elif not func_decl.is_operator:
        return bind_cxx_non_operator_function(
            shim_writer, func_decl, skip_prefix=skip_prefix, exclude=exclude
        )

    return None


def bind_cxx_functions(
    shim_writer: ShimWriter,
    functions: list[Function],
    skip_prefix: str | None = None,
    skip_non_device: bool = True,
    exclude: set[str] = set(),
) -> list[object]:
    """Create bindings for a list of C++ functions.

    Parameters
    ----------
    shim_writer : ShimWriter
        The shim writer to write the generated shim layer code.

    functions : list[Function]
        A list of function declarations in CXX.

    skip_prefix : str | None
        Skip functions with this prefix. Has no effect if None or empty.

    skip_non_device : bool
        Skip non device functions. Default to True.

    exclude : set[str]
        A set of function names to exclude. Default to empty set.

    Returns
    -------
    funcs : list[object]
        A list of Numba-CUDA-callable Python APIs for the functions.
    """

    funcs = []
    for func_decl in functions:
        F = bind_cxx_function(
            shim_writer,
            func_decl,
            skip_prefix=skip_prefix,
            skip_non_device=skip_non_device,
            exclude=exclude,
        )
        # overloaded operator (e.g. "+") do not need to have a separate API
        # as they are called directly from the Python operator.
        if F and not func_decl.is_overloaded_operator():
            funcs.append(F)

    return funcs
