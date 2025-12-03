# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Tuple
from collections import defaultdict
import re

from numba.cuda.compiler import ExternFunction

from ast_canopy import pylibastcanopy

OVERLOADS_CNT: dict[str, int] = defaultdict(int)  # overload counter


def make_device_caller_with_nargs(
    name: str, nargs: int, wrapped: ExternFunction
) -> Callable:
    """Create a Python wrapper for ``wrapped`` with ``nargs`` arguments.

    Parameters
    ----------
    name : str
        The name of the function.
    nargs : int
        The number of arguments to replace the function with.

    wrapped : ExternFunction
        The function to wrap.

    Returns
    -------
    func : callable
        The function stub.
    """

    args = ", ".join(f"arg{i}" for i in range(nargs))

    func = f"""
def {name}({args}):
    return wrapped({args})
    """

    globals = {"wrapped": wrapped}
    exec(func, globals)
    func_obj = globals[name]

    assert isinstance(func_obj, Callable)
    return func_obj


def deduplicate_overloads(func_name: str) -> str:
    """Deduplicate a function name across overloads.

    Parameters
    ----------
    func_name : str
        The name of the function.

    Returns
    -------
    func_name : str
        The deduplicated function name.
    """

    OVERLOADS_CNT[func_name] += 1
    return func_name + f"_{OVERLOADS_CNT[func_name]}"


def paramvar_to_str(arg: pylibastcanopy.ParamVar):
    """Convert a ``ParamVar`` into a C++ formal-argument declaration string.

    Performs necessary downcasting of array-typed ``ParamVar`` to pointer types.
    """
    array_pattern = r"(.*)(\[\d+\]+)"

    # For each of the arguments, elevate to pointer type.
    match = re.match(array_pattern, arg.type_.unqualified_non_ref_type_name)
    if match:
        # Array type
        base_ty, sizes = match.groups()
        if "*" in base_ty:
            # Pointer to array type: int (*arr)[10]
            loc = base_ty.rfind("*")
            fml_arg = (
                base_ty[: loc + 1] + f"*{arg.name}" + base_ty[loc + 1 :] + sizes
            )
        else:
            # Regular array type: int arr[10]
            fml_arg = base_ty + f" (*{arg.name})" + sizes
    else:
        fml_arg = f"{arg.type_.unqualified_non_ref_type_name}* {arg.name}"

    return fml_arg


def assemble_arglist_string(params: list[pylibastcanopy.ParamVar]) -> str:
    """Assemble comma separated arg string prefixed by a single comma.
    If parameter list is empty, return empty string.
    """
    if not params:
        return ""

    arglist = ", ".join(paramvar_to_str(arg) for arg in params)
    return ", " + arglist


def assemble_dereferenced_params_string(
    params: list[pylibastcanopy.ParamVar],
) -> str:
    """Assemble comma separated dereferenced param string."""
    return ", ".join(f"*{p.name}" for p in params)


def get_return_type_strings(return_type: str) -> Tuple[str, str]:
    """Get the return type string for a C++ type."""
    if return_type == "void":
        retval = ""
        return_type = "int"
    else:
        retval = "retval = "

    return retval, return_type


def make_function_shim(
    shim_name: str,
    func_name: str,
    return_type: str,
    params: list[pylibastcanopy.ParamVar],
    includes: list[str] = [],
) -> str:
    """Create a shim function for a C++ standalone function.

    Parameters
    ----------
    shim_name : str
        The name of the shim function.
    func_name : str
        The name of the function to build shim for.
    return_type : str
        The return type of the function.
    params : list[pylibastcanopy.ParamVar]
        The parameters of the function.
    includes : list[str]
        The list of header paths to be included to the shim.

    Returns
    -------
    shim : str
        The function shim layer source.
    """

    function_binding_shim_template = """{includes}
extern "C" __device__ int
{shim_name}({return_type} &retval {formal_args}) {{
    {retval}{func_name}({actual_args});
    return 0;
}}
    """

    retval, return_type = get_return_type_strings(return_type)

    formal_args_str = assemble_arglist_string(params)
    acutal_args_str = assemble_dereferenced_params_string(params)

    include_str = "\n".join([f"#include <{include}>" for include in includes])
    shim = function_binding_shim_template.format(
        includes=include_str,
        shim_name=shim_name,
        return_type=return_type,
        func_name=func_name,
        formal_args=formal_args_str,
        retval=retval,
        actual_args=acutal_args_str,
    )

    return shim


def make_struct_ctor_shim(
    shim_name: str,
    struct_name: str,
    params: list[pylibastcanopy.ParamVar],
    includes: list[str] = [],
) -> str:
    """Create a struct constructor shim function.

    Parameters
    ----------
    shim_name : str
        The name of the shim function.
    struct_name : str
        The name of the struct to construct.
    params : list[pylibastcanopy.ParamVar]
        The parameters of the function.
    includes : list[str]
        The list of header paths to be included to the shim.

    Returns
    -------
    shim : str
        The function shim layer source.
    """

    ctor_binding_shim = """{includes}
extern "C" __device__ int
{shim_name}(int &ignore, {struct_name} *self {formal_args}) {{
    new (self) {struct_name}({actual_args});
    return 0;
}}
    """

    formal_args = [paramvar_to_str(arg) for arg in params]

    formal_args_str = ", ".join(formal_args)
    if formal_args_str:
        # If there are formal arguments, add a comma before them
        # otherwise it's an empty string.
        formal_args_str = ", " + formal_args_str

    acutal_args_str = ", ".join("*" + arg.name for arg in params)

    include_str = "\n".join([f"#include <{include}>" for include in includes])

    shim = ctor_binding_shim.format(
        includes=include_str,
        shim_name=shim_name,
        struct_name=struct_name,
        formal_args=formal_args_str,
        actual_args=acutal_args_str,
    )

    return shim


def make_struct_regular_method_shim(
    shim_name: str,
    struct_name: str,
    method_name: str,
    return_type: str,
    params: list[pylibastcanopy.ParamVar],
    includes: list[str] = [],
) -> str:
    struct_method_shim_layer_template = """{includes}
    extern "C" __device__ int
    {shim_name}({return_type} &retval, {struct_name}* self {arglist}) {{
        {retval}self->{method_name}({args});
        return 0;
    }}
    """

    retval, return_type = get_return_type_strings(return_type)

    formal_args_str = assemble_arglist_string(params)
    acutal_args_str = assemble_dereferenced_params_string(params)

    include_str = "\n".join([f"#include <{include}>" for include in includes])
    shim = struct_method_shim_layer_template.format(
        retval=retval,
        return_type=return_type,
        struct_name=struct_name,
        method_name=method_name,
        arglist=formal_args_str,
        shim_name=shim_name,
        args=acutal_args_str,
        includes=include_str,
    )

    return shim


def make_struct_conversion_operator_shim(
    shim_name: str,
    struct_name: str,
    method_name: str,
    return_type: str,
    includes: list[str] = [],
) -> str:
    """Create a shim function for a C++ struct conversion operator.

    Parameters
    ----------
    shim_name : str
        The name of the shim function.
    struct_name : str
        The name of the struct to construct.
    method_name : str
        The name of the operator to call.
    return_type : str
        The return type of the conversion operator.
    includes : list[str]
        The list of header paths to be included to the shim.

    Returns
    -------
    shim : str
        The function shim layer source.
    """

    conv_op_shim = """{includes}
extern "C" __device__ int
{shim_name}({return_type} &retval, {struct_name} *self) {{
    retval = self->{method_name}();
    return 0;
}}
    """

    include_str = "\n".join([f"#include <{include}>" for include in includes])

    shim = conv_op_shim.format(
        includes=include_str,
        shim_name=shim_name,
        struct_name=struct_name,
        method_name=method_name,
        return_type=return_type,
    )

    return shim


def _apply_prefix_removal(name: str, prefix_to_remove: list[str]) -> str:
    """
    Remove the first matching prefix from a name.

    Parameters:
        name (str): The original identifier (e.g., struct, function, or enum name).
        prefix_to_remove (list[str]): Ordered list of prefixes to try; the first prefix that matches the start of `name` will be removed.

    Returns:
        str: The name with the first matching prefix removed, or the original name if no prefixes match.
    """
    for prefix in prefix_to_remove:
        if name.startswith(prefix):
            return name[len(prefix) :]

    return name
