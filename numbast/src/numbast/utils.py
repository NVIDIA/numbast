# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Callable
from collections import defaultdict
import re

from numba.cuda.compiler import ExternFunction

import pylibastcanopy

OVERLOADS_CNT: dict[str, int] = defaultdict(int)  # overload counter


def make_device_caller_with_nargs(
    name: str, nargs: int, wrapped: ExternFunction
) -> Callable:
    """Create a wrapper for `wrapped` with `nargs` arguments.

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
    """Deduplicate function overloads.

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
    """Convert a ParamVar to a string type name.

    Perform necessary downcasting of array type ParamVar to a pointer type.
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


def make_function_shim(
    shim_name: str,
    func_name: str,
    return_type: str,
    params: list[pylibastcanopy.ParamVar],
    includes: list[str] = [],
) -> str:
    """Create shim function for C++ standalone function.

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
        The function shim layer shim.
    """

    function_binding_shim_template = """{includes}
extern "C" __device__ int
{shim_name}({return_type} &retval {formal_args}) {{
    {retval}{func_name}({actual_args});
    return 0;
}}
    """

    if return_type == "void":
        retval = ""
        return_type = "int"
    else:
        retval = "retval = "

    formal_args = [paramvar_to_str(arg) for arg in params]

    formal_args_str = ", ".join(formal_args)
    if formal_args_str:
        # If there are formal arguments, add a comma before them
        # otherwise it's an empty string.
        formal_args_str = ", " + formal_args_str

    acutal_args_str = ", ".join("*" + arg.name for arg in params)

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
    """Create a struct ctor shim function.

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
        The function shim layer shim.
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


def make_struct_conversion_operator_shim(
    shim_name: str,
    struct_name: str,
    method_name: str,
    return_type: str,
    includes: list[str] = [],
) -> str:
    """Create a shim function for C++ struct conversion operator.

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
        The function shim layer shim.
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
