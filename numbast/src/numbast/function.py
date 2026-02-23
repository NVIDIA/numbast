# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from warnings import warn
from collections import defaultdict

from numba import types as nbtypes
from numba.cuda.typing import signature as nb_signature, Signature
from numba.cuda.typing.templates import ConcreteTemplate
from numba.cuda.cudadecl import register_global, register
from numba.cuda.cudaimpl import lower

from ast_canopy.pylibastcanopy import execution_space
from ast_canopy.decl import Function

from numbast.types import to_numba_type, to_numba_arg_type
from numbast.intent import compute_intent_plan
from numbast.utils import (
    deduplicate_overloads,
    make_function_shim,
)
from numbast.shim_writer import MemoryShimWriter as ShimWriter
from numbast.callconv import FunctionCallConv

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
    shim_writer: ShimWriter,
    func_decl: Function,
    *,
    arg_intent: dict | None = None,
) -> object:
    """
    Create a Numba-callable binding for a C++ operator-overload function.

    Parameters:
        func_decl (Function): C++ function declaration to bind.
        arg_intent (dict | None): Optional mapping that customizes argument intent (e.g., which reference parameters are treated as input, output, or inout). When provided, intent controls visible parameter/pointer treatment and out-return composition.

    Returns:
        FunctionCallConv | None: A callable wrapper used during lowering that performs the bound call, or `None` when the operator is unsupported (e.g., copy assignment operators).
    """
    if func_decl.is_copy_assignment_operator():
        # copy assignment operator, do not support in Numba / Python, skip
        return None

    return_type = to_numba_type(
        func_decl.return_type.unqualified_non_ref_type_name
    )
    param_types = [to_numba_arg_type(arg) for arg in func_decl.param_types]
    arg_is_ref = [
        bool(t.is_left_reference() or t.is_right_reference())
        for t in func_decl.param_types
    ]

    return_type_name = func_decl.return_type.unqualified_non_ref_type_name
    mangled_name = deduplicate_overloads(func_decl.mangled_name)
    shim_func_name = f"{mangled_name}_nbst"

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

    func_cc = FunctionCallConv(
        mangled_name, shim_writer, shim, arg_is_ref=arg_is_ref
    )

    @lower(py_op, *param_types)
    def impl(context, builder, sig, args):
        """
        Delegate lowering to the captured FunctionCallConv instance `func_cc`.

        Parameters:
            context: Numba lowering context used during compilation.
            builder: LLVM IR builder used to emit instructions.
            sig: The function signature being lowered.
            args: Sequence of lowered argument values passed to the call.

        Returns:
            The lowered native value(s) produced by `func_cc`.
        """
        return func_cc(builder, context, sig, args)

    return func_cc


def bind_cxx_non_operator_function(
    shim_writer: ShimWriter,
    func_decl: Function,
    skip_prefix: str | None,
    exclude: set[str],
    *,
    arg_intent: dict | None = None,
) -> object:
    """
    Create a Python-callable binding for a C++ non-operator function.

    Optionally uses an arg_intent override to control which C++ reference parameters are exposed as pointer parameters or returned as out-returns; when no overrides are provided, reference parameters are treated as input-only values.

    Parameters
    ----------
    shim_writer : ShimWriter
        Writer used to emit the generated shim layer code.
    func_decl : Function
        C++ function declaration to bind.
    skip_prefix : str | None
        If provided, skip functions whose names start with this prefix.
    exclude : set[str]
        Set of function names to exclude from binding.
    arg_intent : dict | None, optional
        Optional per-function intent overrides that specify visibility and in/out semantics for reference parameters.

    Returns
    -------
    object
        The Python-callable function object registered for the bound C++ function, or `None` if the function is skipped.
    """
    global overload_registry

    if (
        skip_prefix and func_decl.name.startswith(skip_prefix)
    ) or func_decl.name in exclude:
        # Non public API
        return None

    cxx_return_type = to_numba_type(
        func_decl.return_type.unqualified_non_ref_type_name
    )

    overrides = arg_intent.get(func_decl.name) if arg_intent else None
    if overrides is None:
        # Backward-compatible default: refs are input-only values.
        return_type = cxx_return_type
        param_types = [to_numba_arg_type(arg) for arg in func_decl.param_types]
        arg_is_ref = [
            bool(t.is_left_reference() or t.is_right_reference())
            for t in func_decl.param_types
        ]
        intent_plan = None
        out_return_types = None
    else:
        # Opt-in: user controls which reference args are exposed as pointers or out-returns.
        intent_plan = compute_intent_plan(
            params=func_decl.params,
            param_types=func_decl.param_types,
            overrides=overrides,
            allow_out_return=True,
        )

        # Visible param types in original order
        param_types = []
        for orig_idx in intent_plan.visible_param_indices:
            base = to_numba_type(
                func_decl.param_types[orig_idx].unqualified_non_ref_type_name
            )
            if intent_plan.intents[orig_idx].value in ("inout_ptr", "out_ptr"):
                param_types.append(nbtypes.CPointer(base))
            else:
                param_types.append(base)

        out_return_types = [
            to_numba_type(
                func_decl.param_types[i].unqualified_non_ref_type_name
            )
            for i in intent_plan.out_return_indices
        ]

        if out_return_types:
            if cxx_return_type == nbtypes.void:
                if len(out_return_types) == 1:
                    return_type = out_return_types[0]
                else:
                    return_type = nbtypes.Tuple(tuple(out_return_types))
            else:
                return_type = nbtypes.Tuple(
                    tuple([cxx_return_type, *out_return_types])
                )
        else:
            return_type = cxx_return_type

        # In intentful mode, pass-through pointers are controlled by intent_plan,
        # not by whether the C++ parameter is a reference.
        arg_is_ref = None

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
    mangled_name = deduplicate_overloads(func_decl.mangled_name)
    shim_func_name = f"{mangled_name}_nbst"

    shim = make_function_shim(
        shim_func_name, func_decl.name, return_type_name, func_decl.params
    )

    func_cc = FunctionCallConv(
        mangled_name,
        shim_writer,
        shim,
        arg_is_ref=arg_is_ref,
        intent_plan=intent_plan,
        out_return_types=out_return_types,
        cxx_return_type=cxx_return_type,
    )

    # Lowering
    @lower(func, *param_types)
    def impl(context, builder, sig, args):
        return func_cc(builder, context, sig, args)

    return func


def bind_cxx_function(
    shim_writer: ShimWriter,
    func_decl: Function,
    skip_prefix: str | None = None,
    skip_non_device: bool = True,
    exclude: set[str] = set(),
    *,
    arg_intent: dict | None = None,
) -> object:
    """
    Create Python bindings for a C++ function.

    Parameters:
        shim_writer (ShimWriter): Writer that emits the generated C/C++ shim code.
        func_decl (Function): C++ function declaration to bind.
        skip_prefix (str | None): If provided, skip functions whose names start with this prefix.
        skip_non_device (bool): If True, skip functions not marked for device or host_device execution.
        exclude (set[str]): Names of functions to exclude from binding.
        arg_intent (dict | None): Optional explicit intent overrides that control which C++ reference
            parameters are exposed as inputs, outputs, or inout pointers and which parameters are
            promoted to out-returns.

    Returns:
        object or None: The Numba-CUDA-callable Python binding object for the function, or `None`
        if the function is skipped or not exposed.
    """

    if skip_non_device and func_decl.exec_space not in {
        execution_space.device,
        execution_space.host_device,
    }:
        # Skip non device functions
        warn(f"Skipped non device function {func_decl.name}.")
        return None

    if func_decl.is_overloaded_operator():
        return bind_cxx_operator_overload_function(
            shim_writer, func_decl, arg_intent=arg_intent
        )
    elif not func_decl.is_operator:
        return bind_cxx_non_operator_function(
            shim_writer,
            func_decl,
            skip_prefix=skip_prefix,
            exclude=exclude,
            arg_intent=arg_intent,
        )

    return None


def bind_cxx_functions(
    shim_writer: ShimWriter,
    functions: list[Function],
    skip_prefix: str | None = None,
    skip_non_device: bool = True,
    exclude: set[str] = set(),
    *,
    arg_intent: dict | None = None,
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
            arg_intent=arg_intent,
        )
        # overloaded operator (e.g. "+") do not need to have a separate API
        # as they are called directly from the Python operator.
        if F and not func_decl.is_overloaded_operator():
            funcs.append(F)

    return funcs
