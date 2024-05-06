# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from llvmlite import ir

from numba import types as nbtypes
from numba.core.extending import (
    register_model,
    lower_cast,
    make_attribute_wrapper,
)
from numba.core.typing import signature as nb_signature
from numba.core.typing.templates import ConcreteTemplate, AttributeTemplate
from numba.core.datamodel.models import StructModel, PrimitiveModel
from numba.cuda import declare_device
from numba.cuda.cudadecl import register_global, register, register_attr
from numba.cuda.cudaimpl import lower

from pylibastcanopy import access_kind
from ast_canopy.decl import Struct

from numbast.types import CTYPE_MAPS as C2N, to_numba_type
from numbast.utils import (
    deduplicate_overloads,
    make_device_caller_with_nargs,
)
from numbast.shim_writer import MemoryShimWriter as ShimWriter

struct_ctor_shim_layer_template = """
extern "C" __device__ int
{func_name}(int &ignore, {name} *self {arglist}) {{
    new (self) {name}({args});
    return 0;
}}
"""

struct_method_shim_layer_template = """
extern "C" __device__ int
{func_name}({return_type} &retval, {name}* self {arglist}) {{
    retval =  self->{method_name}({args});
    return 0;
}}
"""


def bind_cxx_struct(
    shim_writer: ShimWriter,
    struct_decl: Struct,
    parent_type: type = nbtypes.Type,
    data_model: type = StructModel,
    aliases: dict[str, list[str]] = {},  # XXX: this should be just a list of aliases
) -> object:
    """
    Make bindings for a C++ struct.

    Parameters
    ----------
    shim_writer : ShimWriter
        The shim writer to write the shim layer code.
    struct_decl : Struct
        Declaration of the struct type in CXX
    parent_type : nbtypes.Type, optional
        Parent type of the Python API, by default nbtypes.Type
    data_model : DataModel, optional
        Data model for the struct, by default StructModel
    aliases : dict[str, list[str]], optional
        Mappings from the name of the struct to a list of alises.
        For example in C++: typedef A B; typedef A C; then
        aliases = {"A": ["B", "C"]}

    Returns
    -------
    S : object
        The Python API of the struct.
    shim: str
        The generated shim layer code for struct methods.
    """

    # Typing
    class S_type(parent_type):
        def __init__(self, decl):
            super().__init__(name=f"{struct_decl.name}")
            self.alignof_ = struct_decl.alignof_
            self.bitwidth = struct_decl.sizeof_ * 8

    s_type = S_type(struct_decl)

    # Python API
    S = type(struct_decl.name, (), {"_nbtype": s_type})

    # Any type that was parsed from C++ should be added to type record:
    # It also needs to happen before method typings - because copy constructors
    # needs to know the type of itself even if the definition is incomplete.
    C2N[struct_decl.name] = s_type
    if struct_decl.name in aliases:
        for alias in aliases[struct_decl.name]:
            C2N[alias] = s_type

    # Data Model
    if data_model == PrimitiveModel:

        @register_model(S_type)
        class S_model(data_model):
            def __init__(self, dmm, fe_type):
                be_type = ir.IntType(fe_type.bitwidth)
                super(S_model, self).__init__(dmm, fe_type, be_type)

    elif data_model == StructModel:

        @register_model(S_type)
        class S_model(data_model):
            def __init__(self, dmm, fe_type, struct_decl=struct_decl):
                members = [
                    (f.name, to_numba_type(f.type_.unqualified_non_ref_type_name))
                    for f in struct_decl.fields
                ]
                super().__init__(dmm, fe_type, members)

    # ----------------------------------------------------------------------------------
    # Attributes Typing and Lowering:
    if data_model == StructModel:
        public_fields = {
            f.name: f for f in struct_decl.fields if f.access == access_kind.public_
        }

        @register_attr
        class S_attr(AttributeTemplate):
            key = s_type

            def generic_resolve(self, typ, attr):
                try:
                    ty_name = public_fields[attr].type_.unqualified_non_ref_type_name
                    return to_numba_type(ty_name)
                except KeyError:
                    raise AttributeError(attr)

        for f in public_fields.values():
            make_attribute_wrapper(S_type, f.name, f.name)

    shim = ""

    # ----------------------------------------------------------------------------------
    # Constructors:
    ctor_params = []
    for ctor in struct_decl.constructors():
        if ctor.is_move_constructor:
            # move constructor, do not support in Numba / Python, skip
            continue

        param_types = [
            to_numba_type(arg.unqualified_non_ref_type_name) for arg in ctor.param_types
        ]

        # Cache parameter types
        ctor_params.append(param_types)

        # Lowering
        # Note that libclang always consider the return type of a constructor
        # is void. So we need to manually specify the return type here.
        func_name = deduplicate_overloads(f"__{struct_decl.name}__{ctor.mangled_name}")

        # FIXME: temporary solution for mismatching function prototype against definition.
        # If params are passed by value, at prototype the signature of __nv_bfloat16 is set
        # to `b32` type, but to `b64` at definition, causing a linker error. A temporary solution
        # is to pass all params by pointer and dereference them in shim. See dereferencing at the
        # shim generation below.
        ctor_shim_decl = declare_device(
            func_name,
            nbtypes.int32(
                nbtypes.CPointer(s_type), *map(nbtypes.CPointer, param_types)
            ),
        )

        ctor_shim_call = make_device_caller_with_nargs(
            func_name + "_shim",
            1 + len(param_types),  # the extra argument for placement new pointer
            ctor_shim_decl,
        )

        @lower(S, *param_types)
        def ctor_impl(
            context,
            builder,
            sig,
            args,
            ctor_shim_call=ctor_shim_call,
            param_types=param_types,
            s_type=s_type,
        ):
            selfptr = builder.alloca(context.get_value_type(s_type), name="selfptr")
            argptrs = [builder.alloca(context.get_value_type(arg)) for arg in sig.args]
            for ptr, ty, arg in zip(argptrs, sig.args, args):
                if ty == s_type:
                    builder.store(arg, ptr, align=ty.alignof_)
                else:
                    builder.store(arg, ptr)

            context.compile_internal(
                builder,
                ctor_shim_call,
                nb_signature(
                    nbtypes.int32,
                    nbtypes.CPointer(s_type),
                    *map(nbtypes.CPointer, param_types),
                ),
                (selfptr, *argptrs),
            )
            return builder.load(selfptr, align=s_type.alignof_)

        # Dynamically generate the shim layer:
        # FIXME: All params are passed by pointers, then dereferenced in shim.
        # temporary solution for mismatching function prototype against definition.
        # See above lowering for details.
        arglist = ", ".join(
            f"{arg.type_.unqualified_non_ref_type_name}* {arg.name}"
            for arg in ctor.params
        )
        if arglist:
            arglist = ", " + arglist
        shim += struct_ctor_shim_layer_template.format(
            func_name=func_name,
            name=struct_decl.name,
            arglist=arglist,
            args=", ".join("*" + arg.name for arg in ctor.params),
        )

    # Constructor typing:
    @register
    class CtorTemplate(ConcreteTemplate):
        key = S
        cases = [nb_signature(s_type, *arglist) for arglist in ctor_params]

    register_global(S, nbtypes.Function(CtorTemplate))

    # ----------------------------------------------------------------------------------
    # Conversion operators:

    casted_types = []
    for conv in struct_decl.conversion_operators():
        casted_type = to_numba_type(conv.return_type.unqualified_non_ref_type_name)
        casted_types.append(casted_type)

        # Lowering:
        func_name = deduplicate_overloads(f"__{struct_decl.name}__{conv.mangled_name}")

        shim_decl = declare_device(func_name, casted_type(nbtypes.CPointer(s_type)))
        # Types cast shims always has 1 parameter: self*.
        shim_call = make_device_caller_with_nargs(func_name + "_shim", 1, shim_decl)

        @lower_cast(S_type, casted_type)
        def impl(
            context,
            builder,
            fromty,
            toty,
            value,
            s_type=s_type,
            casted_type=casted_type,
            shim_call=shim_call,
        ):
            ptr = builder.alloca(context.get_value_type(s_type))
            if hasattr(s_type, "alignof_"):
                builder.store(value, ptr, align=s_type.alignof_)
            else:
                builder.store(value, ptr)
            result = context.compile_internal(
                builder,
                shim_call,
                nb_signature(casted_type, nbtypes.CPointer(s_type)),
                (ptr,),
            )
            return result

        # Conversion operators has no arguments
        shim += struct_method_shim_layer_template.format(
            func_name=func_name,
            return_type=conv.return_type.unqualified_non_ref_type_name,
            name=struct_decl.name,
            arglist="",
            method_name=conv.name,
            args="",
        )

    shim_writer.write_to_shim(shim, struct_decl.name)

    # Return the handle to the type in Numba
    return S


def bind_cxx_structs(
    shim_writer: ShimWriter,
    structs: list[Struct],
    parent_types: dict[str, type] = {},
    data_models: dict[str, type] = {},
    aliases: dict[str, list[str]] = {},
) -> list[object]:
    """
    Make bindings for a list of C++ structs.

    Parameters
    ----------
    shim_writer : ShimWriter
        The shim writer to write the shim layer code.
    structs : list[Struct]
        List of declarations of the struct types in CXX
    parent_type : nbtypes.Type, optional
        Parent type of the Python API, by default nbtypes.Type
    data_model : DataModel, optional
        Data model for the struct, by default StructModel
    aliases : dict[str, list[str]], optional
        Mappings from the name of the struct to a list of alises.
        For example in C++: typedef A B; typedef A C; then
        aliases = {"A": ["B", "C"]}

    Returns
    -------
    list[object]
        The Python APIs of the structs.
    """

    python_apis = []
    for s in structs:
        # Determine the type specialization and data model specialization
        if s.name.startswith("unnamed"):
            # Any alias for the unnamed object should suffice.
            alias = aliases[s.name][0]
            type_spec = parent_types[alias]
            data_model_spec = data_models[alias]
        else:
            type_spec = parent_types[s.name]
            data_model_spec = data_models[s.name]

        # Bind the struct
        S = bind_cxx_struct(
            shim_writer,
            s,
            type_spec,
            data_model_spec,
            aliases,
        )
        python_apis.append(S)

    return python_apis
