# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional
from collections import defaultdict

from llvmlite import ir

from numba.cuda import types as nbtypes
from numba.cuda.extending import (
    register_model,
    lower_cast,
    make_attribute_wrapper,
)
from numba.cuda.typing import signature as nb_signature
from numba.cuda.typing.templates import ConcreteTemplate, AttributeTemplate
from numba.cuda.datamodel.models import StructModel, PrimitiveModel
from numba.cuda import declare_device
from numba.cuda.cudadecl import register_global, register, register_attr
from numba.cuda.cudaimpl import lower

from ast_canopy.pylibastcanopy import method_kind
from ast_canopy.decl import Struct, StructMethod

from numbast.types import CTYPE_MAPS as C2N, to_numba_type
from numbast.utils import (
    deduplicate_overloads,
    make_device_caller_with_nargs,
    make_struct_regular_method_shim,
    assemble_arglist_string,
    assemble_dereferenced_params_string,
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


def bind_cxx_struct_ctor(
    ctor: StructMethod,
    struct_name: str,
    s_type: nbtypes.Type,
    S: object,
    shim_writer: ShimWriter,
) -> Optional[list]:
    """Create bindings for a C++ struct constructor and return its argument types.

    Parameters
    ----------

    ctor : StructMethod
        Constructor declaration of struct in CXX
    struct_name : str
        The name of the struct from which this constructor belongs to
    s_type : numba.types.Type
        The Numba type of the struct
    S : object
        The Python API of the struct
    shim_writer : ShimWriter
        The shim writer to write the shim layer code.

    Returns
    -------
    list of argument types, optional
        If the constructor is a move constructor, return ``None``. Otherwise,
        return the list of argument types.
    """

    if ctor.is_move_constructor:
        # move constructor is trivially supported in Numba / Python, skip
        return None

    param_types = [
        to_numba_type(arg.unqualified_non_ref_type_name)
        for arg in ctor.param_types
    ]

    # Lowering
    # Note that libclang always consider the return type of a constructor
    # is void. So we need to manually specify the return type here.
    func_name = deduplicate_overloads(f"{ctor.mangled_name}_nbst")

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

    # Dynamically generate the shim layer:
    # FIXME: All params are passed by pointers, then dereferenced in shim.
    # temporary solution for mismatching function prototype against definition.
    # See above lowering for details.
    arglist = assemble_arglist_string(ctor.params)

    shim = struct_ctor_shim_layer_template.format(
        func_name=func_name,
        name=struct_name,
        arglist=arglist,
        args=assemble_dereferenced_params_string(ctor.params),
    )

    @lower(S, *param_types)
    def ctor_impl(context, builder, sig, args):
        # Delay writing the shim function at lowering time. This avoids writing
        # shim functions from the parsed header that's unused in kernels.
        shim_writer.write_to_shim(shim, func_name)

        selfptr = builder.alloca(context.get_value_type(s_type), name="selfptr")
        argptrs = [
            builder.alloca(context.get_value_type(arg)) for arg in sig.args
        ]
        for ptr, ty, arg in zip(argptrs, sig.args, args):
            builder.store(arg, ptr, align=getattr(ty, "alignof_", None))

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
        return builder.load(selfptr, align=getattr(s_type, "alignof_", None))

    if ctor.kind == method_kind.converting_constructor:
        assert len(param_types) == 1, (
            "isConvertinConstructor in clang ensures that only one parameter is passed"
        )

        @lower_cast(*param_types, s_type)
        def conversion_impl(context, builder, fromty, toty, value):
            return ctor_impl(
                context,
                builder,
                nb_signature(
                    s_type,
                    *map(nbtypes.CPointer, param_types),
                ),
                value,
            )

    return param_types


def bind_cxx_struct_ctors(
    struct_decl: Struct,
    S: object,
    s_type: nbtypes.Type,
    shim_writer: ShimWriter,
):
    """Given a C++ struct declaration, generate bindings for its constructors.

    Parameters
    ----------

    struct_decl: Struct
        The declaration of the struct in CXX
    S: object
        The Python API of the struct
    s_type: numba.types.Type
        The Numba type of the struct
    shim_writer: ShimWriter
        The shim writer to write the shim layer code.
    """

    ctor_params: list[list[Any]] = []
    for ctor in struct_decl.constructors():
        param_types = bind_cxx_struct_ctor(
            ctor, struct_decl.name, s_type, S, shim_writer
        )
        if param_types is not None:
            ctor_params.append(param_types)

    # Constructor typing:
    @register
    class CtorTemplate(ConcreteTemplate):
        key = S
        cases = [nb_signature(s_type, *arglist) for arglist in ctor_params]

    register_global(S, nbtypes.Function(CtorTemplate))


def bind_cxx_struct_conversion_opeartor(
    conv: StructMethod,
    struct_name: str,
    s_type: nbtypes.Type,
    shim_writer: ShimWriter,
):
    """Bind a C++ struct conversion operator to Numba.

    Parameters
    ----------

    conv : StructMethod
        The conversion operator declaration of the struct in CXX
    struct_name : str
        The name of the struct to which this conversion operator belongs
    s_type : nbtypes.Type
        The Numba type of the struct
    shim_writer : ShimWriter
        The shim writer to write the shim layer code.
    """
    casted_type = to_numba_type(conv.return_type.unqualified_non_ref_type_name)

    # Lowering:
    func_name = deduplicate_overloads(f"__{struct_name}__{conv.mangled_name}")

    shim_decl = declare_device(func_name, casted_type(nbtypes.CPointer(s_type)))
    # Types cast shims always has 1 parameter: self*.
    shim_call = make_device_caller_with_nargs(func_name + "_shim", 1, shim_decl)

    # Conversion operators has no arguments
    shim = struct_method_shim_layer_template.format(
        func_name=func_name,
        return_type=conv.return_type.unqualified_non_ref_type_name,
        name=struct_name,
        arglist="",
        method_name=conv.name,
        args="",
    )

    @lower_cast(s_type, casted_type)
    def impl(
        context,
        builder,
        fromty,
        toty,
        value,
    ):
        shim_writer.write_to_shim(shim, func_name)
        ptr = builder.alloca(context.get_value_type(s_type))
        builder.store(value, ptr, align=getattr(s_type, "alignof_", None))
        result = context.compile_internal(
            builder,
            shim_call,
            nb_signature(casted_type, nbtypes.CPointer(s_type)),
            (ptr,),
        )
        return result


def bind_cxx_struct_conversion_opeartors(
    struct_decl: Struct, s_type: nbtypes.Type, shim_writer: ShimWriter
):
    """Bind all conversion operators for a C++ struct."""
    for conv in struct_decl.conversion_operators():
        bind_cxx_struct_conversion_opeartor(
            conv, struct_decl.name, s_type, shim_writer
        )


def bind_cxx_struct_regular_method(
    struct_decl: Struct,
    method_decl: StructMethod,
    s_type: nbtypes.Type,
    shim_writer: ShimWriter,
) -> nb_signature:
    param_types = [
        to_numba_type(arg.unqualified_non_ref_type_name)
        for arg in method_decl.param_types
    ]
    return_type = to_numba_type(
        method_decl.return_type.unqualified_non_ref_type_name
    )

    # Lowering
    func_name = deduplicate_overloads(f"__{method_decl.mangled_name}_nbst")

    c_sig = return_type(
        nbtypes.CPointer(s_type), *map(nbtypes.CPointer, param_types)
    )

    shim_decl = declare_device(func_name, c_sig)

    shim_call = make_device_caller_with_nargs(
        func_name + "_shim", 1 + len(param_types), shim_decl
    )

    shim = make_struct_regular_method_shim(
        shim_name=func_name,
        struct_name=struct_decl.name,
        method_name=method_decl.name,
        return_type=method_decl.return_type.unqualified_non_ref_type_name,
        params=method_decl.params,
    )

    qualname = f"{s_type}.{method_decl.name}"

    @lower(qualname, s_type, *param_types)
    def _method_impl(context, builder, sig, args):
        shim_writer.write_to_shim(shim, func_name)

        argptrs = [
            builder.alloca(context.get_value_type(arg)) for arg in sig.args
        ]
        for ptr, ty, arg in zip(argptrs, sig.args, args):
            builder.store(arg, ptr, align=getattr(ty, "alignof_", None))

        return context.compile_internal(
            builder,
            shim_call,
            c_sig,
            argptrs,
        )

    return nb_signature(return_type, *param_types, recvr=s_type)


def bind_cxx_struct_regular_methods(
    struct_decl: Struct,
    s_type: nbtypes.Type,
    shim_writer: ShimWriter,
) -> dict[str, ConcreteTemplate]:
    """

    Return
    ------

    Mapping from function names to list of signatures.
    """

    method_overloads: dict[str, list[nb_signature]] = defaultdict(list)

    for method in struct_decl.regular_member_functions():
        sig = bind_cxx_struct_regular_method(
            struct_decl, method, s_type, shim_writer
        )
        method_overloads[method.name].append(sig)

    method_templates: dict[str, ConcreteTemplate] = {}

    for name, sigs in method_overloads.items():

        class MethodDecl(ConcreteTemplate):
            key = f"{s_type}.{name}"
            cases = sigs

        method_templates[name] = MethodDecl

    return method_templates


def bind_cxx_struct(
    shim_writer: ShimWriter,
    struct_decl: Struct,
    parent_type: type = nbtypes.Type,
    data_model: type = StructModel,
    aliases: dict[
        str, list[str]
    ] = {},  # XXX: this should be just a list of aliases
) -> object:
    """
    Create bindings for a C++ struct.

    Parameters
    ----------
    shim_writer : ShimWriter
        The shim writer to write the shim layer code.
    struct_decl : Struct
        Declaration of the struct type in CXX
    parent_type : nbtypes.Type, optional
        Parent type of the Python API, by default nbtypes.Type
    data_model : type, optional
        Data model for the struct, by default StructModel
    aliases : dict[str, list[str]], optional
        Mappings from the name of the struct to a list of aliases.
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
                super().__init__(dmm, fe_type, be_type)

    elif data_model == StructModel:

        @register_model(S_type)
        class S_model(data_model):
            def __init__(self, dmm, fe_type, struct_decl=struct_decl):
                members = [
                    (
                        f.name,
                        to_numba_type(f.type_.unqualified_non_ref_type_name),
                    )
                    for f in struct_decl.fields
                ]
                super().__init__(dmm, fe_type, members)

    if data_model == StructModel:
        # ----------------------------------------------------------------------------------
        # Method, Attributes Typing and Lowering:

        method_templates = bind_cxx_struct_regular_methods(
            struct_decl, s_type, shim_writer
        )

        public_fields_tys = {
            f.name: f.type_ for f in struct_decl.public_fields()
        }

        @register_attr
        class S_attr(AttributeTemplate):
            key = s_type

            def _field_ty(self, attr: str) -> nbtypes.Type:
                field_ty = public_fields_tys[attr]
                return to_numba_type(field_ty.unqualified_non_ref_type_name)

            def _method_ty(self, typ, attr: str) -> nbtypes.BoundFunction:
                template = method_templates[attr]
                return nbtypes.BoundFunction(template, typ)

            def generic_resolve(self, typ, attr):
                if attr in public_fields_tys:
                    return self._field_ty(attr)
                elif attr in method_templates:
                    return self._method_ty(typ, attr)
                else:
                    raise AttributeError(attr)

        for field_name in public_fields_tys.keys():
            make_attribute_wrapper(S_type, field_name, field_name)

    # ----------------------------------------------------------------------------------
    # Constructors:
    bind_cxx_struct_ctors(struct_decl, S, s_type, shim_writer)

    # ----------------------------------------------------------------------------------
    # Conversion operators:
    bind_cxx_struct_conversion_opeartors(struct_decl, s_type, shim_writer)

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
    Create bindings for a list of C++ structs.

    Parameters
    ----------
    shim_writer : ShimWriter
        The shim writer to write the shim layer code.
    structs : list[Struct]
        List of declarations of the struct types in CXX
    parent_type : nbtypes.Type, optional
        Parent type of the Python API, by default nbtypes.Type
    data_model : type, optional
        Data model for the struct, by default StructModel
    aliases : dict[str, list[str]], optional
        Mappings from the name of the struct to a list of aliases.
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
