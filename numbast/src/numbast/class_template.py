# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional
from collections import defaultdict
from tempfile import NamedTemporaryFile
import inspect

from ast_canopy import pylibastcanopy

from numba import types as nbtypes
from numba.core.extending import (
    register_model,
    make_attribute_wrapper,
    lower_builtin,
)
from numba.core.typing import signature as nb_signature
from numba.cuda.typing.templates import (
    ConcreteTemplate,
    AttributeTemplate,
    CallableTemplate,
)
from numba.core.datamodel.models import StructModel, OpaqueModel
from numba.cuda import declare_device
from numba.cuda.cudadecl import register_global, register, register_attr
from numba.cuda.cudaimpl import lower
from numba.cuda.core.imputils import numba_typeref_ctor
from numba.core.typing.npydecl import parse_dtype
from numba.core.errors import RequireLiteralValue, TypingError

from ast_canopy.api import parse_declarations_from_source
from ast_canopy.decl import (
    ClassTemplateSpecialization,
    StructMethod,
    ClassTemplate,
)

from numbast.types import (
    CTYPE_MAPS as C2N,
    to_numba_type,
    is_c_floating_type,
    is_c_integral_type,
    to_c_type_str,
)
from numbast.utils import (
    deduplicate_overloads,
    make_device_caller_with_nargs,
    assemble_arglist_string,
    assemble_dereferenced_params_string,
    make_struct_regular_method_shim,
)
from numbast.shim_writer import ShimWriterBase

ConcreteTypeCache: dict[str, nbtypes.Type] = {}

ConcreteTypeCache2: dict[str, nbtypes.Type] = {}

struct_ctor_shim_layer_template = """
extern "C" __device__ int
{func_name}(int &ignore, {name} *self {arglist}) {{
    new (self) {name}({args});
    return 0;
}}
"""


class MetaType(nbtypes.Type):
    def __init__(self, template_name):
        super().__init__(name=template_name + "MetaType")
        self.template_name = template_name


def bind_cxx_struct_ctor(
    ctor: StructMethod,
    struct_name: str,
    s_type_ref: nbtypes.TypeRef,
    shim_writer: ShimWriterBase,
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
    shim_writer : ShimWriterBase
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
            nbtypes.CPointer(s_type_ref.instance_type),
            *map(nbtypes.CPointer, param_types),
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

    @lower(numba_typeref_ctor, s_type_ref, *param_types)
    def ctor_impl(context, builder, sig, args):
        s_type = s_type_ref.instance_type
        # Delay writing the shim function at lowering time. This avoids writing
        # shim functions from the parsed header that's unused in kernels.
        shim_writer.write_to_shim(shim, func_name)

        selfptr = builder.alloca(context.get_value_type(s_type), name="selfptr")
        argptrs = [
            builder.alloca(context.get_value_type(arg)) for arg in sig.args[1:]
        ]
        for ptr, ty, arg in zip(argptrs, sig.args[1:], args[1:]):
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

    return param_types


def bind_cxx_ctsd_ctors(
    struct_decl: ClassTemplateSpecialization,
    s_type_ref: nbtypes.TypeRef,
    shim_writer: ShimWriterBase,
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
    shim_writer: ShimWriterBase
        The shim writer to write the shim layer code.
    """

    s_type = s_type_ref.instance_type

    ctor_params: list[list[Any]] = []
    for ctor in struct_decl.constructors():
        param_types = bind_cxx_struct_ctor(
            ctor, struct_decl.name, s_type_ref, shim_writer
        )
        if param_types is not None:
            ctor_params.append(param_types)

    # Constructor typing:
    @register
    class TypeRefCallTemplate(ConcreteTemplate):
        key = numba_typeref_ctor
        cases = [
            nb_signature(s_type, s_type_ref, *arglist)
            for arglist in ctor_params
        ]

    register_global(numba_typeref_ctor, nbtypes.Function(TypeRefCallTemplate))

    @register
    class CtorTemplate(ConcreteTemplate):
        key = s_type
        cases = [nb_signature(s_type, *arglist) for arglist in ctor_params]


def bind_cxx_struct_regular_method(
    struct_decl: ClassTemplateSpecialization,
    method_decl: StructMethod,
    s_type: nbtypes.Type,
    shim_writer: ShimWriterBase,
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
    struct_decl: ClassTemplateSpecialization,
    s_type: nbtypes.Type,
    shim_writer: ShimWriterBase,
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


def bind_cxx_class_template_specialization(
    shim_writer: ShimWriterBase,
    struct_decl: ClassTemplateSpecialization,
    instance_type_ref: nbtypes.Type,
    aliases: dict[
        str, list[str]
    ] = {},  # XXX: this should be just a list of aliases
) -> object:
    """
    Create bindings for a C++ struct.

    Parameters
    ----------
    shim_writer : ShimWriterBase
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

    s_type = instance_type_ref.instance_type
    S_type = type(s_type)

    # Any type that was parsed from C++ should be added to type record:
    # It also needs to happen before method typings - because copy constructors
    # needs to know the type of itself even if the definition is incomplete.
    C2N[struct_decl.name] = s_type
    if struct_decl.name in aliases:
        for alias in aliases[struct_decl.name]:
            C2N[alias] = s_type

    # Data Model
    @register_model(S_type)
    class S_model(StructModel):
        def __init__(self, dmm, fe_type, struct_decl=struct_decl):
            members = [
                (
                    f.name,
                    to_numba_type(f.type_.unqualified_non_ref_type_name),
                )
                for f in struct_decl.fields
            ]
            super().__init__(dmm, fe_type, members)

    # ----------------------------------------------------------------------------------
    # Method, Attributes Typing and Lowering:

    method_templates = bind_cxx_struct_regular_methods(
        struct_decl, s_type, shim_writer
    )

    public_fields_tys = {f.name: f.type_ for f in struct_decl.public_fields()}

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
            elif attr == "__call__":
                # Special case when invoking tranpoline typing of numba_typeref_ctor
                # Reject to look for internal typing.
                pass
            else:
                raise AttributeError(attr)

        for field_name in public_fields_tys.keys():
            make_attribute_wrapper(S_type, field_name, field_name)

    # ----------------------------------------------------------------------------------
    # Constructors:
    bind_cxx_ctsd_ctors(struct_decl, instance_type_ref, shim_writer)

    # Return the handle to the type in Numba
    return s_type


def concrete_typing():
    """Create a new concrete type object"""

    class ConcreteType(nbtypes.Type):
        def __init__(self, meta_type, **targs):
            self.meta_type = meta_type
            self.targs = targs
            super().__init__(name=self.angled_targs_str())

        def angled_targs_str(self) -> str:
            """Return type name with `key=value` Numba type format surrounded by angle brackets.
            Example: BlockScan<T=int32, BLOCK_DIM_X=128>
            Mostly used for debugging purposes.
            """
            return (
                self.meta_type.template_name
                + f"<{', '.join([f'{tparam_name}={targ}' for tparam_name, targ in self.targs.items()])}>"
            )

        def angled_targs_str_as_c(self) -> str:
            """Return C++ style template instantiation string.
            Example: BlockScan<int, 128>
            Used to format shim function strings.
            """
            return (
                self.meta_type.template_name
                + f"<{', '.join([f'{targ}' for targ in self.targs_dict_as_c().values()])}>"
            )

        def targs_dict_as_c(self):
            def to_c_str(obj: nbtypes.Type | int | float) -> str:
                if isinstance(obj, nbtypes.Type):
                    return to_c_type_str(obj)

                if isinstance(obj, (int, float)):
                    return str(obj)

                raise ValueError(
                    f"Unknown object to use in C shim function: {obj}"
                )

            return {
                tparam_name: to_c_str(targ)
                for tparam_name, targ in self.targs.items()
            }

    return ConcreteType


def struct_type_from_instantiation(
    instance: nbtypes.Type,
    shim_writer: ShimWriterBase,
    header_path: str,
):
    # Clang determines to populate all members of an instantiated class based
    # on whether an explicit instantitation definition exists. The following
    # code snippet creates such instantiation.
    src = f"""\n
#include "{header_path}"
template class {instance.angled_targs_str_as_c()};
"""

    with NamedTemporaryFile("w") as f:
        f.write(src)
        f.flush()

        try:
            decls = parse_declarations_from_source(
                f.name, [header_path, f.name], compute_capability="sm_86"
            )
        except pylibastcanopy.ParseError as e:
            e.add_note(f"Error when parsing string: {src}")
            raise

    specializations = decls.class_template_specializations
    decl = specializations[0]

    instance_type_ref = nbtypes.TypeRef(instance)
    _block_scan_type = bind_cxx_class_template_specialization(
        shim_writer, decl, instance_type_ref
    )

    return instance_type_ref


def _bind_tparams(
    decl: ClassTemplate, *args, **kwargs
) -> dict[str, nbtypes.Type]:
    def _get_literal_type_cls(c_typ_: str) -> nbtypes.Literal:
        """Get the corresbonding Numba literal type based on C type string.
        Fall back to Literal if unknown.
        """
        if is_c_integral_type(c_typ_):
            return nbtypes.IntegerLiteral
        elif is_c_floating_type(c_typ_):
            return nbtypes.FloatLiteral
        else:
            return nbtypes.Literal

    if args:
        raise TypingError(
            "Template parameter error message is only configurable via keyword argument, not positional argument."
        )

    res: dict[str, nbtypes.Type] = {}

    targs = decl.tparam_dict
    for tparam_name, type_ in kwargs.items():
        param_decl = targs.get(tparam_name, None)
        if param_decl:
            # If it's a non-type template parameter
            if param_decl.kind == pylibastcanopy.template_param_kind.non_type:
                literal_type = _get_literal_type_cls(param_decl.type_)
                if not isinstance(type_, literal_type):
                    # Require the input to be a literal value
                    raise RequireLiteralValue(kwargs[tparam_name])

                res[tparam_name] = type_.literal_value
            else:
                res[tparam_name] = parse_dtype(type_)

    # Reorder the result dict to the same key order as C++
    res = {k: res[k] for k in targs if k in res}
    return res


def _rewrite_typer_signature(decl: ClassTemplate, typer):
    """Rewrites the typer signature to match class template arglist"""
    param_names = list(decl.tparam_dict.keys())
    params = [
        inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for name in param_names
    ]
    pubsig = inspect.Signature(params)
    typer.__signature__ = pubsig


def _register_meta_type(
    stub: object,
    meta_type: nbtypes.Type,
    ctd: ClassTemplate,
    shim_writer: ShimWriterBase,
    header_path: str,
):
    @register
    class MetaType_template_decl(CallableTemplate):
        key = stub

        def generic(
            self,
            meta_type=meta_type,
            decl=ctd,
            shim_writer=shim_writer,
            header_path=header_path,
        ):
            def typer(*args, **kwargs):
                targs = _bind_tparams(decl, *args, **kwargs)
                ConcreteType = concrete_typing()
                instance = ConcreteType(meta_type, **targs)
                unique_id = instance.name

                if unique_id in ConcreteTypeCache:
                    return ConcreteTypeCache[unique_id]

                instance_type_ref = struct_type_from_instantiation(
                    instance, shim_writer, header_path
                )

                ConcreteTypeCache[unique_id] = instance_type_ref

                self.context.refresh()
                return instance_type_ref

            _rewrite_typer_signature(decl, typer)
            return typer

    register_global(stub, nbtypes.Function(MetaType_template_decl))


def bind_cxx_class_template(
    class_template_decl: ClassTemplate,
    shim_writer: ShimWriterBase,
    header_path: str,
):
    # Stub class
    class TC:
        pass

    # Typing
    TC_templated_type = MetaType(class_template_decl.record.name)

    # Data model
    @register_model(MetaType)
    class TC_Template_model(OpaqueModel):
        def __init__(self, dmm, fe_type):
            OpaqueModel.__init__(self, dmm, fe_type)

    # MetaType Lowering, NO-OP
    @lower_builtin(TC, nbtypes.VarArg(nbtypes.Any))
    def lower_noop(context, builder, sig, args):
        return context.get_constant(nbtypes.int32, 0)

    _register_meta_type(
        TC, TC_templated_type, class_template_decl, shim_writer, header_path
    )

    return TC


def bind_cxx_class_templates(
    class_templates: list[ClassTemplate],
    header_path: str,
    shim_writer: ShimWriterBase,
) -> list[object]:
    """
    Create bindings for a list of C++ class templates.

    Parameters
    ----------
    shim_writer : ShimWriter
        The shim writer to write the shim layer code.
    class_templates : list[ClassTemplate]
        List of declarations of the class template types in CXX
    header_path : str
        The path to the header file containing the class template declarations.

    Returns
    -------
    list[object]
        The Python APIs of the class templates.
    """

    python_apis = []
    for ct in class_templates:
        # Bind the cxx class template
        TC = bind_cxx_class_template(
            class_template_decl=ct,
            shim_writer=shim_writer,
            header_path=header_path,
        )
        python_apis.append(TC)

    return python_apis
