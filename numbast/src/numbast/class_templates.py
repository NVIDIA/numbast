# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
import typing

from numba import types
from numba.extending import lower_builtin
from numba.core.typing.templates import (
    AttributeTemplate,
    CallableTemplate,
    ConcreteTemplate,
    AbstractTemplate,
)
from numba.core.typing import signature
from numba.extending import (
    models,
    register_model,
)

from numba.cuda.stubs import Stub, stub_function
from numba.cuda.cudaimpl import lower, lower_attr
from numba.cuda.cudadecl import register_attr, register, register_global
from numba.cuda import declare_device
from numba.core.typing.npydecl import parse_dtype
from numba.core.errors import RequireLiteralValue

from ast_canopy.decl import (
    bindings,
    StructMethod,
    ClassTemplate,
    TemplatedStructMethod,
    Function,
    FunctionTemplate,
)

from numbast.types import (
    CTYPE_MAPS as C2N,
    NUMBA_TO_CTYPE_MAPS as N2C,
    to_numba_type,
    FunctorType,
)
from numbast.utils import make_device_caller_with_nargs, deduplicate_overloads
from numbast.struct import struct_method_shim_layer_template
from numbast.shim_writer import MemoryShimWriter as ShimWriter
from numbast.functor import Functor

template_class_method_shim = """
extern "C" __device__ int
{func_name}(int &ignored, {name}<{tparam_names}> *ptr) {{
    new (ptr) {name}<{tparam_names}>();
    return 0;
}}
"""


class ConcreteType(types.Type):
    """A complete type, which is a specialization of a template.
    The specialized parameters are stored in self.args.
    """

    def __init__(self, template_name, tparam_type_names, params):
        # Type name is the UID of the type instance,
        # The ID for each instantiated type should relate to its template parameters.
        # The following creates a unique ID for each type instance.

        assert len(tparam_type_names) == len(params)
        self.tparam_type_names = tparam_type_names

        self.tparams_strs = []
        for tparam in params:
            if nbtype := parse_dtype(tparam):
                self.tparams_strs.append(str(N2C[nbtype]))
            else:
                self.tparams_strs.append(str(tparam))

        instantiated_type_uid = (
            f"{template_name}_{'_'.join(self.tparams_strs)}_ConcreteType"
        ).replace(" ", "_")
        super().__init__(name=instantiated_type_uid)
        self.args = params
        self.template_name = template_name

    def ctor_name(self):
        """The name of this instantiated type."""
        return self.name.strip("_ConcreteType")

    def comma_separated_tparams_str(self):
        return ", ".join(self.tparams_strs)

    def get_actual_type_from_tparam_type_name(self, tparam_type_name):
        """Given a tparam type name, return the actual type name in this ConcreteType.
        Returns None if not found.
        """
        for i, instantiated_tparam_name in enumerate(self.tparam_type_names):
            if tparam_type_name == instantiated_tparam_name:
                return self.tparams_strs[i]

        return None

    def _instantiate(
        self, return_type: bindings.Type, params: list[bindings.ParamVar]
    ) -> tuple[bindings.Type, list[bindings.ParamVar]]:
        new_return_type = return_type
        if return_type.name.startswith("type-parameter"):
            if return_type_name := self.get_actual_type_from_tparam_type_name(
                return_type.name
            ):
                new_return_type = bindings.Type(
                    return_type_name, return_type_name, False, False
                )

        new_params = []
        for param in params:
            new_param_type = param.type_
            if param.type_.name.startswith("type-parameter"):
                if param_type_name := self.get_actual_type_from_tparam_type_name(
                    param.type_.name
                ):
                    new_param_type = bindings.Type(
                        param_type_name, param_type_name, False, False
                    )
            new_param_var = bindings.ParamVar(param.name, new_param_type)

            new_params.append(new_param_var)

        return new_return_type, new_params

    def instantiate_method(self, decl: TemplatedStructMethod) -> StructMethod:
        name = decl.decl_name

        kind = decl.kind
        is_move_constructor = decl.is_move_constructor

        new_return_type, new_params = self._instantiate(decl.return_type, decl.params)

        return StructMethod(
            name, new_return_type, new_params, kind, is_move_constructor
        )

    def instantiate_templated_method(self, decl: Function) -> Function:
        new_return_type, new_params = self._instantiate(decl.return_type, decl.params)

        return Function(decl.name, new_return_type, new_params)

    def cxx_name(self):
        """The name of this instantiated type in C++."""
        return self.template_name + f"<{self.comma_separated_tparams_str()}>"


@register_model(ConcreteType)
class ConcreteModel(models.OpaqueModel):
    def __init__(self, dmm, fe_type):
        models.OpaqueModel.__init__(self, dmm, fe_type)


def make_template_stub_with_nargs(name: str, ntparam: int):
    """Create a template stub with `ntparam` for the `instance` method.

    `ntparam` should be the number of template parameters from the C++ side.

    Parameters
    ----------
    name : str
        The name of the template stub.
    ntparam : int
        The number of template parameters, which is the number of arguments
        of the `.instance` method of the generated stub object.

    Returns
    -------
    stub_obj : Stub
        The stub object.
    """

    args = ", ".join(f"arg{i}" for i in range(ntparam))

    func = f"""
from numba.cuda.stubs import Stub, stub_function
class {name}(Stub):
    @stub_function
    def instance({args}):
        pass
"""

    globals = {}
    exec(func, globals)
    stub_obj = globals[name]

    assert issubclass(stub_obj, Stub)
    return stub_obj


def make_typer_with_argument_names(template_name: str, names: list[str]):
    """

    Parameters
    ----------

    Returns
    -------
    """

    args = ", ".join(names)
    print(args)

    func = f"""
from numba import types
from numba.core.errors import RequireLiteralValue
from numba.core.typing.npydecl import parse_dtype
def typer({args}):
    for arg in [{args}]:
        if isinstance(arg, types.Integer) and not isinstance(arg, types.IntegerLiteral):
            raise RequireLiteralValue(arg)
    instantiated_args = []
    for arg in [{args}]:
        if isinstance(arg, types.IntegerLiteral):
            instantiated_args.append(arg.literal_value)
        else:
            instantiated_args.append(parse_dtype(arg))
    return ConcreteType("{template_name}", *instantiated_args)
"""

    globals = {"ConcreteType": ConcreteType}
    exec(func, globals)
    func_obj = globals["typer"]

    return func_obj


def bind_cxx_class_template(
    class_template: ClassTemplate,
    shim_writer: ShimWriter,
):
    """
    Bind a C++ class template.

    Parameters
    ----------
    class_template : bindings.ClassTemplate
        The C++ class template to be bound.

    Returns
    -------
    stub : object
        The Python API of the class template.
    shim: str
        The shim code for the class template.
    """

    class CT(Stub):
        @stub_function
        def instance():
            pass

    CT.__name__ = class_template.record.name

    @register
    class InstanceDecl(CallableTemplate):
        key = CT.instance

        def generic(self):
            def typer(T, BLOCK_DIM_X):
                if not isinstance(BLOCK_DIM_X, types.IntegerLiteral):
                    raise RequireLiteralValue(BLOCK_DIM_X)

                tparam_type_names = [
                    tparam.type_.name for tparam in class_template.template_parameters
                ]
                tparam_type_names = tparam_type_names[
                    : class_template.num_min_required_args
                ]  # FIXME: only support 2 template parameters for now.

                instance = ConcreteType(
                    class_template.record.name,
                    tparam_type_names,
                    [T, BLOCK_DIM_X.literal_value],
                )

                # Typing and lowering for default constructor:
                class RecordType(types.Type):
                    """The type of the instantiated, constructed record."""

                    def __init__(self, instance):
                        super().__init__(name=instance.ctor_name() + "_Type")
                        self.instance = instance

                @register_model(RecordType)
                class RecordModel(models.OpaqueModel):
                    def __init__(self, dmm, fe_type):
                        models.OpaqueModel.__init__(self, dmm, fe_type)

                record_type = RecordType(instance)

                @register
                class InstanceCall(AbstractTemplate):
                    key = instance

                    def generic(self, args, kws):
                        return signature(record_type)  # FIXME, missing other ctors

                @lower_builtin(instance)
                def lower_instance(context, builder, sig, args, instance=instance):
                    shim_name = instance.ctor_name() + "_ctor"

                    ctor_shim_call = declare_device(
                        shim_name,
                        types.int32(types.CPointer(instance)),
                    )

                    def ctor_func(ptr):
                        return ctor_shim_call(ptr)

                    ptr = builder.alloca(context.get_value_type(instance))
                    context.compile_internal(
                        builder,
                        ctor_func,
                        signature(types.int32, types.CPointer(instance)),
                        (ptr,),
                    )

                    # shim
                    shim_txt = template_class_method_shim.format(
                        func_name=shim_name,
                        name=class_template.record.name,
                        tparam_names=instance.comma_separated_tparams_str(),
                    )

                    shim_writer.write_to_shim(shim_txt, shim_name)

                    return builder.load(ptr)

                # Bindings for non-ctor methods
                # Maps methods name to their argument lists
                overloads: dict[str, list] = defaultdict(list)
                for method in class_template.record.methods:
                    if method.decl_name == class_template.record.name:
                        # Let's ignore the constructor for now.
                        continue

                    instantiated_method = instance.instantiate_method(method)

                    param_types = [C2N[T.name] for T in instantiated_method.param_types]
                    return_type = C2N[instantiated_method.return_type.name]

                    # Cache the overloads and type later
                    overloads[method.decl_name].append([return_type, *param_types])

                    # Lowering
                    @lower(
                        record_type.name + "." + method.decl_name,
                        record_type,
                        *param_types,
                    )
                    def lower_method(
                        context,
                        builder,
                        sig,
                        args,
                        instantiated_method=instantiated_method,
                        record_type=record_type,
                        return_type=return_type,
                        param_types=param_types,
                    ):
                        # Make shim
                        shim_name = f"__{instance.ctor_name()}__{instantiated_method.mangled_name}"
                        shim_name = deduplicate_overloads(shim_name)

                        shim_func = declare_device(
                            shim_name,
                            return_type(
                                types.CPointer(record_type),
                                *map(types.CPointer, param_types),
                            ),
                        )

                        shim_func_call = make_device_caller_with_nargs(
                            shim_name + "_shim", 1 + len(param_types), shim_func
                        )

                        arglist = ", ".join(
                            f"{arg.type_.unqualified_non_ref_type_name}* {arg.name}"
                            for arg in instantiated_method.params
                        )
                        if arglist:
                            arglist = ", " + arglist

                        shim = struct_method_shim_layer_template.format(
                            func_name=shim_name,
                            name=record_type.instance.cxx_name(),
                            arglist=arglist,
                            method_name=instantiated_method.name,
                            args=", ".join(
                                "*" + arg.name for arg in instantiated_method.params
                            ),
                            return_type=instantiated_method.return_type.name,
                        )

                        shim_writer.write_to_shim(shim, shim_name)

                        # IR:
                        selfptr = builder.alloca(context.get_value_type(record_type))
                        builder.store(args[0], selfptr)

                        param_tys = sig.args[1:]
                        params = args[1:]
                        argptrs = [
                            builder.alloca(context.get_value_type(arg))
                            for arg in param_tys
                        ]
                        for ptr, ty, arg in zip(argptrs, param_tys, params):
                            if hasattr(ty, "decl"):
                                builder.store(arg, ptr, align=ty.decl.alignof_)
                            else:
                                builder.store(arg, ptr)

                        return context.compile_internal(
                            builder,
                            shim_func_call,
                            signature(
                                return_type,
                                types.CPointer(record_type),
                                *map(types.CPointer, param_types),
                            ),
                            (selfptr, *argptrs),
                        )

                # Bindings for method templates:
                # Maps from template names to a list of dictionaries, where keys are
                # string of serialized tparams, and values are list of argument types.
                templated_method_overloads: dict[str, list[FunctionTemplate]] = (
                    defaultdict(list)
                )
                for templated_method in class_template.record.templated_methods:
                    # TODO: remove
                    if (
                        templated_method.function.name == "Reduce"
                        and len(templated_method.function.params) == 2
                        and templated_method.function.params[0].name == "input"
                    ):
                        templated_method_overloads[
                            templated_method.function.name
                        ].append(templated_method)

                # Typing for methods
                @register_attr
                class RecordTemplate(AttributeTemplate):
                    key = record_type

                for method_name, argtyslist in overloads.items():
                    sigs = [
                        signature(*argtys, recvr=record_type) for argtys in argtyslist
                    ]

                    class RecordMethodDecl(ConcreteTemplate):
                        key = record_type.name + "." + method_name
                        cases = sigs

                    def resolve(self, cls):
                        return types.BoundFunction(RecordMethodDecl, record_type)

                    setattr(RecordTemplate, "resolve_" + method_name, resolve)

                for method_template_name, decls in templated_method_overloads.items():
                    for decl in decls:

                        class MethodInstanceObj(Stub):
                            @stub_function
                            def instance():
                                pass

                        MethodInstanceObj.__name__ = (
                            record_type.name + "_" + method_template_name
                        )

                        def resolve(self, mod, ty=types.Module(MethodInstanceObj)):
                            return ty

                        setattr(
                            RecordTemplate, "resolve_" + method_template_name, resolve
                        )

                        @lower_attr(record_type, method_template_name)
                        def lower_method_template(context, builder, sig, args):
                            # No-op
                            # `args` is the constructed record type ir.
                            if not hasattr(context, "record_index"):
                                context.record_index = {}
                            context.record_index[record_type.name] = args
                            return context.get_constant(types.int32, 0)

                        tparam_type_names = "_".join(
                            tp.type_.name for tp in decl.template_parameters
                        )

                        @register
                        class MethodInstanceDecl(AbstractTemplate):
                            key = (
                                MethodInstanceObj.__name__
                                + tparam_type_names
                                + ".instance"
                            )

                            def generic(self, args, kws, decl=decl):
                                # Instantiation complete

                                tparam_type_names = [
                                    tp.type_.name for tp in decl.template_parameters
                                ]

                                instantiated_method_type = ConcreteType(
                                    self.key,
                                    tparam_type_names,
                                    args,
                                )

                                # We need some kind of hierarchical tparam resolution here.
                                ins_func1 = instance.instantiate_templated_method(
                                    decl.function
                                )

                                instantiated_method = instantiated_method_type.instantiate_templated_method(
                                    ins_func1
                                )

                                @lower(self.key, *args)
                                def lower_method_template_instance(
                                    context, builder, sig, args
                                ):
                                    # No-op
                                    return context.get_constant(types.int32, 0)

                                @register
                                class InstantiatedMethodCall(AbstractTemplate):
                                    key = instantiated_method_type

                                    def generic(
                                        self,
                                        args,
                                        kws,
                                        instantiated_method=instantiated_method,
                                    ):
                                        param_types = [
                                            to_numba_type(argty.name)
                                            for argty in instantiated_method.param_types
                                        ]

                                        @lower(
                                            self.key,
                                            *param_types,
                                        )
                                        def lower_method(
                                            context,
                                            builder,
                                            sig,
                                            args,
                                            instantiated_method=instantiated_method,
                                        ):
                                            param_types = [
                                                to_numba_type(argty.name)
                                                for argty in instantiated_method.param_types
                                            ]

                                            # find functor obj, assume only 1 functor for now.
                                            fntr: typing.Optional[Functor] = None
                                            for ty, arg in zip(sig.args, args):
                                                if isinstance(ty, FunctorType):
                                                    fntr = Functor.functor_maps[
                                                        arg.constant
                                                    ]
                                                    break

                                            # Generate shim to functor
                                            if fntr:
                                                ptx = fntr.compile_ptx()
                                                functor_shim = fntr.shim()
                                                shim_writer.write_to_shim(
                                                    functor_shim,
                                                    fntr.name + "Functor",
                                                )
                                                shim_writer.write_to_ptx_shim(
                                                    ptx, fntr.name
                                                )

                                            # Generate shim to struct method
                                            param_type_str = "_".join(
                                                str(ty) for ty in param_types
                                            )

                                            shim_func_name = "_".join(
                                                [
                                                    instance.name,
                                                    instantiated_method.name,
                                                    param_type_str,
                                                    "shim",
                                                ]
                                            )

                                            param_types_without_fctr = [
                                                ty
                                                for ty in param_types
                                                if not isinstance(ty, FunctorType)
                                            ]

                                            shim_func = declare_device(
                                                shim_func_name,
                                                signature(
                                                    C2N[
                                                        instantiated_method.return_type.name
                                                    ],
                                                    types.CPointer(record_type),
                                                    *map(
                                                        types.CPointer,
                                                        param_types_without_fctr,
                                                    ),
                                                ),
                                            )

                                            shim_func_call = (
                                                make_device_caller_with_nargs(
                                                    shim_func_name + "_shim",
                                                    1 + len(param_types_without_fctr),
                                                    shim_func,
                                                )
                                            )

                                            arglist = ", ".join(
                                                f"{arg.type_.unqualified_non_ref_type_name}* {arg.name}"
                                                for arg, ty in zip(
                                                    instantiated_method.params,
                                                    param_types,
                                                )
                                                if not isinstance(ty, FunctorType)
                                            )
                                            if arglist:
                                                arglist = ", " + arglist

                                            params_strs = []
                                            for arg, ty in zip(
                                                instantiated_method.params, param_types
                                            ):
                                                if isinstance(ty, FunctorType):
                                                    # Note that we assume only 1 functor for now.
                                                    params_strs.append(
                                                        fntr.name + "Functor{}"
                                                    )
                                                else:
                                                    params_strs.append("*" + arg.name)
                                            paramstr = ", ".join(params_strs)

                                            shim = struct_method_shim_layer_template.format(
                                                func_name=shim_func_name,
                                                name=record_type.instance.cxx_name(),
                                                arglist=arglist,
                                                method_name=instantiated_method.name,
                                                args=paramstr,
                                                return_type=instantiated_method.return_type.name,
                                            )

                                            shim_writer.write_to_shim(
                                                shim, shim_func_name
                                            )

                                            # IR:

                                            selfptr = builder.alloca(
                                                context.get_value_type(record_type)
                                            )
                                            selfir = context.record_index[
                                                record_type.name
                                            ]
                                            builder.store(selfir, selfptr)

                                            param_tys = []
                                            params = []
                                            for ty, arg in zip(sig.args, args):
                                                if not isinstance(ty, FunctorType):
                                                    param_tys.append(ty)
                                                    params.append(arg)
                                            argptrs = [
                                                builder.alloca(
                                                    context.get_value_type(arg)
                                                )
                                                for arg in param_tys
                                            ]
                                            for ptr, ty, arg in zip(
                                                argptrs, param_tys, params
                                            ):
                                                if hasattr(ty, "decl"):
                                                    builder.store(
                                                        arg, ptr, align=ty.decl.alignof_
                                                    )
                                                else:
                                                    builder.store(arg, ptr)

                                            return context.compile_internal(
                                                builder,
                                                shim_func_call,
                                                signature(
                                                    to_numba_type(
                                                        instantiated_method.return_type.name
                                                    ),
                                                    types.CPointer(record_type),
                                                    *map(
                                                        types.CPointer,
                                                        param_types_without_fctr,
                                                    ),
                                                ),
                                                (selfptr, *argptrs),
                                            )

                                        return signature(
                                            to_numba_type(
                                                instantiated_method.return_type.name
                                            ),
                                            *param_types,
                                        )

                                return signature(
                                    instantiated_method_type,
                                    *args,
                                )

                        @register_attr
                        class MethodTemplateDecl(AttributeTemplate):
                            key = types.Module(MethodInstanceObj)

                            def resolve_instance(self, mod):
                                return types.Function(MethodInstanceDecl)

                self.context.refresh()
                print("Registered Instance")
                return instance

            return typer

    @register_attr
    class TemplateDecl(AttributeTemplate):
        key = types.Module(CT)

        def resolve_instance(self, mod):
            return types.Function(InstanceDecl)

    @lower_builtin(CT.instance, types.Any, types.Any)
    def lower_instance(context, builder, sig, args):
        return context.get_constant(types.int32, 0)

    register_global(CT, types.Module(CT))

    return CT
