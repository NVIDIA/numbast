from numbast.shim_writer import ShimWriterBase
from numbast.args import prepare_ir_types
from numba.cuda import types, cgutils

from llvmlite import ir


class BaseCallConv:
    shim_function_template = "{mangled_name}_nbst"

    def __init__(
        self,
        itanium_mangled_name: str,
        shim_writer: ShimWriterBase,
        shim_code: str,
    ):
        self.shim_writer = shim_writer
        self.itanium_mangled_name = itanium_mangled_name
        self.shim_code = shim_code

        self.shim_function_name = self.shim_function_template.format(
            mangled_name=self.itanium_mangled_name
        )

    def _lazy_write_shim(self, shim_code: str):
        self.shim_writer.write_to_shim(shim_code, self.shim_function_name)

    def _lower(self, builder, context, sig, args):
        print(f"{self.shim_code=}")
        self._lazy_write_shim(self.shim_code)
        return self._lower_impl(builder, context, sig, args)

    def _lower_impl(self, builder, context, sig, args):
        raise NotImplementedError

    def __call__(self, builder, context, sig, args):
        return self._lower(builder, context, sig, args)


class RecordCtorCallConv(BaseCallConv):
    def __init__(
        self,
        itanium_mangled_name: str,
        shim_writer: ShimWriterBase,
        shim_code: str,
        s_type: types.Type,
    ):
        super().__init__(itanium_mangled_name, shim_writer, shim_code)
        self.self_type = s_type

    def _lower_impl(self, builder, context, sig, args):
        """Lower a record constructor call to shim function.

        The shim function uses a placement new to construct the record on a pointer passed as the first argument.
        In lowering, we prepare the place for the record to be constructed on the stack and pass the pointer to
        the shim function. The return value is the loaded value of the record pointer.

        Parameters
        ----------
        builder : llvmlite.IRBuilder
            The builder to use for the IR.
        context : numba.cuda.context.CUDATargetContext
            The context to use for the IR.
        sig : numba.cuda.typing.templates.Signature
            The signature of the constructor call to lower.
        args : list
            The argument values to the constructor call

        Returns
        -------
        res : llvmlite.IRValue
            The loaded value of the record pointer.

        """
        selfty = context.get_value_type(self.self_type)
        selfptr = builder.alloca(selfty, name="selfptr")

        # Declare shim function
        arg_pointer_types = prepare_ir_types(context, sig.args)
        fnty = ir.FunctionType(
            ir.IntType(32),
            [ir.PointerType(ir.IntType(32)), ir.PointerType(selfty)]
            + arg_pointer_types,
        )
        fn = cgutils.get_or_insert_function(
            builder.module, fnty, self.shim_function_name
        )

        # Prepare arguments as pointers
        # alloca_once should accept align parameter
        ignored = builder.alloca(ir.IntType(32), name="ignored")
        ptrs = [
            cgutils.alloca_once(builder, context.get_value_type(argty))
            for argty in sig.args
        ]
        for ptr, argty, arg in zip(ptrs, sig.args, args):
            builder.store(arg, ptr, align=getattr(argty, "alignof_", None))

        builder.call(fn, (ignored, selfptr, *args))
        return builder.load(
            selfptr, align=getattr(self.self_type, "alignof_", None)
        )
