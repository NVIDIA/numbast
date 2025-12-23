from numbast.args import prepare_ir_types
from numba.cuda import types, cgutils

from llvmlite import ir


class BaseCallConv:
    shim_function_template = "{mangled_name}_nbst"

    def __init__(
        self,
        itanium_mangled_name: str,
        shim_writer: object,
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
        self._lazy_write_shim(self.shim_code)
        return self._lower_impl(builder, context, sig, args)

    def _lower_impl(self, builder, context, sig, args):
        raise NotImplementedError

    def __call__(self, builder, context, sig, args):
        return self._lower(builder, context, sig, args)


class FunctionCallConv(BaseCallConv):
    def __init__(
        self,
        itanium_mangled_name: str,
        shim_writer: object,
        shim_code: str,
        return_type: types.Type,
    ):
        super().__init__(itanium_mangled_name, shim_writer, shim_code)
        self.return_type = return_type

    def _lower_impl(self, builder, context, sig, args):
        # 1. Prepare return value pointer
        if self.return_type == types.void:
            # Void return type in C++ is shimmed as int& ignored
            retval_ty = ir.IntType(32)
            retval_ptr = builder.alloca(retval_ty, name="ignored")
        else:
            retval_ty = context.get_value_type(self.return_type)
            retval_ptr = builder.alloca(retval_ty, name="retval")

        # 2. Prepare arguments
        arg_pointer_types = prepare_ir_types(context, sig.args)

        # All arguments are passed by pointer
        ptrs = [
            cgutils.alloca_once(builder, context.get_value_type(argty))
            for argty in sig.args
        ]
        for ptr, argty, arg in zip(ptrs, sig.args, args):
            builder.store(arg, ptr, align=getattr(argty, "alignof_", None))

        # 3. Declare shim
        # Shim signature: int (retval_type*, arg0_type*, ...)
        fnty = ir.FunctionType(
            ir.IntType(32), [ir.PointerType(retval_ty)] + arg_pointer_types
        )
        fn = cgutils.get_or_insert_function(
            builder.module, fnty, self.shim_function_name
        )

        # 4. Call shim
        builder.call(fn, (retval_ptr, *ptrs))

        # 5. Return
        if self.return_type == types.void:
            return None
        else:
            return builder.load(
                retval_ptr, align=getattr(self.return_type, "alignof_", None)
            )
