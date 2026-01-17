from numbast.args import prepare_ir_types
from numbast.intent import IntentPlan
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
        *,
        arg_is_ref: list[bool] | None = None,
        intent_plan: IntentPlan | None = None,
        out_return_types: list[types.Type] | None = None,
        cxx_return_type: types.Type | None = None,
    ):
        super().__init__(itanium_mangled_name, shim_writer, shim_code)
        self._arg_is_ref = list(arg_is_ref) if arg_is_ref is not None else None
        self._intent_plan = intent_plan
        self._out_return_types = (
            list(out_return_types) if out_return_types is not None else None
        )
        self._cxx_return_type = cxx_return_type

    def _lower_impl(self, builder, context, sig, args):
        # Numba-visible return type may differ from the underlying C++ return type
        # when out_return parameters are enabled (tuple returns, etc.).
        cxx_return_type = (
            self._cxx_return_type
            if self._cxx_return_type is not None
            else sig.return_type
        )
        # 1. Prepare return value pointer
        if cxx_return_type == types.void:
            # Void return type in C++ is shimmed as int& ignored
            retval_ty = ir.IntType(32)
            retval_ptr = builder.alloca(retval_ty, name="ignored")
        else:
            retval_ty = context.get_value_type(cxx_return_type)
            retval_ptr = builder.alloca(retval_ty, name="retval")

        # 2. Prepare arguments
        if self._intent_plan is None:
            pass_ptr_mask = (
                self._arg_is_ref
                if self._arg_is_ref is not None
                else [False] * len(sig.args)
            )
            arg_pointer_types = prepare_ir_types(
                context, sig.args, pass_ptr_mask=pass_ptr_mask
            )
        else:
            plan = self._intent_plan
            if len(sig.args) != len(plan.visible_param_indices):
                raise ValueError(
                    "Signature args do not match intent plan visible params: "
                    f"sig has {len(sig.args)} args but plan expects {len(plan.visible_param_indices)}"
                )
            if len(plan.pass_ptr_mask) != len(sig.args):
                raise ValueError(
                    "Intent plan pass_ptr_mask length does not match signature args length: "
                    f"{len(plan.pass_ptr_mask)} != {len(sig.args)}"
                )
            if plan.out_return_indices:
                if self._out_return_types is None:
                    raise ValueError(
                        "out_return intent plan requires out_return_types to be provided"
                    )
                if len(self._out_return_types) != len(plan.out_return_indices):
                    raise ValueError(
                        "out_return_types length does not match intent plan out_return_indices: "
                        f"{len(self._out_return_types)} != {len(plan.out_return_indices)}"
                    )
            arg_pointer_types = []  # computed below alongside ptrs

        # ABI:
        # - default: pass pointer-to-value to shim (alloca + store)
        # - for C++ reference args mapped to CPointer(T): pass pointer value directly
        ptrs = []
        out_return_ptrs: list[tuple[types.Type, ir.Value]] = []
        if self._intent_plan is None:
            for argty, arg, passthrough in zip(sig.args, args, pass_ptr_mask):
                vty = context.get_value_type(argty)
                if passthrough and isinstance(vty, ir.PointerType):
                    ptrs.append(arg)
                else:
                    ptr = cgutils.alloca_once(builder, vty)
                    builder.store(
                        arg, ptr, align=getattr(argty, "alignof_", None)
                    )
                    ptrs.append(ptr)
        else:
            plan = self._intent_plan
            n_orig = len(plan.intents)
            # Map original parameter index -> visible signature position / out_return position
            orig_to_vis = [None] * n_orig
            for vis_pos, orig_idx in enumerate(plan.visible_param_indices):
                orig_to_vis[orig_idx] = vis_pos
            orig_to_out = [None] * n_orig
            for out_pos, orig_idx in enumerate(plan.out_return_indices):
                orig_to_out[orig_idx] = out_pos

            for orig_idx in range(n_orig):
                out_pos = orig_to_out[orig_idx]
                if out_pos is not None:
                    out_nbty = self._out_return_types[out_pos]
                    vty = context.get_value_type(out_nbty)
                    ptr = cgutils.alloca_once(builder, vty)
                    ptrs.append(ptr)
                    arg_pointer_types.append(ir.PointerType(vty))
                    out_return_ptrs.append((out_nbty, ptr))
                    continue

                vis_pos = orig_to_vis[orig_idx]
                if vis_pos is None:
                    raise ValueError(
                        f"Internal error: original param {orig_idx} is neither visible nor out_return"
                    )
                argty = sig.args[vis_pos]
                arg = args[vis_pos]
                passthrough = bool(plan.pass_ptr_mask[vis_pos])
                vty = context.get_value_type(argty)
                if passthrough and isinstance(vty, ir.PointerType):
                    ptrs.append(arg)
                    arg_pointer_types.append(vty)
                else:
                    ptr = cgutils.alloca_once(builder, vty)
                    builder.store(
                        arg, ptr, align=getattr(argty, "alignof_", None)
                    )
                    ptrs.append(ptr)
                    arg_pointer_types.append(ir.PointerType(vty))

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
        if (
            self._intent_plan is None
            or not self._intent_plan.out_return_indices
        ):
            if cxx_return_type == types.void:
                return None
            return builder.load(
                retval_ptr, align=getattr(cxx_return_type, "alignof_", None)
            )

        # out_return enabled: return either a value or a tuple (ret, out1, out2, ...)
        ret_vals: list[ir.Value] = []
        if cxx_return_type != types.void:
            ret_vals.append(
                builder.load(
                    retval_ptr, align=getattr(cxx_return_type, "alignof_", None)
                )
            )
        for out_ty, out_ptr in out_return_ptrs:
            ret_vals.append(
                builder.load(out_ptr, align=getattr(out_ty, "alignof_", None))
            )

        # If Numba-visible return is a tuple, use context.make_tuple.
        # Otherwise (void + single out), return the single out value directly.
        if hasattr(sig.return_type, "types"):
            return context.make_tuple(builder, sig.return_type, ret_vals)
        if len(ret_vals) != 1:
            raise ValueError(
                "Non-tuple return type requires exactly one return value; "
                f"got {len(ret_vals)}"
            )
        return ret_vals[0]
