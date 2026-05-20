# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from numbast.experimental.mlir.args import prepare_ir_types
from numbast.experimental.mlir.intent import IntentPlan

# NBST:BEGIN_CALLCONV
from numba_cuda_mlir import types
from numba_cuda_mlir.lowering_utilities import get_or_insert_function

from numba_cuda_mlir._mlir import ir
from numba_cuda_mlir._mlir.dialects import func, llvm


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

        # Ensure in-memory link objects (e.g. cuda.CUSource snapshots) already
        # include this shim when users call shim_writer.links() at decoration time.
        self._lazy_write_shim(self.shim_code)

    def _lazy_write_shim(self, shim_code: str):
        self.shim_writer.write_to_shim(shim_code, self.shim_function_name)

    def _lower(self, builder, target, args, kws):
        self._lazy_write_shim(self.shim_code)
        return self._lower_impl(builder, target, args, kws)

    def _lower_impl(self, builder, target, args, kws):
        raise NotImplementedError

    def __call__(self, builder, target, args, kws):
        return self._lower(builder, target, args, kws)


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
        """
        Initialize a FunctionCallConv with shim information and optional ABI/intent hints.

        Parameters:
            itanium_mangled_name (str): The Itanium-mangled C++ function name used to derive the shim name.
            shim_writer (object): Writer used to emit the shim code when required.
            shim_code (str): LLVM/IR code template for the shim.
            arg_is_ref (list[bool] | None): Per-argument mask indicating whether an argument should be passed as a pointer (True) or by value (False). If None, pointer-passing is determined later or defaults to all False.
            intent_plan (IntentPlan | None): Optional plan describing visible parameter indices, which parameters should be passed as pointers, and which parameters are out-returns; when present it drives argument mapping and out-return handling.
            out_return_types (list[types.Type] | None): Types of the out-return values in the order declared by the IntentPlan; required when the intent_plan defines out-return indices.
            cxx_return_type (types.Type | None): The C++ ABI return type to use for allocating/shimming the return slot; if None, the signature's return type is used.
        """
        super().__init__(itanium_mangled_name, shim_writer, shim_code)
        self._arg_is_ref = list(arg_is_ref) if arg_is_ref is not None else None
        self._intent_plan = intent_plan
        self._out_return_types = (
            list(out_return_types) if out_return_types is not None else None
        )
        self._cxx_return_type = cxx_return_type

    def _lower_impl(self, builder, target, args, kws):
        # Numba-visible return type may differ from the underlying C++ return type
        # when out_return parameters are enabled (tuple returns, etc.).
        """
        Lower the configured call into a shim invocation, preparing return and argument pointers according to arg_is_ref or an IntentPlan and materializing the final Numba-visible return value.

        Parameters:
            builder: LLVM IR builder used to emit allocations, stores, and calls.
            context: Compilation context used to map numba types to LLVM value types and to construct tuple return values.
            sig: Numba function signature describing the visible parameter and return types.
            args: Sequence of LLVM IR values corresponding to the visible signature parameters.

        Returns:
            The loaded return value(s) according to the signature and intent plan:
            - `None` if the effective C++ return type is void and no out-return values are present.
            - A single LLVM value for a single visible return.
            - A tuple object constructed via `context.make_tuple` when the visible return type is a tuple; the tuple contains the C++ return (if non-void) followed by any out-return values.

        Raises:
            ValueError: if the provided IntentPlan does not align with `sig` (mismatched visible_param_indices or pass_ptr_mask), if `out_return_types` are required but missing or length-mismatched, or if a non-tuple visible return is expected but multiple return values are produced.
        """
        cxx_return_type = (
            self._cxx_return_type
            if self._cxx_return_type is not None
            else builder.get_numba_type(target)
        )
        arg_types = [builder.get_numba_type(arg.name) for arg in args]
        # 1. Prepare return value pointer
        if cxx_return_type == types.void:
            # Void return type in C++ is shimmed as int& ignored
            retval_ty = ir.IntegerType.get_signless(32)
            retval_ptr = builder.alloca(retval_ty)
        else:
            retval_ty = builder.get_mlir_type(cxx_return_type)
            retval_ptr = builder.alloca(retval_ty)

        # 2. Prepare arguments
        if self._intent_plan is None:
            pass_ptr_mask = (
                self._arg_is_ref
                if self._arg_is_ref is not None
                else [False] * len(args)
            )
            arg_pointer_types = prepare_ir_types(
                builder, arg_types, pass_ptr_mask=pass_ptr_mask
            )
        else:
            plan = self._intent_plan
            if len(args) != len(plan.visible_param_indices):
                raise ValueError(
                    "Signature args do not match intent plan visible params: "
                    f"sig has {len(args)} args but plan expects {len(plan.visible_param_indices)}"
                )
            if len(plan.pass_ptr_mask) != len(args):
                raise ValueError(
                    "Intent plan pass_ptr_mask length does not match signature args length: "
                    f"{len(plan.pass_ptr_mask)} != {len(args)}"
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
            for argty, arg, passthrough in zip(arg_types, args, pass_ptr_mask):
                arg_value = builder.load_var(arg)
                vty = builder.get_mlir_type(argty)
                if passthrough and isinstance(vty, llvm.PointerType):
                    ptrs.append(arg_value)
                else:
                    ptr = builder.alloca(vty)
                    llvm.store(value=arg_value, addr=ptr)
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
                    vty = builder.get_mlir_type(out_nbty)
                    ptr = builder.alloca(vty)
                    ptrs.append(ptr)
                    arg_pointer_types.append(ptr.type)
                    out_return_ptrs.append((out_nbty, ptr))
                    continue

                vis_pos = orig_to_vis[orig_idx]
                if vis_pos is None:
                    raise ValueError(
                        f"Internal error: original param {orig_idx} is neither visible nor out_return"
                    )
                argty = arg_types[vis_pos]
                arg = args[vis_pos]
                arg_value = builder.load_var(arg)
                passthrough = bool(plan.pass_ptr_mask[vis_pos])
                vty = builder.get_mlir_type(argty)
                if passthrough and isinstance(vty, llvm.PointerType):
                    # This argument is pointerType and is passthrough to API as a pointer.
                    ptrs.append(arg_value)
                    arg_pointer_types.append(vty)
                else:
                    ptr = builder.alloca(vty)
                    llvm.store(value=arg_value, addr=ptr)
                    ptrs.append(ptr)
                    arg_pointer_types.append(ptr.type)

        # 3. Declare shim
        # Shim signature: int (retval_type*, arg0_type*, ...)
        call_status_ty = ir.IntegerType.get_signless(32)
        fnty = ir.FunctionType.get(
            inputs=[retval_ptr.type] + arg_pointer_types,
            results=[call_status_ty],
        )
        fn = get_or_insert_function(
            self.shim_function_name, fnty, builder.mlir_gpu_module
        )
        # 4. Call shim
        func.call(
            result=[call_status_ty],
            callee=fn.name.value,
            operands_=[retval_ptr, *ptrs],
        )

        # 5. Return
        if (
            self._intent_plan is None
            or not self._intent_plan.out_return_indices
        ):
            if cxx_return_type == types.void:
                builder.store_var(target, None)
                return
            result = llvm.load(res=retval_ty, addr=retval_ptr)
            builder.store_var(target, result)
            return

        # out_return enabled: return either a value or a tuple (ret, out1, out2, ...)
        ret_vals: list[ir.Value] = []
        if cxx_return_type != types.void:
            ret_vals.append(llvm.load(res=retval_ty, addr=retval_ptr))
        for out_ty, out_ptr in out_return_ptrs:
            ret_ty = builder.get_mlir_type(out_ty)
            ret_vals.append(llvm.load(res=ret_ty, addr=out_ptr))

        if len(ret_vals) == 1:
            builder.store_var(target, ret_vals[0])
        else:
            builder.store_var(target, tuple(ret_vals))


# NBST:END_CALLCONV
