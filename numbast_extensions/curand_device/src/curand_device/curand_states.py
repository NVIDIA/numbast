# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import operator
import re

from numba import cuda, types
from numba.extending import models, register_model, typeof_impl
from numba.cuda.cudadecl import register_global
from numba.cuda.cudaimpl import lower
from numba.core.typing.templates import AbstractTemplate, signature
import numpy as np

from llvmlite import ir

from numbast.types import CTYPE_MAPS

from ast_canopy.decl import Struct

# C-compatible type mapping, see:
# https://numpy.org/devdocs/reference/arrays.scalars.html#integer-types
ctystr_to_npystr = {
    "char": f"int{np.dtype(np.byte).itemsize * 8}",
    "short": f"int{np.dtype(np.short).itemsize * 8}",
    "int": f"int{np.dtype(np.intc).itemsize * 8}",
    "long": f"int{np.dtype(np.int_).itemsize * 8}",
    "long long": f"int{np.dtype(np.longlong).itemsize * 8}",
    "unsigned char": f"uint{np.dtype(np.ubyte).itemsize * 8}",
    "unsigned short": f"uint{np.dtype(np.ushort).itemsize * 8}",
    "unsigned int": f"uint{np.dtype(np.uintc).itemsize * 8}",
    "unsigned long": f"uint{np.dtype(np.uint).itemsize * 8}",
    "unsigned long long": f"uint{np.dtype(np.ulonglong).itemsize * 8}",
    "float": f"float{np.dtype(np.single).itemsize * 8}",
    "double": f"float{np.dtype(np.double).itemsize * 8}",
}

vector_type_base_to_npystr = {
    "char": f"int{np.dtype(np.byte).itemsize * 8}",
    "short": f"int{np.dtype(np.short).itemsize * 8}",
    "int": f"int{np.dtype(np.intc).itemsize * 8}",
    "long": f"int{np.dtype(np.int_).itemsize * 8}",
    "longlong": f"int{np.dtype(np.longlong).itemsize * 8}",
    "uchar": f"uint{np.dtype(np.ubyte).itemsize * 8}",
    "ushort": f"uint{np.dtype(np.ushort).itemsize * 8}",
    "uint": f"uint{np.dtype(np.uintc).itemsize * 8}",
    "ulong": f"uint{np.dtype(np.uint).itemsize * 8}",
    "ulonglong": f"uint{np.dtype(np.ulonglong).itemsize * 8}",
    "float": f"float{np.dtype(np.single).itemsize * 8}",
    "double": f"float{np.dtype(np.double).itemsize * 8}",
}


def to_numpy_dtype(ctystr: str):
    """Given a C type string, return the corresponding NumPy dtype.
    If the type is an array type, return a tuple of the base dtype and array size.
    Otherwise, return the base dtype and None.
    """
    vector_type_pattern = r"(.*)([1-4])$"
    vector_type_match = re.match(vector_type_pattern, ctystr)

    if vector_type_match:
        base_type, size = vector_type_match.groups()
        dtype = np.dtype(vector_type_base_to_npystr[base_type])
        return (dtype, int(size))
    elif "[" in ctystr:
        base_type = ctystr.split("[")[0]
        array_size = int(ctystr.split("[")[1].split("]")[0])
        dtype = np.dtype(ctystr_to_npystr[base_type])
        return (dtype, array_size)
    else:
        dtype = np.dtype(ctystr_to_npystr[ctystr])
        return (dtype,)


def make_curand_states(curand_state_decl: Struct):
    """Create a boxed curand states object to represent an array of curand state.
    Should invoke after creating the numba types for each curand state.
    """

    curand_state_name = curand_state_decl.name

    state_ty = CTYPE_MAPS[curand_state_name]

    curand_states_name = curand_state_name.replace("State", "States")

    # cuRAND state type as a NumPy dtype - this mirrors the state defined in
    # curand_kernel.h. Can be used to inspect the state through the device array
    # held by CurandStates.

    state_fields = [
        (f.name, *to_numpy_dtype(f.type_.name)) for f in curand_state_decl.fields
    ]

    curandStateDtype = np.dtype(state_fields, align=True)

    # Hold an array of cuRAND states - somewhat analogous to a curandState* in
    # C/C++.

    class CurandStates:
        def __init__(self, n):
            self._array = cuda.device_array(n, dtype=curandStateDtype)

        @property
        def data(self):
            return self._array.__cuda_array_interface__["data"][0]

    CurandStates.__name__ = curand_states_name

    class CurandStatePointer(types.Type):
        def __init__(self, name):
            self.dtype = state_ty
            super().__init__(name=name)

    curand_state_pointer = CurandStatePointer(curand_states_name + "*")

    @typeof_impl.register(CurandStates)
    def typeof_curand_states(val, c):
        return curand_state_pointer

    @register_global(operator.getitem)
    class CurandStatesPointerGetItem(AbstractTemplate):
        def generic(self, args, kws):
            assert not kws
            [ptr, idx] = args
            if ptr == curand_state_pointer:
                return signature(
                    types.CPointer(state_ty), curand_state_pointer, types.int64
                )

    register_model(CurandStatePointer)(models.PointerModel)

    @lower(operator.getitem, curand_state_pointer, types.int64)
    def lower_curand_states_getitem(context, builder, sig, args):
        [ptr, idx] = args

        # Working out my own GEP
        ptrint = builder.ptrtoint(ptr, ir.IntType(64))
        itemsize = curandStateDtype.itemsize
        offset = builder.mul(idx, context.get_constant(types.int64, itemsize))
        ptrint = builder.add(ptrint, offset)
        ptr = builder.inttoptr(ptrint, ptr.type)
        return ptr

    # Argument handling. When a CurandStatePointer is passed into a kernel, we
    # really only need to pass the pointer to the data, not the whole underlying
    # array structure. Our handler here transforms these arguments into a uint64
    # holding the pointer.

    class CurandStateArgHandler:
        def prepare_args(self, ty, val, **kwargs):
            if isinstance(val, CurandStates):
                assert ty == curand_state_pointer
                return types.uint64, val.data
            else:
                return ty, val

    curand_state_arg_handler = CurandStateArgHandler()

    return CurandStates, curand_state_arg_handler
