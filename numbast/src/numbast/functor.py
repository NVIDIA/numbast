# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from numba import types
from numba.cuda.extending import typeof_impl
from numba.cuda.core.imputils import lower_constant
from numba import cuda

from numbast.types import NUMBA_TO_CTYPE_MAPS as N2C, FunctorType

functor_shim = """
extern "C"
{RT} __device__ {functor}({T1}, {T2});

struct {functor}Functor {{
  {RT} __device__ operator()({T1}& a, {T2}&b) {{
    return {functor}(a, b);
  }}
}};
"""


class Functor:
    functor_maps = {}
    reverse_functor_maps = {}

    def __init__(self, func, sig):
        self.fid = len(Functor.functor_maps)
        self.func = func
        self.sig = sig
        Functor.functor_maps[self.fid] = self
        Functor.reverse_functor_maps[self] = self.fid

    def __call__(self, *args):
        return self.func(*args)

    @staticmethod
    def get_id(func):
        return Functor.reverse_functor_maps[func]

    def shim(self):
        """Return the C++ shim source for invoking the functor as operator()."""
        return functor_shim.format(
            RT=N2C[self.sig.return_type],
            functor=self.name,
            T1=N2C[self.sig.args[0]],
            T2=N2C[self.sig.args[1]],
            T=self.sig.return_type,
        )

    def c_params(self):
        return [N2C[arg] for arg in self.sig.args]

    @property
    def name(self):
        c_params_str = "_".join(self.c_params())
        c_params_str_normalized = c_params_str.replace(" ", "_")
        return self.func.__name__ + c_params_str_normalized

    def compile_ptx(self):
        cc = cuda.get_current_device().compute_capability
        ptx, _ = cuda.compile_ptx(
            self.func,
            self.sig,
            device=True,
            cc=cc,
            abi="c",
            abi_info={"abi_name": self.name},
        )
        return ptx


@typeof_impl.register(Functor)
def typeof_functor(val: Functor, c):
    return FunctorType(val.func.__name__)


@lower_constant(FunctorType)
def constant_functortype(context, builder, ty, pyval):
    return context.get_constant(types.int64, Functor.get_id(pyval))


def cpp_functor(sig):
    """Decorator that wraps a Python function as a C++-compatible functor."""

    def wrapper(func):
        return Functor(func, sig)

    return wrapper
