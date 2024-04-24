# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import numba.cuda as cuda
import torch

if cuda.get_current_device().compute_capability < (8, 0):
    pytest.skip(
        reason="bfloat16 require compute capability 8.0+",
        allow_module_level=True,
    )


from bf16 import (
    nv_bfloat16,
    htrunc,
    hceil,
    hfloor,
    hrint,
    hsqrt,
    hrsqrt,
    hrcp,
    hlog,
    hlog2,
    hlog10,
    hcos,
    hsin,
    hexp,
    hexp2,
    hexp10,
    get_shims,
)

import numpy as np

from numba import int16, int32, int64, uint16, uint32, uint64, float32, float64
from numba.types import float16

from numba.np import numpy_support

# what is the constructor vs what is the numba type ?
numpy_support.FROM_DTYPE[np.dtype("bfloat16")] = nv_bfloat16.nb_type


def test_ctor():
    @cuda.jit(link=get_shims())
    def simple_kernel():
        a = nv_bfloat16(float64(1.0))
        b = nv_bfloat16(float32(2.0))
        c = nv_bfloat16(int16(3))
        d = nv_bfloat16(int32(4))
        e = nv_bfloat16(int64(5))
        f = nv_bfloat16(uint16(6))
        g = nv_bfloat16(uint32(7))
        h = nv_bfloat16(uint64(8))

        i = nv_bfloat16(float16(9))

    simple_kernel[1, 1]()


def test_casts():
    @cuda.jit(link=get_shims())
    def simple_kernel(b, c, d, e, f, g, h):
        a = nv_bfloat16(3.14)

        b[0] = float32(a)
        c[0] = int16(a)
        d[0] = int32(a)
        e[0] = int64(a)
        f[0] = uint16(a)
        g[0] = uint32(a)
        h[0] = uint64(a)

    b = np.zeros(1, dtype=np.float32)
    c = np.zeros(1, dtype=np.int16)
    d = np.zeros(1, dtype=np.int32)
    e = np.zeros(1, dtype=np.int64)
    f = np.zeros(1, dtype=np.uint16)
    g = np.zeros(1, dtype=np.uint32)
    h = np.zeros(1, dtype=np.uint64)

    simple_kernel[1, 1](b, c, d, e, f, g, h)

    np.testing.assert_allclose(b[0], 3.14, atol=1e-2)
    assert c[0] == 3
    assert d[0] == 3
    assert e[0] == 3
    assert f[0] == 3
    assert g[0] == 3
    assert h[0] == 3


@pytest.mark.parametrize(
    "dtype", [int16, int32, int64, uint16, uint32, uint64, float32]
)
def test_ctor_cast_loop(dtype):
    @cuda.jit(link=get_shims())
    def simple_kernel(a):
        a[0] = dtype(nv_bfloat16(dtype(3.14)))

    a = np.zeros(1, dtype=str(dtype))
    simple_kernel[1, 1](a)

    if np.dtype(str(dtype)).kind == "f":
        np.testing.assert_allclose(a[0], 3.14, atol=1e-2)
    else:
        assert a[0] == 3


def test_arithmetic():
    @cuda.jit(link=get_shims())
    def simple_kernel(arith, logic):
        # Binary Arithmetic Operators
        a = nv_bfloat16(1.0)
        b = nv_bfloat16(2.0)

        arith[0] = float32(a + b)
        arith[1] = float32(a - b)
        arith[2] = float32(a * b)
        arith[3] = float32(a / b)

        # Arithmetic Assignment Operators
        a = nv_bfloat16(1.0)
        b = nv_bfloat16(2.0)

        a += b
        arith[4] = float32(a)
        a -= b
        arith[5] = float32(a)
        a *= b
        arith[6] = float32(a)
        a /= b
        arith[7] = float32(a)

        # Unary Arithmetic Operators
        a = nv_bfloat16(1.0)

        arith[8] = float32(+a)
        arith[9] = float32(-a)

        # Comparison Operators
        a = nv_bfloat16(1.0)
        b = nv_bfloat16(2.0)

        logic[0] = a == b
        logic[1] = a != b
        logic[2] = a > b
        logic[3] = a < b
        logic[4] = a >= b
        logic[5] = a <= b

    arith = np.zeros(10, dtype=np.float32)
    logic = np.zeros(6, dtype=np.bool_)

    simple_kernel[1, 1](arith, logic)

    a = 1.0
    b = 2.0
    np.testing.assert_allclose(
        arith,
        [
            a + b,
            a - b,
            a * b,
            a / b,
            a + b,
            a + b - b,
            (a + b - b) * b,
            (a + b - b) * b / b,
            +a,
            -a,
        ],
        atol=1e-2,
    )
    np.testing.assert_equal(logic, [a == b, a != b, a > b, a < b, a >= b, a <= b])


def test_math_func():
    @cuda.jit(link=get_shims())
    def simple_kernel(a):
        x = nv_bfloat16(3.14)

        a[0] = float32(htrunc(x))
        a[1] = float32(hceil(x))
        a[2] = float32(hfloor(x))
        a[3] = float32(hrint(x))
        a[4] = float32(hsqrt(x))
        a[5] = float32(hrsqrt(x))
        a[6] = float32(hrcp(x))
        a[7] = float32(hlog(x))
        a[8] = float32(hlog2(x))
        a[9] = float32(hlog10(x))
        a[10] = float32(hcos(x))
        a[11] = float32(hsin(x))
        a[12] = float32(hexp(x))
        a[13] = float32(hexp2(x))
        a[14] = float32(hexp10(x))

    a = np.zeros(15, dtype=np.float32)
    simple_kernel[1, 1](a)

    x = 3.14
    np.testing.assert_allclose(
        a[:12],
        [
            np.trunc(x),
            np.ceil(x),
            np.floor(x),
            np.rint(x),
            np.sqrt(x),
            1 / np.sqrt(x),
            1 / x,
            np.log(x),
            np.log2(x),
            np.log10(x),
            np.cos(x),
            np.sin(x),
        ],
        atol=1e-2,
    )

    np.testing.assert_allclose(
        a[12:], [np.exp(x), np.exp2(x), np.power(10, x)], atol=1e2
    )


# implement proxy object for bf16
# proxy should implement CAI which numba will consume directly
# .__cuda_array_interface__


class ProxyTorch(torch.Tensor):
    def __init__(self, tensor):
        self._tensor = tensor

    def __getattr__(self, attr):
        if attr == "__cuda_array_interface__":
            return self.__cuda_array_interface__

        return super(ProxyTorch, self).__getattr__(attr)

    @property
    def __cuda_array_interface__(self):
        typestr = "bfloat16"

        if self._tensor.is_contiguous():
            # __cuda_array_interface__ v2 requires the strides to be omitted
            # (either not set or set to None) for C-contiguous arrays.
            strides = None
        else:
            strides = tuple(s * torch.bfloat16.itemsize for s in self._tensor.stride())
        shape = tuple(self.shape)
        data_ptr = self._tensor.data_ptr() if self._tensor.numel() > 0 else 0
        data = (data_ptr, False)  # read-only is false
        return dict(typestr=typestr, shape=shape, strides=strides, data=data, version=2)


def test_torchbf16():
    torch = pytest.importorskip("torch")

    @cuda.jit(link=get_shims())
    def torch_add(a, b, out):
        i, j = cuda.grid(2)
        if i < out.shape[0] and j < out.shape[1]:
            out[i, j] = a[i, j] + b[i, j]

    a = torch.ones([2, 2], device=torch.device("cuda:0"), dtype=torch.bfloat16)
    b = torch.ones([2, 2], device=torch.device("cuda:0"), dtype=torch.bfloat16)
    aa = ProxyTorch(a)
    bb = ProxyTorch(b)
    twos = aa + bb

    out = torch.zeros([2, 2], device=torch.device("cuda:0"), dtype=torch.bfloat16)
    out = ProxyTorch(out)

    threadsperblock = (16, 16)
    blockspergrid = (1, 1)
    torch_add[blockspergrid, threadsperblock](aa, bb, out)
    assert torch.equal(twos, out)
