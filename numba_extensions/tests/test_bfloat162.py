# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from bf16 import nv_bfloat162, nv_bfloat16
from bf16 import (
    make_bfloat162,
    h2trunc,
    h2ceil,
    h2floor,
    h2rint,
    h2sin,
    h2cos,
    h2exp,
    h2exp2,
    h2exp10,
    h2log2,
    h2log,
    h2log10,
    h2rcp,
    h2rsqrt,
    h2sqrt,
)

import numpy as np
from numba import float32
import numba.cuda as cuda


def test_ctor():
    @cuda.jit(link=["bf16_shim.cu"])
    def simple_kernel():
        x = nv_bfloat16(1.0)
        a = nv_bfloat162(x, x)
        b = make_bfloat162(x, x)

    simple_kernel[1, 1]()


def test_arithmetic():
    @cuda.jit(link=["bf16_shim.cu"])
    def simple_kernel(arith, logic):
        one = nv_bfloat16(1.0)
        ten = nv_bfloat16(10.0)
        a = nv_bfloat162(one, one)
        b = nv_bfloat162(ten, ten)

        # Binary Arithmetic Operators
        c = a + b
        d = a - b
        e = a * b
        f = a / b
        # Arithmetic Assignment Operators
        a += b
        arith[6, 0], arith[6, 1] = float32(a.x), float32(a.y)
        a -= b
        arith[7, 0], arith[7, 1] = float32(a.x), float32(a.y)
        a *= b
        arith[8, 0], arith[8, 1] = float32(a.x), float32(a.y)
        a /= b
        arith[9, 0], arith[9, 1] = float32(a.x), float32(a.y)

        # Unary Arithmetic Operators
        g = +a
        h = -a

        # Comparison Operators
        i = a == b
        j = a != b
        k = a > b
        L = a < b
        m = a >= b
        n = a <= b

        # readouts
        arith[0, 0], arith[0, 1] = float32(c.x), float32(c.y)
        arith[1, 0], arith[1, 1] = float32(d.x), float32(d.y)
        arith[2, 0], arith[2, 1] = float32(e.x), float32(e.y)
        arith[3, 0], arith[3, 1] = float32(f.x), float32(f.y)
        arith[4, 0], arith[4, 1] = float32(g.x), float32(g.y)
        arith[5, 0], arith[5, 1] = float32(h.x), float32(h.y)

        logic[0] = i
        logic[1] = j
        logic[2] = k
        logic[3] = L
        logic[4] = m
        logic[5] = n

    arith = np.zeros((10, 2), dtype=np.float32)
    logic = np.zeros(6, dtype=np.bool_)
    simple_kernel[1, 1](arith, logic)

    arith_gold = [
        [11.0, 11.0],
        [-9.0, -9.0],
        [10.0, 10.0],
        [0.1, 0.1],
        [1.0, 1.0],
        [-1.0, -1.0],
        [11.0, 11.0],
        [1.0, 1.0],
        [10.0, 10.0],
        [1.0, 1.0],
    ]
    logic_gold = [False, True, False, True, False, True]

    np.testing.assert_allclose(arith, arith_gold, atol=1e-2)
    np.testing.assert_equal(logic, logic_gold)


def test_math_functions():
    @cuda.jit(link=["bf16_shim.cu"])
    def simple_kernel(arr):
        x = nv_bfloat16(3.14)
        x2 = nv_bfloat162(x, x)

        a = h2trunc(x2)
        b = h2ceil(x2)
        c = h2floor(x2)
        d = h2rint(x2)
        e = h2sin(x2)
        f = h2cos(x2)
        g = h2exp(x2)
        h = h2exp2(x2)
        i = h2exp10(x2)
        j = h2log2(x2)
        k = h2log(x2)
        L = h2log10(x2)
        m = h2rcp(x2)
        n = h2rsqrt(x2)
        o = h2sqrt(x2)

        # Readout
        arr[0, 0], arr[0, 1] = float32(a.x), float32(a.y)
        arr[1, 0], arr[1, 1] = float32(b.x), float32(b.y)
        arr[2, 0], arr[2, 1] = float32(c.x), float32(c.y)
        arr[3, 0], arr[3, 1] = float32(d.x), float32(d.y)
        arr[4, 0], arr[4, 1] = float32(e.x), float32(e.y)
        arr[5, 0], arr[5, 1] = float32(f.x), float32(f.y)

        arr[6, 0], arr[6, 1] = float32(j.x), float32(j.y)
        arr[7, 0], arr[7, 1] = float32(k.x), float32(k.y)
        arr[8, 0], arr[8, 1] = float32(L.x), float32(L.y)
        arr[9, 0], arr[9, 1] = float32(m.x), float32(m.y)
        arr[10, 0], arr[10, 1] = float32(n.x), float32(n.y)
        arr[11, 0], arr[11, 1] = float32(o.x), float32(o.y)

        arr[12, 0], arr[12, 1] = float32(g.x), float32(g.y)
        arr[13, 0], arr[13, 1] = float32(h.x), float32(h.y)
        arr[14, 0], arr[14, 1] = float32(i.x), float32(i.y)

    arr = np.zeros((15, 2), dtype=np.float32)
    simple_kernel[1, 1](arr)

    x = 3.14
    a_gold = [
        [np.trunc(x), np.trunc(x)],
        [np.ceil(x), np.ceil(x)],
        [np.floor(x), np.floor(x)],
        [np.rint(x), np.rint(x)],
        [np.sin(x), np.sin(x)],
        [np.cos(x), np.cos(x)],
        [np.log2(x), np.log2(x)],
        [np.log(x), np.log(x)],
        [np.log10(x), np.log10(x)],
        [1 / x, 1 / x],
        [1 / np.sqrt(x), 1 / np.sqrt(x)],
        [np.sqrt(x), np.sqrt(x)],
        [np.exp(x), np.exp(x)],
        [np.exp2(x), np.exp2(x)],
        [np.power(10, x), np.power(10, x)],
    ]

    np.testing.assert_allclose(arr[:12], a_gold[:12], atol=1e-2)
    np.testing.assert_allclose(arr[12:], a_gold[12:], atol=1e2)
