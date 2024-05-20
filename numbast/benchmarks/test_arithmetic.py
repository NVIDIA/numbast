import numba.cuda as cuda
import numpy as np
from numba import float32

from fp16 import (
    half,
    get_shims,
)


def test_arithmetic(benchmark):
    def bench():
        @cuda.jit(link=get_shims())
        def kernel(arith):
            # Binary Arithmetic Operators
            a = half(1.0)
            b = half(2.0)

            arith[0] = float32(a + b)
            arith[1] = float32(a - b)
            arith[2] = float32(a * b)
            arith[3] = float32(a / b)

        arith = np.zeros(4, dtype=np.float32)

        kernel[1, 1](arith)

    benchmark(bench)
