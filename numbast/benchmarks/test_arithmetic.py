import numba.cuda as cuda
import numpy as np
from numba import float32

from numbast_extensions.bf16 import nv_bfloat16, get_shims

import pytest


@pytest.mark.skip(reason="benchmark is not run by default")
def test_arithmetic(benchmark):
    def bench():
        @cuda.jit(link=get_shims())
        def kernel(arith):
            # Binary Arithmetic Operators
            a = nv_bfloat16(1.0)
            b = nv_bfloat16(2.0)

            arith[0] = float32(a + b)
            arith[1] = float32(a - b)
            arith[2] = float32(a * b)
            arith[3] = float32(a / b)

        arith = np.zeros(4, dtype=np.float32)

        kernel[1, 1](arith)

    benchmark(bench)
