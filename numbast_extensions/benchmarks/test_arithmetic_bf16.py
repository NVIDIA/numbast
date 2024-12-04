import click
import os
import warnings

import numba.cuda as cuda
import numpy as np
from numba import float32

from numbast_extensions.bf16 import (
    nv_bfloat16,
    get_shims,
)


repetition_char = os.getenv("NUMBAST_BENCH_KERN_REPETITION", None)
if repetition_char is None:
    warnings.warn(
        "Unable to retrieve NUMBAST_BENCH_KERN_REPETITION environment variable in `py`."
        "Assume repetition 1000."
    )
    repetition = 1000
else:
    repetition = int(repetition_char)


@click.command()
@click.option("--lto", type=click.BOOL, required=True)
def _run(lto):
    @cuda.jit(link=get_shims(), lto=lto)
    def kernel(arith):
        # Binary Arithmetic Operators
        a = nv_bfloat16(1.0)
        b = nv_bfloat16(2.0)

        arith[0] = float32(a + b)
        arith[1] = float32(a - b)
        arith[2] = float32(a * b)
        arith[3] = float32(a / b)

    arith = np.zeros(4, dtype=np.float32)

    for _ in range(repetition):
        kernel[1, 1](arith)

    assert all(arith == [3.0, -1.0, 2.0, 0.5])


if __name__ == "__main__":
    _run()
