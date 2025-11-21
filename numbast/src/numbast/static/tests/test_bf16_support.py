import pytest

from numba import cuda, float32
from numba.cuda.bf16 import bfloat16


def test_bindings_from_bf16(make_binding):
    res1 = make_binding("bf16.cuh", {}, {})

    binding1 = res1["bindings"]

    add = binding1["add"]
    add2 = binding1["add2"]

    @cuda.jit
    def kernel(arr):
        x = add(bfloat16(3.14), bfloat16(3.14))
        arr[0] = float32(x)
        arr[1] = float32(add2(bfloat16(3.14), bfloat16(3.14)))

    arr = cuda.device_array((2,), dtype="float32")
    kernel[1, 1](arr)

    assert pytest.approx(arr[0], 1e-2) == 6.28
    assert pytest.approx(arr[1], 1e-2) == 6.28

    # Check that bfloat16 is imported
    assert "from numba.cuda.types import bfloat16" in res1["src"]
