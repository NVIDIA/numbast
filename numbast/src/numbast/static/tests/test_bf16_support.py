import pytest

from numba import cuda, float32
from numba.cuda.bf16 import bfloat16


def test_bindings_from_bf16(make_binding):
    res1 = make_binding("bf16.cuh", {}, {})

    binding1 = res1["bindings"]

    add = binding1["add"]

    @cuda.jit
    def kernel(arr):
        x = add(bfloat16(3.14), bfloat16(3.14))
        arr[0] = float32(x)

    arr = cuda.device_array((1,), dtype="float32")
    kernel[1, 1](arr)

    assert pytest.approx(arr[0], 1e-2) == 6.28

    # Check that bfloat16 is imported
    assert "from numba.cuda.types import bfloat16" in res1["src"]


def test_bindings_from_bf16raw(make_binding):
    res = make_binding("bf16.cuh", {}, {})

    binding = res["bindings"]

    bf16_from_raw = binding["bf16_from_raw"]
    bf16_to_raw = binding["bf16_to_raw"]

    @cuda.jit
    def kernel(arr):
        x = bfloat16(3.14)

        x_raw = bf16_to_raw(x)
        x2 = bf16_from_raw(x_raw)

        arr[0] = float32(x2)

    arr = cuda.device_array((1,), dtype="float32")

    kernel[1, 1](arr)

    assert pytest.approx(arr[0], 1e-2) == 3.14
