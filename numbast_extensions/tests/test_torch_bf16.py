# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import numba.cuda as cuda
from numbast_extensions.bf16 import get_shims

torch = pytest.importorskip("torch")


if cuda.get_current_device().compute_capability < (8, 0):
    pytest.skip(
        reason="bfloat16 require compute capability 8.0+",
        allow_module_level=True,
    )


def test_torchbf16():
    @cuda.jit(link=get_shims())
    def torch_add(a, b, out):
        i, j = cuda.grid(2)
        if i < out.shape[0] and j < out.shape[1]:
            out[i, j] = a[i, j] + b[i, j]

    a = torch.ones([2, 2], device=torch.device("cuda:0"), dtype=torch.bfloat16)
    b = torch.ones([2, 2], device=torch.device("cuda:0"), dtype=torch.bfloat16)
    twos = a + b

    out = torch.zeros([2, 2], device=torch.device("cuda:0"), dtype=torch.bfloat16)

    threadsperblock = (16, 16)
    blockspergrid = (1, 1)
    torch_add[blockspergrid, threadsperblock](a, b, out)
    assert torch.equal(twos, out)
