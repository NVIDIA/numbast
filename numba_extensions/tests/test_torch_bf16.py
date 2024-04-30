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


from bf16 import nv_bfloat16, get_shims
import numpy as np


# ml_dtypes needed to patch np.dtype with bfloat16
from ml_dtypes import bfloat16  # noqa: F401
from numba.np import numpy_support

# what is the constructor vs what is the numba type ?
numpy_support.FROM_DTYPE[np.dtype("bfloat16")] = nv_bfloat16.nb_type

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
