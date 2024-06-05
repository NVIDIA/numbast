# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

try:
    import torch

    # ml types should be installed in the torch container
    # ml_dtypes needed to patch np.dtype with bfloat16
    from ml_dtypes import bfloat16  # noqa: F401 E402

    _WRAP_TENSOR = True
except ImportError:
    _WRAP_TENSOR = False

import numpy as np
from numba.np import numpy_support  # noqa: E402

import numba.cuda.dispatcher
from numba.cuda.dispatcher import _LaunchConfiguration

from numba_extensions.bf16 import nv_bfloat16


def patch_numba():
    if _WRAP_TENSOR:
        # what is the constructor vs what is the numba type ?
        numpy_support.FROM_DTYPE[np.dtype("bfloat16")] = nv_bfloat16._nbtype

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
                    strides = tuple(
                        s * torch.bfloat16.itemsize for s in self._tensor.stride()
                    )
                shape = tuple(self.shape)
                data_ptr = self._tensor.data_ptr() if self._tensor.numel() > 0 else 0
                data = (data_ptr, False)  # read-only is false
                return dict(
                    typestr=typestr, shape=shape, strides=strides, data=data, version=2
                )

        class _BF16TorchWrappedLaunchConfiguration(_LaunchConfiguration):
            def __call__(self, *args):
                torch_filtered = [
                    ProxyTorch(arg)
                    if (isinstance(arg, torch.Tensor) and arg.dtype == torch.bfloat16)
                    else arg
                    for arg in args
                ]
                return super(_BF16TorchWrappedLaunchConfiguration, self).__call__(
                    *torch_filtered
                )

        numba.cuda.dispatcher._LaunchConfiguration = (
            _BF16TorchWrappedLaunchConfiguration
        )

    else:
        return None
