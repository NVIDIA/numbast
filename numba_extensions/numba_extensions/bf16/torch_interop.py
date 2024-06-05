# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings
import numpy as np
from numba.np import numpy_support  # noqa: E402

import numba.cuda.dispatcher
from numba.cuda.dispatcher import _LaunchConfiguration

from numba_extensions.bf16 import nv_bfloat16

try:
    import torch

    _TORCH_IMPORTED = True
except ImportError:
    _TORCH_IMPORTED = False

try:
    from ml_dtypes import bfloat16  # noqa: F401 E402

    _ML_DTYPES_IMPORTED = True
except ImportError:
    if _TORCH_IMPORTED:
        warnings.warn(
            "Pytorch installation detected. To use Numba-extensions' bfloat16"
            "bindings, please also install ml_dtypes via `python -m pip install ml_dtypes`"
        )

    _ML_DTYPES_IMPORTED = False

_WRAP_BF16_TENSOR = _TORCH_IMPORTED and _ML_DTYPES_IMPORTED


def patch_numba():
    """
    This function patches Numba to support bfloat16 data type from PyTorch.
    It creates a proxy object that implements the CUDA array interface (CAI) that provides
    correct interpretation to bfloat16 tensors. Then it patches the _LaunchConfiguration
    object in Numba to enable Numba to directly consume bfloat16 tensors.

    In order for this patching to work, PyTorch and ml_dtypes must be installed.

    Returns:
        None
    """
    if not _WRAP_BF16_TENSOR:
        return

    # Register the NumPy dtype for bfloat16 with the Numba bfloat16 type
    numpy_support.FROM_DTYPE[np.dtype("bfloat16")] = nv_bfloat16._nbtype

    class ProxyTorch(torch.Tensor):
        """
        Proxy object for PyTorch tensors that implements the CUDA array interface (CAI)
        to enable Numba to directly consume bfloat16 tensors.
        """

        def __init__(self, tensor):
            """
            Initialize the proxy object with the given PyTorch tensor.

            Args:
                tensor (torch.Tensor): The PyTorch tensor to wrap.
            """
            self._tensor = tensor

        def __getattr__(self, attr):
            """
            Delegate attribute access to the wrapped PyTorch tensor.

            Args:
                attr (str): The attribute name.

            Returns:
                The attribute value.
            """
            if attr == "__cuda_array_interface__":
                return self.__cuda_array_interface__

            return super(ProxyTorch, self).__getattr__(attr)

        @property
        def __cuda_array_interface__(self):
            """
            Implement the CUDA array interface for the proxy object.

            Returns:
                dict: A dictionary representing the CUDA array interface.
            """
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
        """
        Custom launch configuration for Numba CUDA kernels that wraps bfloat16 PyTorch
        tensors with the proxy object before launching the kernel.
        """

        def __call__(self, *args):
            """
            Wrap PyTorch tensors with the proxy object before launching the kernel.

            Args:
                *args: The arguments to the kernel.

            Returns:
                The result of the kernel launch.
            """
            torch_filtered = [
                ProxyTorch(arg)
                if (isinstance(arg, torch.Tensor) and arg.dtype == torch.bfloat16)
                else arg
                for arg in args
            ]
            return super(_BF16TorchWrappedLaunchConfiguration, self).__call__(
                *torch_filtered
            )

    numba.cuda.dispatcher._LaunchConfiguration = _BF16TorchWrappedLaunchConfiguration
