# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import pytest

import ast_canopy
from numbast import bind_cxx_class_templates  # type: ignore[attr-defined]
from cuda.pathfinder import find_nvidia_header_directory
from numba.cuda.cudadrv.runtime import get_version
from numba import cuda

# Get CUDA runtime version
CUDA_VERSION = get_version()
CUDA_MAJOR_VERSION = CUDA_VERSION[0] if CUDA_VERSION else 0

# CUDA 13+ required for CCCL tests
requires_cuda_13 = pytest.mark.skipif(
    CUDA_MAJOR_VERSION < 13,
    reason=f"CCCL tests require CUDA 13+, found CUDA {CUDA_MAJOR_VERSION}",
)


def make_bindings(path, shim_writer, class_name, arg_intent):
    """Create bindings for a class template from a header file."""
    decls = ast_canopy.parse_declarations_from_source(path, [path], "sm_50")

    bindings = bind_cxx_class_templates(
        decls.class_templates, path, shim_writer, arg_intent=arg_intent
    )

    for ct in bindings:
        if ct.__name__ == class_name:
            return ct

    return None


def _get_cccl_include_path():
    """Get the CCCL include directory using cuda-pathfinder."""
    cccl_path = find_nvidia_header_directory("cccl")
    if cccl_path is None:
        pytest.skip("CCCL include directory not found")
    return cccl_path


@pytest.fixture(scope="module")
def cccl_include_path():
    """Fixture providing the CCCL include directory path."""
    return _get_cccl_include_path()


@pytest.fixture(scope="module")
def block_scan_header(cccl_include_path):
    """Fixture providing the path to block_scan.cuh."""
    path = os.path.join(cccl_include_path, "cub", "block", "block_scan.cuh")
    if not os.path.exists(path):
        pytest.skip(f"block_scan.cuh not found at {path}")
    return path


@pytest.fixture(scope="module")
def block_load_header(cccl_include_path):
    """Fixture providing the path to block_load.cuh."""
    path = os.path.join(cccl_include_path, "cub", "block", "block_load.cuh")
    if not os.path.exists(path):
        pytest.skip(f"block_load.cuh not found at {path}")
    return path


@pytest.fixture(scope="module")
def block_store_header(cccl_include_path):
    """Fixture providing the path to block_store.cuh."""
    path = os.path.join(cccl_include_path, "cub", "block", "block_store.cuh")
    if not os.path.exists(path):
        pytest.skip(f"block_store.cuh not found at {path}")
    return path


@pytest.fixture(scope="module")
def compute_capability():
    """Fixture providing the current device compute capability string."""
    cc = cuda.get_current_device().compute_capability
    return f"sm_{cc[0]}{cc[1]}"
