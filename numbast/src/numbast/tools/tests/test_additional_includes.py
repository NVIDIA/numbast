# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import cffi

import numpy as np
from numba import cuda
import pytest

from click.testing import CliRunner

from numbast.tools.static_binding_generator import static_binding_generator


@pytest.fixture
def kernel():
    def _lazy_kernel(globals):
        ffi = cffi.FFI()
        set42 = globals["set42"]

        @cuda.jit
        def kernel(arr):
            ptr = ffi.from_buffer(arr)
            set42(ptr)

        arr = np.array([0, 0], dtype="int32")
        kernel[1, 1](arr)
        assert arr[0] == 42

    return _lazy_kernel


@pytest.fixture
def patch_extra_include_paths():
    old_extra_include_paths = cuda.config.CUDA_NVRTC_EXTRA_SEARCH_PATHS

    cuda.config.CUDA_NVRTC_EXTRA_SEARCH_PATHS = (
        f"{os.path.join(os.path.dirname(__file__), 'include')}"
    )
    yield
    cuda.config.CUDA_NVRTC_EXTRA_SEARCH_PATHS = old_extra_include_paths


def test_cli_yml_inputs_additional_includes(
    tmpdir, kernel, patch_extra_include_paths, arch_str
):
    name = "additional_include"
    subdir = tmpdir.mkdir("sub")
    data = os.path.join(os.path.dirname(__file__), f"{name}.cuh")

    cfg = f"""Name: Test Data
Version: 0.0.1
Entry Point: {data}
File List:
    - {data}
GPU Arch:
    - {arch_str}
Exclude: {{}}
Types: {{}}
Data Models: {{}}
Clang Include Paths:
    - {os.path.join(os.path.dirname(__file__), "include")}
"""

    cfg_file = subdir / "cfg.yaml"
    with open(cfg_file, "w") as f:
        f.write(cfg)

    runner = CliRunner()
    result = runner.invoke(
        static_binding_generator,
        [
            "--cfg-path",
            cfg_file,
            "--output-dir",
            subdir,
        ],
    )

    assert result.exit_code == 0, f"CMD ERROR: {result.stdout}"

    output = subdir / f"{name}.py"
    assert os.path.exists(output)

    with open(output) as f:
        bindings = f.read()

    globals = {}
    exec(bindings, globals)

    kernel(globals)
