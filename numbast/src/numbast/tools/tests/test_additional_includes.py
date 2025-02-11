import os

import cffi

import numpy as np
from numba import cuda
import pytest

from click.testing import CliRunner

from numbast import numba_patch
from numbast.tools.static_binding_generator import static_binding_generator


@pytest.fixture
def kernel():
    def _lazy_kernel(globals):
        ffi = cffi.FFI()
        set42 = globals["set42"]
        c_ext_shim_source = globals["c_ext_shim_source"]

        @cuda.jit(link=[c_ext_shim_source])
        def kernel(arr):
            ptr = ffi.from_buffer(arr)
            set42(ptr)

        arr = np.array([0, 0], dtype="int32")
        kernel[1, 1](arr)
        assert arr[0] == 42

    return _lazy_kernel


@pytest.fixture
def patch_extra_include_paths():
    old_extra_include_paths = numba_patch.extra_include_paths
    numba_patch.extra_include_paths = numba_patch.extra_include_paths + [
        f"-I{os.path.join(os.path.dirname(__file__), 'include')}"
    ]
    yield
    numba_patch.extra_include_paths = old_extra_include_paths


def test_cli_yml_inputs_additional_includes(tmpdir, kernel, patch_extra_include_paths):
    name = "additional_include"
    subdir = tmpdir.mkdir("sub")
    data = os.path.join(os.path.dirname(__file__), f"{name}.cuh")

    cfg = f"""Name: Test Data
Version: 0.0.1
Entry Point: {data}
File List:
    - {data}
Exclude: {{}}
Types:
    Foo: Type
Data Models:
    Foo: StructModel
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
            "--compute-capability",
            "sm_50",
            "--output-dir",
            subdir,
        ],
    )

    assert result.exit_code == 0, f"CMD ERROR: {result.stdout}"

    output = subdir / f"{name}.py"
    assert os.path.exists(output)

    with open(output, "r") as f:
        bindings = f.read()

    globals = {}
    exec(bindings, globals)

    kernel(globals)
