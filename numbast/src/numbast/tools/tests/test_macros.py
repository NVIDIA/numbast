import os

import numpy as np
from numba import cuda
import pytest

from click.testing import CliRunner

from numbast.tools.static_binding_generator import static_binding_generator


@pytest.fixture
def kernel():
    def _lazy_kernel(globals):
        forty_two_int = globals["forty_two_int"]
        forty_two_float = globals["forty_two_float"]
        forty_two_double = globals["forty_two_double"]
        c_ext_shim_source = globals["c_ext_shim_source"]

        @cuda.jit(link=[c_ext_shim_source])
        def kernel(arr):
            arr[0] = forty_two_int()
            arr[1] = int(forty_two_float())
            arr[2] = int(forty_two_double())

        arr = np.array([0, 0, 0], dtype="int32")
        kernel[1, 1](arr)
        assert (arr == [42] * 3).all()

    return _lazy_kernel


def test_cli_yml_inputs_macro_expansion(tmpdir, kernel):
    name = "macros"
    subdir = tmpdir.mkdir("sub")
    data = os.path.join(os.path.dirname(__file__), f"{name}.cuh")

    cfg = f"""Name: Test Data
Version: 0.0.1
Entry Point: {data}
File List:
    - {data}
Macro-expanded Function Prefixes:
    - forty_two_
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

    assert (
        result.exit_code == 0
    ), f"Exception raised: {result.exception}, Stdout: {result.stdout}"

    output = subdir / f"{name}.py"
    assert os.path.exists(output)

    with open(output, "r") as f:
        bindings = f.read()

    globals = {}
    exec(bindings, globals)

    kernel(globals)
