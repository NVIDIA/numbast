# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cuda.bindings.driver import (
    CUresult,
    cuGetErrorName,
    cuModuleGetGlobal,
)
from numba_cuda_mlir import cuda
from numba_cuda_mlir.types import int32


def _check_cuda_result(result):
    err = result[0]
    if err != CUresult.CUDA_SUCCESS:
        _, name = cuGetErrorName(err)
        raise RuntimeError(f"CUDA driver call failed: {name}")
    return result[1:]


def test_module_link_variables_used_is_passed_to_mlir_linker(
    run_in_isolated_folder, arch_str
):
    res = run_in_isolated_folder(
        "module_link_variables_used.yml.j2",
        "module_link_variables_used.cuh",
        {"arch_str": arch_str},
        load_symbols=True,
    )
    symbols = res["symbols"]

    assert symbols["module_link_variables_used"] == [
        "retained_global",
        "another_retained_global",
    ]

    test_function = symbols["test_function"]

    @cuda.jit
    def kernel():
        test_function(int32(1), int32(2))

    kernel[1, 1]()

    overload = kernel.overloads[()]
    linker = overload.metadata["linker"]
    variables_used = None
    for attr in ("variable_used", "variables_used", "_variables_used"):
        if hasattr(linker, attr):
            variables_used = getattr(linker, attr)
            break

    assert variables_used == [
        "retained_global",
        "another_retained_global",
    ]

    overload._codelibrary.get_cufunc()
    retained_globals = {}
    for name in symbols["module_link_variables_used"]:
        _dptr, size = _check_cuda_result(
            cuModuleGetGlobal(overload._codelibrary._module, name.encode())
        )
        retained_globals[name] = size

    assert retained_globals == {
        "retained_global": 4,
        "another_retained_global": 4,
    }
