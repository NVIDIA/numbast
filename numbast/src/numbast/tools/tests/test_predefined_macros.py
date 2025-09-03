# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from numba import cuda


def test_predefined_macros(run_in_isolated_folder, arch_str):
    """Consider both the config and output folder are traversible in the same
    tree, this PR makes sure that reproducible info accurately reflect where
    the config file can be located as a relative path to the binding file.
    """

    res = run_in_isolated_folder(
        "predefined_macros.yml.j2",
        "predefined_macros.cuh",
        {"arch_str": arch_str},
        ruff_format=True,
        load_symbols=True,
    )

    result = res["result"]
    assert result.exit_code == 0

    symbols = res["symbols"]
    identity = symbols["identity"]

    @cuda.jit
    def kernel():
        _ = identity(3.14)

    kernel[1, 1]()
