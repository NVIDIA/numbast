# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


def test_skip_prefix_set(run_in_isolated_folder, arch_str):
    """Tests:
    1. Output binding can be skipped via `Skip Prefix` entry.
    """
    output_name = "test1.py"
    res = run_in_isolated_folder(
        "skip_prefix.yml.j2",
        "data.cuh",
        {"arch_str": arch_str, "skip_prefix": "m"},
        output_name=output_name,
        ruff_format=False,
        load_symbols=True,
    )

    assert res["result"].exit_code == 0
    assert "add" in res["symbols"]
    assert "mul" not in res["symbols"]


def test_skip_prefix_unset(run_in_isolated_folder, arch_str):
    """Tests:
    1. Output binding is unaffected if `Skip Prefix` is not set
    """
    output_name = "test2.py"
    res = run_in_isolated_folder(
        "skip_prefix.yml.j2",
        "data.cuh",
        {"arch_str": arch_str},
        output_name=output_name,
        ruff_format=False,
        load_symbols=True,
    )

    assert res["result"].exit_code == 0
    assert "add" in res["symbols"]
    assert "mul" in res["symbols"]
