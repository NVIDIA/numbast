# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


def test_skip_prefix_set(run_in_isolated_folder, arch_str):
    """Tests:
    1. Output binding can be skipped via `Skip Prefix` entry.
    """
    res = run_in_isolated_folder(
        "skip_prefix.yml.j2",
        "data2.cuh",
        {"arch_str": arch_str, "skip_prefix": "m"},
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
    res = run_in_isolated_folder(
        "skip_prefix.yml.j2",
        "data2.cuh",
        {"arch_str": arch_str},
        ruff_format=False,
        load_symbols=True,
    )

    assert res["result"].exit_code == 0
    assert "add" in res["symbols"]
    assert "mul" in res["symbols"]
