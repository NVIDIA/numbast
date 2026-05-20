# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest


@pytest.mark.parametrize("cc, expected", [("sm_80", False), ("sm_90", True)])
def test_cc_generated(run_in_isolated_folder, cc, expected):
    res = run_in_isolated_folder(
        "cfg.yml.j2",
        "data.cuh",
        {"arch_str": cc},
        ruff_format=False,
        load_symbols=True,
    )

    result = res["result"]
    symbols = res["symbols"]

    assert result.exit_code == 0

    assert ("mul" in symbols) == expected
