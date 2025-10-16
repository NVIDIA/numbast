# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import pytest


@pytest.mark.parametrize("bypass_parse_error", [True, False])
def test_bypass_parse_error(
    run_in_isolated_folder, arch_str, bypass_parse_error
):
    """Test when bypass_parse_error is set, error will be ignored."""

    if not bypass_parse_error:
        with pytest.raises(Exception):
            run_in_isolated_folder(
                "bypass_parse_errors.yml.j2",
                "error_code.cuh",
                {"arch_str": arch_str},
                ruff_format=False,
                bypass_parse_error=bypass_parse_error,
            )
    else:
        run_in_isolated_folder(
            "bypass_parse_errors.yml.j2",
            "error_code.cuh",
            {"arch_str": arch_str},
            ruff_format=False,
            bypass_parse_error=bypass_parse_error,
        )
