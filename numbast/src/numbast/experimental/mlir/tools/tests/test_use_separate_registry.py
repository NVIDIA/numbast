# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import pytest


@pytest.mark.parametrize("use_separate_registry", [True, False])
def test_use_separate_registry(
    run_in_isolated_folder, arch_str, use_separate_registry
):
    """Test when use_separate_registry is set, the geneerated binding will create or reuse registries accordingly."""
    res = run_in_isolated_folder(
        "use_separate_registry.yml.j2",
        "data.cuh",
        {"arch_str": arch_str, "use_separate_registry": use_separate_registry},
        ruff_format=False,
    )

    run_result = res["result"]
    assert run_result.exit_code == 0

    binding = res["binding"]

    if use_separate_registry:
        assert "TypingRegistry" in binding
        assert "TargetRegistry" in binding
    else:
        assert "TypingRegistry" not in binding
        assert "TargetRegistry" not in binding
