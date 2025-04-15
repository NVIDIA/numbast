# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import pytest


@pytest.mark.parametrize("require_pynvjitlink", [True, False])
def test_require_pynvjitlink(run_in_isolated_folder, require_pynvjitlink):
    """Tests:
    1. Additional Import field actually adds custom import libs in binding
    2. Shim Include Override overrides the shim include line
    """
    res = run_in_isolated_folder(
        "require_pynvjitlink.yml.j2",
        "data.cuh",
        {"require_pynvjitlink": require_pynvjitlink},
        ruff_format=False,
    )

    binding = res["binding"]

    assert (
        'if not importlib.util.find_spec("pynvjitlink"):' in binding
    ) is require_pynvjitlink, binding
