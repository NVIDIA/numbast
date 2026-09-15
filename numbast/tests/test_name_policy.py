# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from numbast.name_policy import apply_prefix_removal


def test_apply_prefix_removal_uses_first_matching_prefix():
    name = "library_detail_function"

    assert apply_prefix_removal(name, ["library_", "library_detail_"]) == (
        "detail_function"
    )
    assert apply_prefix_removal(name, ["library_detail_", "library_"]) == (
        "function"
    )


def test_apply_prefix_removal_preserves_nonmatching_name():
    assert apply_prefix_removal("function", ["library_"]) == "function"


def test_apply_prefix_removal_accepts_empty_prefix_list():
    assert apply_prefix_removal("function", []) == "function"
