# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared public-name transformation policy."""


def apply_prefix_removal(name: str, prefixes: list[str]) -> str:
    """Remove the first matching prefix, preserving configured order."""

    for prefix in prefixes:
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name
