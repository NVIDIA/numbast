# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from numbast.static.renderer import clear_base_renderer_cache
from numbast.static.function import clear_function_apis_registry


def reset_renderer():
    """Clear all renderer cache and api registries.

    Sometimes the renderer needs to run multiple times in the same python
    session (pytest). This function resets the renderer so that it runs in
    a clean state.
    """
    clear_base_renderer_cache()
    clear_function_apis_registry()
