# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from numba import types as nbtypes

enum_underlying_integer_type_registry: dict[str, nbtypes.Type] = {}
