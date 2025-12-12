# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

enum_underlying_integer_type_str_registry: dict[str, str] = {}


def get_enum_underlying_integer_type_dict_as_str() -> str:
    res = ""
    for k, v in enum_underlying_integer_type_str_registry.items():
        res += f'"{k}":types.{v},'

    return "{" + res + "}"
