# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


def _register_alias_numba_type_mappings(type_name: str, alias: str):
    global CTYPE_TO_NBTYPE_STR

    # if type_name not in CTYPE_TO_NBTYPE_STR:
    #     raise ValueError(f"Type '{type_name}' is not registered.")

    CTYPE_TO_NBTYPE_STR[alias] = type_name


def render_aliases(aliases: dict[str, list[str]]):
    rendered = ""
    aliases_template = "{alias} = {underlying_name}\n"

    for underlying_name, alias_list in aliases.items():
        for alias in alias_list:
            type_name = f"_type_{underlying_name}"
            _register_alias_numba_type_mappings(type_name, alias)
            rendered += aliases_template.format(
                alias=alias, underlying_name=underlying_name
            )

    return rendered
