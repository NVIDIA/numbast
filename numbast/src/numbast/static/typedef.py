# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from numbast.static.types import CTYPE_TO_NBTYPE_STR  # noqa


def _register_alias_numba_type_mappings(type_name: str, alias: str):
    global CTYPE_TO_NBTYPE_STR

    # NOTE: Should we check? If we follow the type name generation convention,
    # we know the name for all aliases before the types are even generated.
    # Therefore it's not required for the type names to pre-exist in the type
    # cache.

    # if type_name not in CTYPE_TO_NBTYPE_STR:
    #     raise ValueError(f"Type '{type_name}' is not registered.")

    CTYPE_TO_NBTYPE_STR[alias] = type_name


def render_aliases(aliases: dict[str, list[str]]):
    """Generate python strings that creates alias for the given `aliases` dictionary.

    Parameter
    ---------
    aliases: dict[str, list[str]]
        Keys of `aliases` are the underlying name of all aliases. The value contains a
        list of aliases corresponding to that name.

    Return
    ------
    rendered: str
        Generated python string that create those aliases.
    """
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
