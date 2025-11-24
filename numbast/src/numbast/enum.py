# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum

from ast_canopy import pylibastcanopy as pylibastcanopy
from numbast.types import register_enum_type


def bind_cxx_enum(e: pylibastcanopy.Enum):
    if len(e.enumerator_values) == 0:
        return None

    vals = [int(v) for v in e.enumerator_values]
    # Numba takes python enum object as-is. Thus we only need to dynamically create
    # the enum object and return it.
    pyenum = IntEnum(e.name, dict(zip(e.enumerators, vals)))

    example_value = vals[0]

    # Add the enum to the CTYPE_MAPS
    register_enum_type(e.name, pyenum(example_value))

    return pyenum


def bind_cxx_enums(enums: list[pylibastcanopy.Enum]) -> list[IntEnum]:
    """
    Create bindings for a list of C++ enums.

    Parameters
    ----------
    enums : list[pylibastcanopy.Enum]
        A list of enum declarations in CXX.

    Returns
    -------
    pyenums : list[IntEnum]
    """

    pyenums = []

    for e in enums:
        pyenum = bind_cxx_enum(e)
        if pyenum is not None:
            pyenums.append(pyenum)

    return pyenums
