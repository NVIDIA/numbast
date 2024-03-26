from enum import IntEnum


import pylibastcanopy
from numbast.types import register_enum_type


def bind_cxx_enum(e: pylibastcanopy.Enum):
    vals = [int(v) for v in e.enumerator_values]
    # Numba takes python enum object as-is. Thus we only need to dynamically create
    # the enum object and return it.
    pyenum = IntEnum(e.name, dict(zip(e.enumerators, vals)))

    # Add the enum to the CTYPE_MAPS
    register_enum_type(e.name, pyenum)

    return pyenum
