from numbast.types import CTYPE_MAPS

CTYPE_TO_NBTYPE_STR = {k: str(v) for k, v in CTYPE_MAPS.items()}


def to_numba_type_str(ty: str):
    return CTYPE_TO_NBTYPE_STR[ty]
