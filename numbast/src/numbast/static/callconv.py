import inspect

from numbast import callconv
from numbast import args
from numbast import intent_defs as intent_mod


def _extract_section(src: str, begin: str, end: str) -> str:
    lines = src.splitlines()
    try:
        start = lines.index(begin) + 1
    except ValueError as exc:
        raise ValueError(f"Missing section marker: {begin}") from exc
    try:
        stop = lines.index(end, start)
    except ValueError as exc:
        raise ValueError(f"Missing section marker: {end}") from exc
    return "\n".join(lines[start:stop]).strip() + "\n"


_INTENT_DEFS_SRC = inspect.getsource(intent_mod)
INTENT_SRC = _extract_section(
    _INTENT_DEFS_SRC, "# NBST:BEGIN_INTENT_DEFS", "# NBST:END_INTENT_DEFS"
)

ARGS_SRC = inspect.getsource(args)
_CALLCONV_SRC = inspect.getsource(callconv)
_CALLCONV_SRC_SECTION = _extract_section(
    _CALLCONV_SRC, "# NBST:BEGIN_CALLCONV", "# NBST:END_CALLCONV"
)

CALLCONV_SRC = INTENT_SRC + "\n" + ARGS_SRC + "\n" + _CALLCONV_SRC_SECTION
