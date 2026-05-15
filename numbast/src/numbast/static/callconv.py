import inspect
import pprint

from numbast import callconv
from numbast import args
from numbast import intent_defs as intent_mod
from numbast import return_materialization as return_materialization_mod
from numbast.types import NUMBA_TYPE_ALIGNOF_MAPS


def _extract_section(src: str, begin: str, end: str) -> str:
    """
    Extract the text between two exact marker lines in a source string.

    Parameters:
        src (str): The full source text to search, split into lines.
        begin (str): The marker line that starts the section (the returned text begins after this line).
        end (str): The marker line that ends the section (the returned text ends before this line).

    Returns:
        str: The lines between the first occurrence of `begin` and the next occurrence of `end`, joined with newline characters, trimmed of surrounding whitespace, and terminated with a single trailing newline.

    Raises:
        ValueError: If `begin` is not found in `src` or if `end` is not found after `begin`.
    """
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

_RETURN_MATERIALIZATION_DEFS_SRC = inspect.getsource(return_materialization_mod)
RETURN_MATERIALIZATION_SRC = _extract_section(
    _RETURN_MATERIALIZATION_DEFS_SRC,
    "# NBST:BEGIN_RETURN_MATERIALIZATION_DEFS",
    "# NBST:END_RETURN_MATERIALIZATION_DEFS",
)

ARGS_SRC = inspect.getsource(args)
_CALLCONV_SRC = inspect.getsource(callconv)
_CALLCONV_SRC_SECTION = _extract_section(
    _CALLCONV_SRC, "# NBST:BEGIN_CALLCONV", "# NBST:END_CALLCONV"
)


def _render_numba_type_alignof_src() -> str:
    """
    Render the alignof lookup helper used by generated static bindings.

    Static bindings cannot import ``numbast.types`` at runtime, so the
    generator snapshots the known explicit alignment map as type-name strings.
    Generated struct types still use their ``alignof_`` attribute directly.
    """
    alignof_by_type_name = {
        str(numba_type): alignof
        for numba_type, alignof in NUMBA_TYPE_ALIGNOF_MAPS.items()
    }
    alignof_map_src = pprint.pformat(alignof_by_type_name, sort_dicts=True)
    return f"""
_NUMBA_TYPE_ALIGNOF_MAPS = {alignof_map_src}


def get_numba_type_alignof(numba_type):
    return _NUMBA_TYPE_ALIGNOF_MAPS.get(str(numba_type))
"""


CALLCONV_SRC = (
    INTENT_SRC
    + "\n"
    + RETURN_MATERIALIZATION_SRC
    + "\n"
    + ARGS_SRC
    + "\n"
    + _render_numba_type_alignof_src()
    + "\n"
    + _CALLCONV_SRC_SECTION
)
