import inspect

from numbast import callconv
from numbast import args
from numbast import intent_defs as intent_mod


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

ARGS_SRC = inspect.getsource(args)
_CALLCONV_SRC = inspect.getsource(callconv)
_CALLCONV_SRC_SECTION = _extract_section(
    _CALLCONV_SRC, "# NBST:BEGIN_CALLCONV", "# NBST:END_CALLCONV"
)

CALLCONV_SRC = INTENT_SRC + "\n" + ARGS_SRC + "\n" + _CALLCONV_SRC_SECTION
