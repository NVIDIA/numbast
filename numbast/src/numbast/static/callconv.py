import inspect
from numbast import callconv
from numbast import args

ARGS_SRC = inspect.getsource(args)
_CALLCONV_SRC = inspect.getsource(callconv)

# Patch CALLCONV_SRC
_CALLCONV_SRC_PATCHED = _CALLCONV_SRC.replace(
    "from numbast.args import prepare_ir_types", ""
)

CALLCONV_SRC = ARGS_SRC + "\n" + _CALLCONV_SRC_PATCHED
