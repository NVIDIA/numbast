import inspect

from numbast import callconv
from numbast import args
from numbast import intent_defs as intent_mod

INTENT_SRC = (
    "from dataclasses import dataclass\n"
    "from enum import Enum\n\n"
    + inspect.getsource(intent_mod.ArgIntent)
    + "\n\n"
    + inspect.getsource(intent_mod.IntentPlan)
)

ARGS_SRC = inspect.getsource(args)
_CALLCONV_SRC = inspect.getsource(callconv)

# Patch CALLCONV_SRC
_CALLCONV_SRC_PATCHED = _CALLCONV_SRC.replace(
    "from numbast.args import prepare_ir_types", ""
).replace("from numbast.intent import IntentPlan", "")

CALLCONV_SRC = INTENT_SRC + "\n" + ARGS_SRC + "\n" + _CALLCONV_SRC_PATCHED
