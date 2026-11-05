# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turn parsed declarations plus user config into backend-neutral plans.

This is the only place that answers "is this symbol in scope, what is it called,
and which per-symbol options apply to it". Backends consume the resulting plans
(see :mod:`numbast.model`) and concern themselves solely with emission.
"""

import re
from warnings import warn

from ast_canopy.pylibastcanopy import execution_space

from numbast.model import FunctionPlan
from numbast.name_policy import apply_prefix_removal

__all__ = [
    "is_device_space",
    "matches_any_regex_pattern",
    "function_skip_reason",
    "plan_functions",
]

_DEVICE_SPACES = frozenset(
    {execution_space.device, execution_space.host_device}
)
#: Derived from the members above rather than written out, so the two spellings
#: cannot drift apart.
_DEVICE_SPACE_NAMES = frozenset(str(space) for space in _DEVICE_SPACES)


def is_device_space(exec_space) -> bool:
    """True for ``__device__`` and ``__host__ __device__`` functions.

    Accepts either an ``execution_space`` member or its string form. Both are
    in use: declarations from a parse carry the enum, while callers that
    assemble declarations without one -- to stay fast and clang-free -- carry
    the stringified space. The rule has to mean the same thing either way, so
    it is this predicate's job to know that and nobody else's.
    """
    return (
        exec_space in _DEVICE_SPACES or str(exec_space) in _DEVICE_SPACE_NAMES
    )


def matches_any_regex_pattern(name: str, patterns: list[str]) -> bool:
    """True if ``name`` matches any of ``patterns``.

    Uses :func:`re.search`, i.e. an unanchored match -- patterns need not match
    from the start of the name. Patterns are assumed already validated
    (``Config._verify_regex_patterns``).
    """
    for pattern in patterns:
        if re.search(pattern, name):
            return True
    return False


def function_skip_reason(
    decl,
    *,
    excludes: list[str],
    skip_prefix: str | None,
    skip_non_device: bool,
) -> str | None:
    """Why ``decl`` is out of scope, or ``None`` if it is in scope."""
    if decl.name in excludes:
        return "excluded"

    if skip_prefix and decl.name.startswith(skip_prefix):
        return "skip_prefix"

    if skip_non_device and not is_device_space(decl.exec_space):
        return "non_device"

    return None


def plan_functions(
    decls,
    *,
    header_path: str,
    excludes: list[str] = [],
    skip_prefix: str | None = None,
    skip_non_device: bool = True,
    cooperative_launch_required: list[str] = [],
    prefix_removal: list[str] = [],
    argument_intents: dict | None = None,
) -> list[FunctionPlan]:
    """Resolve curation for ``decls`` into :class:`~numbast.model.FunctionPlan`.

    ``header_path`` is used only to phrase the non-device warning, which is
    emitted here so it fires exactly once per declaration regardless of backend.
    """
    intents = argument_intents or {}
    plans: list[FunctionPlan] = []

    for decl in decls:
        reason = function_skip_reason(
            decl,
            excludes=excludes,
            skip_prefix=skip_prefix,
            skip_non_device=skip_non_device,
        )
        if reason is not None:
            if reason == "non_device":
                warn(
                    f"Skipping non-device function {decl.name} in {header_path}"
                )
            continue

        plans.append(
            FunctionPlan(
                decl=decl,
                exposed_name=apply_prefix_removal(decl.name, prefix_removal),
                use_cooperative=matches_any_regex_pattern(
                    decl.name, cooperative_launch_required
                ),
                argument_intents=intents,
            )
        )

    return plans
