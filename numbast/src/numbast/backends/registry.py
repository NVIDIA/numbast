# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which emission backend a configuration selects.

numbast picks its backend with a boolean today (``MLIR Backend: true``). A
boolean carries exactly one bit, so it can name two backends and no more; the
moment there is a third, the question "which backend?" has no well-formed
answer. This module replaces the bit with a name.

Naming, rather than loading
---------------------------
This module resolves a *name* and nothing else. It deliberately does not import
or return the backend itself: each backend's entry point pulls in its own stack
-- the Numba backends import numba and its renderers at module scope -- and a
user of one backend has no reason to have another's dependencies installed.
Callers therefore import their backend inside the branch that needs it, so
resolving a name stays free.
"""

__all__ = [
    "BACKEND_CONFIG_KEY",
    "NUMBA",
    "NUMBA_MLIR",
    "BackendNotFoundError",
    "available_backends",
    "resolve_backend_name",
]

BACKEND_CONFIG_KEY = "Backend"

NUMBA = "numba"
NUMBA_MLIR = "numba-mlir"

_BACKENDS = frozenset({NUMBA, NUMBA_MLIR})

DEFAULT_BACKEND = NUMBA

#: The keys that selected the MLIR backend before ``Backend`` existed. Read in
#: order, so an explicit ``MLIR Backend: false`` still overrides a stale
#: ``mlir_backend: true`` exactly as it did before.
_LEGACY_MLIR_KEYS = ("MLIR Backend", "mlir_backend")


class BackendNotFoundError(Exception):
    """A configuration named a backend that does not exist."""


def available_backends() -> list[str]:
    """The registered backend names, for help text and error messages."""
    return sorted(_BACKENDS)


def _legacy_mlir_boolean(config_dict: dict) -> bool:
    """Whether the pre-``Backend`` boolean selects the MLIR backend."""
    value = False
    for key in reversed(_LEGACY_MLIR_KEYS):
        if key in config_dict:
            value = config_dict[key]
    return bool(value)


def resolve_backend_name(
    config_dict: dict | None, override: str | None = None
) -> str:
    """Name the backend that ``config_dict`` selects.

    Precedence is ``override`` > ``Backend`` > the legacy boolean > ``numba``.
    An override comes from the command line, which is the more specific
    statement of intent and so wins over the file; the legacy boolean sits below
    ``Backend`` so that adding the new key to an existing config does what it
    looks like it does.

    Raises
    ------
    BackendNotFoundError
        If a name is given that is not registered. This is deliberately an
        error rather than a fallback to the default: a typo in ``Backend``
        would otherwise generate a complete set of bindings for the wrong
        backend, which looks like success.
    """
    config_dict = config_dict or {}

    if override:
        name = override
    elif config_dict.get(BACKEND_CONFIG_KEY):
        name = config_dict[BACKEND_CONFIG_KEY]
    elif _legacy_mlir_boolean(config_dict):
        name = NUMBA_MLIR
    else:
        name = DEFAULT_BACKEND

    if name not in _BACKENDS:
        raise BackendNotFoundError(
            f"Unknown backend {name!r}. Available backends: "
            f"{', '.join(available_backends())}."
        )
    return name
