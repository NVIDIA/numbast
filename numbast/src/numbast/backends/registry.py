# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which emission backend a configuration selects.

numbast picks its backend with a boolean today (``MLIR Backend: true``). A
boolean carries exactly one bit, so it can name two backends and no more. A
third has now arrived -- the CUDA-Oxide renderer emits Rust -- so the question
"which backend?" no longer has a well-formed answer in a bit. This module
replaces the bit with a name.

Naming, rather than loading
---------------------------
This module resolves a *name* and nothing else. It deliberately does not import
or return the backend itself: each backend's entry point pulls in its own stack
-- the Numba backends import numba and its renderers at module scope -- and a
user of one backend has no reason to have another's dependencies installed.
Callers therefore import their backend inside the branch that needs it, so
resolving a name stays free.

For the same reason :func:`generator_module` hands back an import path as a
string rather than a module. The dispatcher in :mod:`numbast.cli` needs to know
*what* to import before it is willing to import anything, which is what keeps
``--help`` and a config error from dragging in a backend's dependencies.
"""

__all__ = [
    "BACKEND_CONFIG_KEY",
    "CUDA_OXIDE",
    "MLIR",
    "NUMBA",
    "BackendNotFoundError",
    "BackendNotRunnableError",
    "available_backends",
    "generator_module",
    "resolve_backend_name",
    "runnable_backends",
]

BACKEND_CONFIG_KEY = "Backend"

NUMBA = "numba"
MLIR = "mlir"
CUDA_OXIDE = "cuda-oxide"

#: Each registered backend, mapped to the module whose
#: ``static_binding_generator`` command drives it, or ``None`` for a backend
#: whose renderer exists but which cannot yet be run end to end.
#:
#: CUDA-Oxide is registered without a driver on purpose. Its binding plan and
#: renderer are both in the tree, so a configuration naming it is asking for
#: something that half exists, and "not wired up yet" is a more useful answer
#: than "no such backend". It also leaves the remaining work as one edit here
#: rather than a design.
_BACKENDS: dict[str, str | None] = {
    NUMBA: "numbast.tools.static_binding_generator",
    MLIR: "numbast.experimental.mlir.tools.static_binding_generator",
    CUDA_OXIDE: None,
}

DEFAULT_BACKEND = NUMBA

#: The keys that selected the MLIR backend before ``Backend`` existed. Read in
#: order, so an explicit ``MLIR Backend: false`` still overrides a stale
#: ``mlir_backend: true`` exactly as it did before.
_LEGACY_MLIR_KEYS = ("MLIR Backend", "mlir_backend")


class BackendNotFoundError(Exception):
    """A configuration named a backend that does not exist."""


class BackendNotRunnableError(Exception):
    """A configuration named a registered backend that has no driver yet."""


def available_backends() -> list[str]:
    """Every registered backend name, for help text and error messages."""
    return sorted(_BACKENDS)


def runnable_backends() -> list[str]:
    """The registered names that can actually generate bindings today."""
    return sorted(name for name, module in _BACKENDS.items() if module)


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
        name = MLIR
    else:
        name = DEFAULT_BACKEND

    if name not in _BACKENDS:
        raise BackendNotFoundError(
            f"Unknown backend {name!r}. Available backends: "
            f"{', '.join(available_backends())}."
        )
    return name


def generator_module(name: str) -> str:
    """The import path of the module that drives backend ``name``.

    Raises
    ------
    BackendNotFoundError
        If ``name`` is not registered.
    BackendNotRunnableError
        If ``name`` is registered but has no driver yet, which is a different
        problem from a typo and worth a different message.
    """
    if name not in _BACKENDS:
        raise BackendNotFoundError(
            f"Unknown backend {name!r}. Available backends: "
            f"{', '.join(available_backends())}."
        )

    module = _BACKENDS[name]
    if module is None:
        raise BackendNotRunnableError(
            f"Backend {name!r} cannot generate bindings yet. Backends that "
            f"can: {', '.join(runnable_backends())}."
        )
    return module
