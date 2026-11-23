# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which emission backend a configuration selects.

numbast picked its backend with a boolean (``MLIR Backend: true``). A boolean
carries exactly one bit, so it can name two backends and no more. A third has
now arrived -- the CUDA-Oxide renderer emits Rust -- so the question "which
backend?" no longer has a well-formed answer in a bit. This module replaces the
bit with a name.

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

Naming a consumer, not a technology
-----------------------------------
The names identify the thing that consumes the output. ``mlir`` would name a
technology instead, and an MLIR *dialect* emitter is already in flight, so two
backends would answer to it; the same argument that says a bit cannot name
three backends says a name must identify exactly one.
"""

__all__ = [
    "BACKEND_CONFIG_KEY",
    "CUDA_OXIDE",
    "NUMBA_CUDA_MLIR",
    "NUMBA_CUDA",
    "BackendNotFoundError",
    "BackendNotRunnableError",
    "MultipleBackendsError",
    "available_backends",
    "generator_module",
    "resolve_backend_name",
    "runnable_backends",
]

BACKEND_CONFIG_KEY = "Backend"

NUMBA_CUDA = "numba-cuda"
NUMBA_CUDA_MLIR = "numba-cuda-mlir"
CUDA_OXIDE = "cuda-oxide"

#: Each registered backend, mapped to the module whose
#: ``static_binding_generator`` command drives it, or ``None`` for a backend
#: whose renderer exists but which cannot yet be run end to end.
_BACKENDS: dict[str, str | None] = {
    NUMBA_CUDA: "numbast.tools.static_binding_generator",
    NUMBA_CUDA_MLIR: "numbast.experimental.mlir.tools.static_binding_generator",
    CUDA_OXIDE: "numbast.tools.cuda_oxide_binding_generator",
}

DEFAULT_BACKEND = NUMBA_CUDA


class BackendNotFoundError(Exception):
    """A configuration named a backend that does not exist."""


class BackendNotRunnableError(Exception):
    """A configuration named a registered backend that has no driver yet."""


class MultipleBackendsError(Exception):
    """A configuration named more than one backend."""


def available_backends() -> list[str]:
    """Every registered backend name, for help text and error messages."""
    return sorted(_BACKENDS)


def runnable_backends() -> list[str]:
    """The registered names that can actually generate bindings today."""
    return sorted(name for name, module in _BACKENDS.items() if module)


def _as_name_list(value) -> list[str]:
    """Read the ``Backend`` key, which may be one name or a list of them.

    A list is accepted so that the spelling does not have to change when one
    parse can feed several renderers, and because a scalar is far less verbose
    for the one-backend case that is all we can run today. This mirrors how
    ``API Prefix Removal`` already takes either.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        if not all(isinstance(name, str) for name in value):
            raise BackendNotFoundError(
                f'Configuration option "{BACKEND_CONFIG_KEY}" must contain '
                "backend names as strings."
            )
        return list(value)
    raise BackendNotFoundError(
        f'Configuration option "{BACKEND_CONFIG_KEY}" must be a backend name '
        "or a list of backend names."
    )


def resolve_backend_name(config_dict: dict | None) -> str:
    """Name the backend that ``config_dict`` selects.

    Raises
    ------
    BackendNotFoundError
        If a name is given that is not registered. This is deliberately an
        error rather than a fallback to the default: a typo in ``Backend``
        would otherwise generate a complete set of bindings for the wrong
        backend, which looks like success.
    MultipleBackendsError
        If more than one name is given. Taking the first would be the same
        "looks like success" failure, so say so instead.
    """
    config_dict = config_dict or {}

    names = _as_name_list(config_dict.get(BACKEND_CONFIG_KEY))
    if not names:
        return DEFAULT_BACKEND
    if len(names) > 1:
        raise MultipleBackendsError(
            "Generating for more than one backend in a single run is not "
            f"supported yet; {BACKEND_CONFIG_KEY} named "
            f"{', '.join(repr(name) for name in names)}."
        )

    name = names[0]
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
