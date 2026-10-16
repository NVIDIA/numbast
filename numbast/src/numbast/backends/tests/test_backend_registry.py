# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for backend name resolution."""

import pytest

from numbast.backends.registry import (
    CUDA_OXIDE,
    NUMBA,
    MLIR,
    BackendNotFoundError,
    BackendNotRunnableError,
    available_backends,
    generator_module,
    resolve_backend_name,
    runnable_backends,
)


def test_the_default_backend_is_numba():
    assert resolve_backend_name({}) == NUMBA


def test_an_absent_config_resolves_rather_than_raising():
    """An empty YAML file parses to ``None``, not to ``{}``."""
    assert resolve_backend_name(None) == NUMBA


def test_a_named_backend_is_selected():
    assert resolve_backend_name({"Backend": MLIR}) == MLIR


@pytest.mark.parametrize("key", ["MLIR Backend", "mlir_backend"])
def test_the_legacy_boolean_still_selects_the_mlir_backend(key):
    """Configurations in the wild predate ``Backend`` and must keep working."""
    assert resolve_backend_name({key: True}) == MLIR


def test_an_explicit_legacy_false_still_beats_the_snake_case_spelling():
    """Preserves the precedence the nested lookup had before this change.

    The two legacy spellings were read as ``get("MLIR Backend",
    get("mlir_backend", False))``, so a stale snake_case key could not override
    an explicit ``MLIR Backend: false``.
    """
    assert (
        resolve_backend_name({"MLIR Backend": False, "mlir_backend": True})
        == NUMBA
    )


def test_the_named_key_beats_the_legacy_boolean():
    """Adding ``Backend`` to an existing config does what it looks like.

    Otherwise a user migrating off the boolean would have to delete it in the
    same edit, and would silently keep the old backend if they forgot.
    """
    assert (
        resolve_backend_name({"Backend": NUMBA, "MLIR Backend": True}) == NUMBA
    )


def test_an_override_beats_the_config():
    """``--backend`` is the more specific statement of intent."""
    assert resolve_backend_name({"Backend": NUMBA}, MLIR) == MLIR


@pytest.mark.parametrize("override", [None, ""])
def test_an_absent_override_defers_to_the_config(override):
    """A flag nobody passed must not mask the config."""
    assert resolve_backend_name({"Backend": MLIR}, override) == MLIR


def test_an_unknown_name_is_an_error_not_a_fallback():
    """Falling back to the default would look like success.

    A typo in ``Backend`` would otherwise generate a complete, plausible set of
    bindings for a backend the user did not ask for.
    """
    with pytest.raises(BackendNotFoundError):
        resolve_backend_name({"Backend": "mlirr"})


def test_an_unknown_override_is_an_error():
    with pytest.raises(BackendNotFoundError):
        resolve_backend_name({}, "tablegen")


def test_the_error_names_the_backends_that_do_exist():
    """The available set is the whole of the useful information here."""
    with pytest.raises(BackendNotFoundError) as excinfo:
        resolve_backend_name({"Backend": "nope"})

    message = str(excinfo.value)
    assert "nope" in message
    for name in available_backends():
        assert name in message


def test_available_backends_is_sorted_and_complete():
    assert available_backends() == sorted([NUMBA, MLIR, CUDA_OXIDE])


def test_a_third_backend_is_nameable_at_all():
    """The whole argument for a name over a bit, exercised directly.

    A boolean cannot express this value, so a test that only ever asked for
    two backends would pass against the mechanism this replaces.
    """
    assert resolve_backend_name({"Backend": CUDA_OXIDE}) == CUDA_OXIDE


@pytest.mark.parametrize("name", [NUMBA, MLIR])
def test_a_runnable_backend_names_its_driver(name):
    assert generator_module(name).endswith("static_binding_generator")


def test_runnable_backends_excludes_the_driverless_one():
    assert runnable_backends() == sorted([NUMBA, MLIR])
    assert CUDA_OXIDE in available_backends()


def test_a_registered_backend_without_a_driver_says_so():
    """Distinct from a typo, because the remedy is different.

    ``cuda-oxide`` has a renderer in the tree, so "no such backend" would be
    misleading: the backend exists and simply cannot be run end to end yet.
    """
    with pytest.raises(BackendNotRunnableError) as excinfo:
        generator_module(CUDA_OXIDE)

    message = str(excinfo.value)
    assert CUDA_OXIDE in message
    for name in runnable_backends():
        assert name in message


def test_an_unknown_name_has_no_driver_either():
    with pytest.raises(BackendNotFoundError):
        generator_module("tablegen")
