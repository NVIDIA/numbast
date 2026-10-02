# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for backend name resolution."""

import pytest

from numbast.backends.registry import (
    NUMBA,
    NUMBA_MLIR,
    BackendNotFoundError,
    available_backends,
    resolve_backend_name,
)


def test_the_default_backend_is_numba():
    assert resolve_backend_name({}) == NUMBA


def test_an_absent_config_resolves_rather_than_raising():
    """An empty YAML file parses to ``None``, not to ``{}``."""
    assert resolve_backend_name(None) == NUMBA


def test_a_named_backend_is_selected():
    assert resolve_backend_name({"Backend": NUMBA_MLIR}) == NUMBA_MLIR


@pytest.mark.parametrize("key", ["MLIR Backend", "mlir_backend"])
def test_the_legacy_boolean_still_selects_the_mlir_backend(key):
    """Configurations in the wild predate ``Backend`` and must keep working."""
    assert resolve_backend_name({key: True}) == NUMBA_MLIR


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
    assert resolve_backend_name({"Backend": NUMBA}, NUMBA_MLIR) == NUMBA_MLIR


@pytest.mark.parametrize("override", [None, ""])
def test_an_absent_override_defers_to_the_config(override):
    """A flag nobody passed must not mask the config."""
    assert resolve_backend_name({"Backend": NUMBA_MLIR}, override) == NUMBA_MLIR


def test_an_unknown_name_is_an_error_not_a_fallback():
    """Falling back to the default would look like success.

    A typo in ``Backend`` would otherwise generate a complete, plausible set of
    bindings for a backend the user did not ask for.
    """
    with pytest.raises(BackendNotFoundError):
        resolve_backend_name({"Backend": "numba-mlirr"})


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
    assert available_backends() == sorted([NUMBA, NUMBA_MLIR])
