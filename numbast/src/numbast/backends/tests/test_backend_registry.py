# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for backend name resolution."""

import pytest

from numbast.backends.registry import (
    CUDA_OXIDE,
    NUMBA_CUDA,
    NUMBA_CUDA_MLIR,
    BackendNotFoundError,
    BackendNotRunnableError,
    MultipleBackendsError,
    available_backends,
    generator_module,
    resolve_backend_name,
    runnable_backends,
)


def test_the_default_backend_is_numba():
    assert resolve_backend_name({}) == NUMBA_CUDA


def test_an_absent_config_resolves_rather_than_raising():
    """An empty YAML file parses to ``None``, not to ``{}``."""
    assert resolve_backend_name(None) == NUMBA_CUDA


def test_a_named_backend_is_selected():
    assert resolve_backend_name({"Backend": NUMBA_CUDA_MLIR}) == NUMBA_CUDA_MLIR


def test_a_single_name_may_be_written_as_a_list():
    """The list spelling is accepted now so it need not change later.

    One parse feeding several renderers is the eventual shape, and the key
    should not have to change spelling when that arrives.
    """
    assert (
        resolve_backend_name({"Backend": [NUMBA_CUDA_MLIR]}) == NUMBA_CUDA_MLIR
    )


def test_naming_two_backends_is_an_error_rather_than_a_silent_choice():
    """Taking the first would be the same failure as falling back on a typo.

    A run drives one renderer today, so a config naming two is asking for
    something that cannot happen, and picking one would look like success.
    """
    with pytest.raises(MultipleBackendsError) as excinfo:
        resolve_backend_name({"Backend": [NUMBA_CUDA, CUDA_OXIDE]})

    message = str(excinfo.value)
    assert NUMBA_CUDA in message and CUDA_OXIDE in message


def test_an_empty_list_falls_back_to_the_default():
    """``Backend:`` with nothing under it parses to an empty list."""
    assert resolve_backend_name({"Backend": []}) == NUMBA_CUDA


@pytest.mark.parametrize(
    "value", [True, 3, {"name": NUMBA_CUDA_MLIR}, [NUMBA_CUDA_MLIR, 7]]
)
def test_a_backend_that_is_not_a_name_is_rejected(value):
    """Including ``True``, which is what the boolean spelling now parses to."""
    with pytest.raises(BackendNotFoundError):
        resolve_backend_name({"Backend": value})


def test_an_unknown_name_is_an_error_not_a_fallback():
    """Falling back to the default would look like success.

    A typo in ``Backend`` would otherwise generate a complete, plausible set of
    bindings for a backend the user did not ask for.
    """
    with pytest.raises(BackendNotFoundError):
        resolve_backend_name({"Backend": "mlirr"})


def test_the_error_names_the_backends_that_do_exist():
    """The available set is the whole of the useful information here."""
    with pytest.raises(BackendNotFoundError) as excinfo:
        resolve_backend_name({"Backend": "nope"})

    message = str(excinfo.value)
    assert "nope" in message
    for name in available_backends():
        assert name in message


def test_available_backends_is_sorted_and_complete():
    assert available_backends() == sorted(
        [NUMBA_CUDA, NUMBA_CUDA_MLIR, CUDA_OXIDE]
    )


def test_a_third_backend_is_nameable_at_all():
    """The whole argument for a name over a bit, exercised directly.

    A boolean cannot express this value, so a test that only ever asked for
    two backends would pass against the mechanism this replaces.
    """
    assert resolve_backend_name({"Backend": CUDA_OXIDE}) == CUDA_OXIDE


@pytest.mark.parametrize("name", [NUMBA_CUDA, NUMBA_CUDA_MLIR])
def test_a_runnable_backend_names_its_driver(name):
    assert generator_module(name).endswith("static_binding_generator")


def test_runnable_backends_excludes_the_driverless_one():
    assert runnable_backends() == sorted([NUMBA_CUDA, NUMBA_CUDA_MLIR])
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
