# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generation refusing to emit a dialect that would fail at registration.

The expected output in ``expected/`` covers what this backend emits. It cannot
cover what it declines to emit, which is what these tests are for.
"""

import pytest

from numbast.backends.tablegen import TableGenOptions, TableGenTypesRenderer
from numbast.backends.tablegen.types import TypeMapper
from numbast.backends.tablegen.types_td import DuplicateTypeMnemonicError

OPTIONS = TableGenOptions(dialect_name="acme", cpp_class_prefix="Acme")


def _mapper_with(**structs) -> TypeMapper:
    """A mapper carrying ``{c_name: record_name}`` and nothing else."""
    mapper = TypeMapper("Acme")
    mapper.structs.update(structs)
    return mapper


def test_records_differing_only_in_case_are_refused():
    """Two record names differing only in case claim one mnemonic.

    A struct's mnemonic is its record name lowered, so ``DevResource`` and
    ``Devresource`` -- a real record and the anonymous one nested inside it --
    both ask for ``devresource``. TableGen keys types on the C++ name and
    accepts the pair, the library links, and the process then dies when the
    dialect is registered with ``Dialect Type with name ... is already
    registered``. Generation is the last point at which this can be reported
    against the declarations that caused it.
    """
    mapper = _mapper_with(
        AcmeDevResource_st="DevResource",
        AcmeDevResource_anon="Devresource",
    )

    with pytest.raises(DuplicateTypeMnemonicError) as excinfo:
        TableGenTypesRenderer(mapper, OPTIONS).render_as_str()

    # The message has to name both culprits. The mnemonic alone sends the
    # reader hunting through a hundred-odd TypeDefs for which two collided.
    message = str(excinfo.value)
    assert "'devresource'" in message
    assert "DevResource" in message and "Devresource" in message


def test_distinct_record_names_are_accepted():
    """The control: one case change makes the same pair legal."""
    mapper = _mapper_with(
        AcmeDevResource_st="DevResource",
        AcmeDevResource_anon="DevResourceAnon",
    )

    td = TableGenTypesRenderer(mapper, OPTIONS).render_as_str()
    assert '"devresource"' in td
    assert '"devresourceanon"' in td
