# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The YAML-facing config: one flat mapping, split in two on the way in.

``TableGenConfig`` is what a recipe is parsed into. The renderers consume only
the ``TableGenOptions`` half, so these tests are the only place the key names
themselves are exercised.
"""

import os

import pytest

from numbast.backends.tablegen import TableGenConfig, TableGenOptions

HERE = os.path.dirname(os.path.abspath(__file__))
HEADER = os.path.join(HERE, "driverlike.h")


def _cfg(**overrides):
    """A minimal valid recipe, with ``overrides`` merged on top."""
    return {
        "Entry Point": HEADER,
        "File List": [HEADER],
        "GPU Arch": ["sm_80"],
        "Dialect Name": "acme",
        "Cpp Class Prefix": "Acme",
        **overrides,
    }


def test_one_mapping_populates_both_halves():
    """Core keys land on the config, backend keys land on ``.tablegen``."""
    cfg = TableGenConfig(
        _cfg(
            **{
                "API Prefix Removal": {"Function": ["acme"]},
                "Include": {"Function": ["acmeStreamCreate"]},
                "Handle Prefix": "Acme",
                "Status": {"Type": "AcmeResult"},
                "Out Param Policy": "c_out_params",
            }
        )
    )

    assert cfg.entry_point == HEADER
    assert cfg.api_prefix_removal == {"Function": ["acme"]}
    assert cfg.include_functions == ["acmeStreamCreate"]

    assert isinstance(cfg.tablegen, TableGenOptions)
    assert cfg.tablegen.dialect_name == "acme"
    assert cfg.tablegen.handle_prefix == "Acme"
    assert cfg.tablegen.status_type == "AcmeResult"
    assert cfg.tablegen.out_param_policy == "c_out_params"


def test_the_backend_half_is_absent_until_asked_for():
    """Omitted backend keys fall back to defaults, not to Numba's."""
    options = TableGenConfig(_cfg()).tablegen
    assert options.status_type is None
    assert options.out_param_policy == "none"
    assert options.skip_non_device is False
    assert options.struct_prefixes == []


def test_it_does_not_inherit_the_numba_backend_s_config():
    """Regression guard for the base class.

    ``Config`` in ``numbast.tools.static_binding_generator`` is an alias for
    ``NumbaConfig``; deriving from it would hand this backend Numba type maps
    and datamodels it cannot act on, and run Numba's dispatch validation.
    """
    cfg = TableGenConfig(_cfg())
    assert not hasattr(cfg, "datamodels")
    assert not hasattr(cfg, "types")
    assert not hasattr(cfg, "mlir_backend")


@pytest.mark.parametrize("missing", ["Dialect Name", "Cpp Class Prefix"])
def test_dialect_identity_is_required(missing):
    config_dict = _cfg()
    del config_dict[missing]
    with pytest.raises(KeyError):
        TableGenConfig(config_dict)


def test_an_unknown_out_param_policy_is_refused():
    """Policy is validated at parse time, not at the first surprising op."""
    with pytest.raises(ValueError, match="Out Param Policy"):
        TableGenConfig(_cfg(**{"Out Param Policy": "guess"}))
