# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config for the TableGen backend.

Two layers, deliberately:

``TableGenOptions``
    The backend-specific knobs the renderers consume. Plain dataclass, no YAML
    and no dependency on the core config, so renderers stay testable in
    isolation.

``TableGenConfig``
    The YAML-facing class. **Inherits the core**
    :class:`numbast.tools.static_binding_generator.Config` so every core key
    (``Entry Point``, ``File List``, ``Exclude``, ``API Prefix Removal``,
    ``Clang Include Paths``, …) keeps working unchanged, and adds this
    backend's keys on top. This is the shape a per-backend schema fragment
    should take: core validates core keys, the backend validates its own.
"""

from dataclasses import dataclass, field

__all__ = [
    "OutParamPolicy",
    # Supplied by ``__getattr__`` below rather than defined here, which ruff
    # cannot see.
    "TableGenConfig",  # noqa: F822
    "TableGenOptions",
]


class OutParamPolicy:
    """How to decide which parameters are lifted to results.

    ``none``
        Only parameters with an explicit ``out_return`` entry in
        ``Function Argument Intents`` are lifted. numbast's existing behaviour;
        requires a per-function declaration.

    ``c_out_params``
        Apply the C out-pointer heuristic: a non-const pointer whose pointee
        maps to a concrete type is an out-parameter. Necessary at scale --
        hand-declaring intents for a few hundred entry points is not practical
        -- and it is what the cuda-python dialect recipe uses.
    """

    NONE = "none"
    C_OUT_PARAMS = "c_out_params"

    ALL = (NONE, C_OUT_PARAMS)


@dataclass
class TableGenOptions:
    """Dialect identity and emission policy."""

    dialect_name: str
    """Dialect mnemonic namespace, e.g. ``"acme"`` -> ``acme.add``."""

    cpp_class_prefix: str
    """TableGen record prefix, e.g. ``"Acme"`` -> ``Acme_AddOp``."""

    handle_prefix: str = ""
    """Prefix stripped when naming minted handle types (``CUstream`` -> ``Stream``)."""

    struct_prefixes: list = field(default_factory=list)
    """Prefixes stripped when naming struct/union types, tried in order so the
    longest match wins. ``["CUDA_", "CU"]`` turns ``CUDA_RESOURCE_DESC`` into
    ``ResourceDesc`` -- with only ``"CU"`` it would become ``DA_RESOURCE_DESC``.
    Separate from ``handle_prefix`` because the two vocabularies differ."""

    strip_version_suffixes: bool = False
    """Normalise versioned entry points onto their base name: ``cuMemAlloc_v2``
    -> ``cuMemAlloc``, and drop the ``_ptds``/``_ptsz`` stream variants. The
    first declaration of a base name wins, matching header order. Without this
    a dialect carries ``MemAllocV2Op`` and several near-duplicate ops."""

    status_type: str | None = None
    """Public C name of the status enum (e.g. ``"CUresult"``). When an entry
    point returns it, the op carries a trailing result of the dialect's result
    type."""

    out_param_policy: str = OutParamPolicy.NONE
    """See :class:`OutParamPolicy`."""

    shim_prefix: str = "_"
    """Prefix on the shim symbol an op lowers to (``cuStreamCreate`` ->
    ``_cuStreamCreate``)."""

    mnemonic_overrides: dict = field(default_factory=dict)
    """Exposed name -> explicit op mnemonic, for names the splitter gets wrong."""

    opaque_structs: list = field(default_factory=list)
    """Record names to keep as the opaque pointer type instead of modelling."""

    skip_non_device: bool = False
    """Drop functions not marked ``__device__``.

    Defaults to the *opposite* of the Numba backends, deliberately: they bind
    CUDA C++ device headers, where a host function is a mistake, while this
    backend exists to mirror host C APIs, where every entry point is a host
    function. Leaving numbast's default in place silently curates away the
    entire library."""

    def __post_init__(self):
        if self.out_param_policy not in OutParamPolicy.ALL:
            raise ValueError(
                f"Out Param Policy must be one of {OutParamPolicy.ALL}, "
                f"got {self.out_param_policy!r}"
            )
        if not self.cpp_class_prefix:
            raise ValueError("Cpp Class Prefix is required")
        if not self.dialect_name:
            raise ValueError("Dialect Name is required")


def _tablegen_options_from_dict(config_dict: dict) -> TableGenOptions:
    """Read this backend's keys out of a numbast YAML config dict."""
    status = config_dict.get("Status", {}) or {}
    return TableGenOptions(
        dialect_name=config_dict["Dialect Name"],
        cpp_class_prefix=config_dict["Cpp Class Prefix"],
        handle_prefix=config_dict.get("Handle Prefix", "") or "",
        struct_prefixes=config_dict.get("Struct Prefixes", []) or [],
        strip_version_suffixes=bool(
            config_dict.get("Strip Version Suffixes", False)
        ),
        status_type=status.get("Type"),
        out_param_policy=config_dict.get(
            "Out Param Policy", OutParamPolicy.NONE
        ),
        shim_prefix=config_dict.get("Shim Prefix", "_"),
        mnemonic_overrides=config_dict.get("Mnemonic Overrides", {}) or {},
        opaque_structs=config_dict.get("Opaque Structs", []) or [],
        skip_non_device=bool(config_dict.get("Skip Non Device", False)),
    )


def _make_config_class():
    """Build ``TableGenConfig`` lazily.

    The core ``Config`` lives in ``numbast.tools.static_binding_generator``,
    which imports numba and the Numba renderers at module scope. Importing it
    eagerly would make this backend depend on the Numba stack, so subclassing
    is deferred until the YAML-facing class is actually requested.
    """
    from numbast.tools.static_binding_generator import Config as _CoreConfig

    class TableGenConfig(_CoreConfig):
        """Core numbast config plus the TableGen backend's keys."""

        def __init__(self, config_dict: dict):
            super().__init__(config_dict)
            self.tablegen = _tablegen_options_from_dict(config_dict)

    return TableGenConfig


def __getattr__(name):
    if name == "TableGenConfig":
        return _make_config_class()
    raise AttributeError(name)
