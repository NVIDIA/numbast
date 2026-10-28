# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config for the TableGen backend.

Two layers, deliberately:

``TableGenOptions``
    The backend-specific knobs the renderers consume. Plain dataclass, no YAML
    and no dependency on the core config, so renderers stay testable in
    isolation.

``TableGenConfig``
    The YAML-facing class. Inherits
    :class:`numbast.tools.config.shared.SharedConfig` so every backend-neutral
    key (``Entry Point``, ``GPU Arch``, ``File List``, ``Exclude``,
    ``API Prefix Removal``, ``Clang Include Paths``, …) keeps working
    unchanged, and adds this backend's keys on top. Same arrangement
    ``CudaOxideConfig`` uses: core validates core keys, the backend validates
    its own.

    Deliberately *not* the ``Config`` alias in
    ``numbast.tools.static_binding_generator``, which resolves to
    ``NumbaConfig`` -- that carries Numba type maps, datamodels, shim include
    overrides and the Numba dispatch validation, none of which a dialect
    emitter can act on.
"""

from dataclasses import dataclass, field

from numbast.tools.config.shared import SharedConfig

__all__ = [
    "OutParamPolicy",
    "TableGenConfig",
    "TableGenOptions",
]


class OutParamPolicy:
    """How to decide which parameters are lifted to results.

    One policy today, and the option exists so a second can be added without
    changing a call signature: ``classify_params`` already takes the policy and
    is the single place either the op renderer or the eventual lowering asks
    the question.

    ``explicit``
        Lift exactly the parameters carrying an ``out_return`` entry in
        ``Function Argument Intents``, which is numbast's existing mechanism
        for the same question. Nothing is inferred.

    A heuristic policy -- a non-const pointer to a concrete type is an
    out-parameter -- is what a few hundred entry points will eventually want,
    since hand-declaring intents at that scale is not practical. It is
    deliberately not here: inferring the direction of a parameter decides
    whether the lowering allocas a slot for it, so it needs its own change with
    its own expected output to argue against.
    """

    EXPLICIT = "explicit"

    ALL = (EXPLICIT,)


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

    out_param_policy: str = OutParamPolicy.EXPLICIT
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
            "Out Param Policy", OutParamPolicy.EXPLICIT
        ),
        shim_prefix=config_dict.get("Shim Prefix", "_"),
        mnemonic_overrides=config_dict.get("Mnemonic Overrides", {}) or {},
        opaque_structs=config_dict.get("Opaque Structs", []) or [],
        skip_non_device=bool(config_dict.get("Skip Non Device", False)),
    )


class TableGenConfig(SharedConfig):
    """Backend-neutral numbast config plus the TableGen backend's keys."""

    def __init__(self, config_dict: dict):
        super().__init__(config_dict)
        self.tablegen = _tablegen_options_from_dict(config_dict)
