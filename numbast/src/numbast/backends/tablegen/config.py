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
    "Resolver",
    "ShimOptions",
    # Supplied by ``__getattr__`` below rather than defined here, which ruff
    # cannot see.
    "TableGenConfig",  # noqa: F822
    "TableGenOptions",
    "UnresolvedPolicy",
]


class Resolver:
    """How the shim finds the real entry point at runtime.

    ``dlsym``
        ``dlopen`` the library and ``dlsym`` the symbol as declared. Simple,
        and the right choice for a library without its own resolver.

    ``get_proc_address``
        Go through the library's own versioned resolver (CUDA's
        ``cuGetProcAddress``), passing the version the shim was generated
        against, with a raw ``dlsym`` fallback. This is what lets one shim work
        across library versions -- the resolver picks the ABI variant.
    """

    DLSYM = "dlsym"
    GET_PROC_ADDRESS = "get_proc_address"

    ALL = (DLSYM, GET_PROC_ADDRESS)


class UnresolvedPolicy:
    """What the shim does when an entry point cannot be resolved.

    ``status``
        Return the status type's not-found value in band, so a missing entry
        point degrades gracefully -- the program sees an API error rather than
        dying. Requires the library to be status-returning.

    ``abort``
        Print the symbol and ``abort()``. The only option when the entry point
        has no status channel to report through.
    """

    STATUS = "status"
    ABORT = "abort"

    ALL = (STATUS, ABORT)


@dataclass
class ShimOptions:
    """Where the shim gets its symbols and what it does when it cannot."""

    soname: str
    """Library to ``dlopen``, e.g. ``"libcuda.so.1"``."""

    resolver: str = Resolver.DLSYM
    """See :class:`Resolver`."""

    proc_address_symbol: str | None = None
    """Symbol implementing the versioned resolver, e.g. ``cuGetProcAddress_v2``.
    Required when ``resolver`` is ``get_proc_address``."""

    version: int | None = None
    """Version handed to the resolver, e.g. ``12090`` for CUDA 12.9."""

    unresolved_policy: str = UnresolvedPolicy.ABORT
    """See :class:`UnresolvedPolicy`."""

    not_found_value: int = 0
    """Status value returned when ``unresolved_policy`` is ``status``."""

    resolve_base_name: bool = True
    """Resolve the version-less base name (``cuMemAlloc``) rather than the
    declared symbol (``cuMemAlloc_v2``). Correct for ``get_proc_address``,
    which does its own version selection; wrong for plain ``dlsym``, where the
    declared symbol is the one that exists."""

    def __post_init__(self):
        if self.resolver not in Resolver.ALL:
            raise ValueError(
                f"resolver must be one of {Resolver.ALL}, got {self.resolver!r}"
            )
        if self.unresolved_policy not in UnresolvedPolicy.ALL:
            raise ValueError(
                f"unresolved_policy must be one of {UnresolvedPolicy.ALL}, "
                f"got {self.unresolved_policy!r}"
            )
        if (
            self.resolver == Resolver.GET_PROC_ADDRESS
            and not self.proc_address_symbol
        ):
            raise ValueError(
                "resolver 'get_proc_address' requires proc_address_symbol"
            )


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

    shim: ShimOptions | None = None
    """Symbol-resolution layer. Required to emit a shim."""

    skip_non_device: bool = False
    """Drop functions not marked ``__device__``.

    Defaults to the *opposite* of the Numba backends, deliberately: they bind
    CUDA C++ device headers, where a host function is a mistake, while this
    backend exists to mirror host C APIs, where every entry point is a host
    function. Leaving numbast's default in place silently curates away the
    entire library."""

    runtime_package: str = ""
    """Python package holding the generic lowering and analysis passes the
    generated descriptor imports, e.g. ``autogen.runtime``."""

    stream_effects: dict = field(default_factory=dict)
    """``exposed_name -> (effect, param_name)``, where ``effect`` is
    ``"enqueue"`` or ``"sync"``.

    The one table no header can supply: which stream operand an entry point
    queues async work on, or waits for. Human-authored semantic knowledge, and
    the input that makes scheduling analysis possible at all.

    Keyed by *parameter name* rather than operand index so it cannot drift when
    a parameter's direction is reclassified."""

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


def _shim_options_from_dict(config_dict: dict) -> ShimOptions | None:
    """Read the ``Shim`` block, if the recipe asks for a shim at all."""
    shim = config_dict.get("Shim") or None
    if shim is None:
        return None
    return ShimOptions(
        soname=shim["Soname"],
        resolver=shim.get("Resolver", Resolver.DLSYM),
        proc_address_symbol=shim.get("Proc Address Symbol"),
        version=shim.get("Version"),
        unresolved_policy=shim.get("Unresolved Policy", UnresolvedPolicy.ABORT),
        not_found_value=shim.get("Not Found Value", 0),
        resolve_base_name=bool(shim.get("Resolve Base Name", True)),
    )


def _stream_effects_from_dict(config_dict: dict) -> dict:
    """Read ``Stream Effects``, whose YAML shape is deliberately readable:

    .. code-block:: yaml

        Stream Effects:
          launchKernel: [enqueue, hStream]
          streamSynchronize: [sync, hStream]
    """
    raw = config_dict.get("Stream Effects") or {}
    out = {}
    for exposed_name, entry in raw.items():
        if isinstance(entry, dict):
            effect, param = entry["Effect"], entry["Parameter"]
        else:
            effect, param = entry
        if effect not in ("enqueue", "sync"):
            raise ValueError(
                f"Stream Effects[{exposed_name}]: effect must be 'enqueue' or "
                f"'sync', got {effect!r}"
            )
        out[exposed_name] = (effect, param)
    return out


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
        shim=_shim_options_from_dict(config_dict),
        skip_non_device=bool(config_dict.get("Skip Non Device", False)),
        runtime_package=config_dict.get("Runtime Package", "") or "",
        stream_effects=_stream_effects_from_dict(config_dict),
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
