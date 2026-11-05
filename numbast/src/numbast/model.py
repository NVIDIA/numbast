# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend-neutral plans: curation decisions resolved once, consumed by any backend.

A *plan* pairs a parsed declaration with the decisions curation makes about it --
the exposed name, whether cooperative launch is required, the per-argument intent
spec. Those decisions are a function of the declaration plus the user's config
only; they do not depend on what we emit. So they can be computed once (see
:mod:`numbast.curate`) and handed to a Numba renderer, an MLIR-dialect emitter,
or anything else.

Scope note
----------
Plans carry the ``ast_canopy`` declaration itself rather than a re-modelled type
tree. Normalising *decisions* is cheap and is what removes duplication between
backends today. Normalising C++ *types* is a substantially larger change, and
every current backend already consumes ``ast_canopy`` declarations directly.

Only functions are modelled so far. Structs, enums, and templates still reach
their renderers directly; extending this does not change the shape of the seam.

What deliberately does **not** live here
----------------------------------------
Some filtering is not curation and cannot be hoisted above the seam:

* a parameter type having no representation in the target's type registry
  (``TypeNotFoundError``), and
* a mangled name colliding inside the target's namespace
  (``MangledFunctionNameConflictError``).

Both are properties of the *emission target*, not of the declaration, so they
stay in the backend.
"""

from dataclasses import dataclass, field
from typing import Any

__all__ = ["FunctionPlan"]


@dataclass(frozen=True)
class FunctionPlan:
    """A function declaration plus the curation decisions made about it."""

    decl: Any
    """The ``ast_canopy.decl.Function`` this plan describes."""

    exposed_name: str
    """Public name after ``API Prefix Removal`` -- what callers will see."""

    use_cooperative: bool
    """True when the name matched ``Cooperative Launch Required Functions Regex``."""

    argument_intents: dict = field(default_factory=dict)
    """The full ``Function Argument Intents`` mapping, as configured."""

    @property
    def c_symbol(self) -> str:
        """The original name as spelled in the header."""
        return self.decl.name
